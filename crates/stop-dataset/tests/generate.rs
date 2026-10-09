//! Generator pipeline tests over wiremock-canned OpenRouter responses:
//! transcript step, room-state prediction step, JSONL output shape.

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::{Value, json};
use stop_core::RoomState;
use stop_dataset::error::DatasetError;
use stop_dataset::generator::{Generator, GeneratorConfig};
use stop_dataset::openrouter::OpenRouterClient;
use wiremock::matchers::method;
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

/// Matches POST bodies containing the given substring (verifies the single
/// combined generation prompt).
struct BodyContains(&'static str);

impl wiremock::Match for BodyContains {
    fn matches(&self, request: &Request) -> bool {
        String::from_utf8_lossy(&request.body).contains(self.0)
    }
}

/// Serves a canned `chat/completions` payload for the matched body marker.
struct Completion(Value);

impl Respond for Completion {
    fn respond(&self, _request: &Request) -> ResponseTemplate {
        ResponseTemplate::new(200).set_body_json(json!({
            "choices": [ { "message": { "role": "assistant", "content": self.0.to_string() } } ]
        }))
    }
}

fn state_json(mutate: impl FnOnce(&mut RoomState)) -> Value {
    let mut state = RoomState::default();
    mutate(&mut state);
    serde_json::to_value(state).expect("state serializes")
}

fn transcript_payload() -> Value {
    json!({
        "utterances": [
            { "index": 1, "text": "uhm nurse uh could you dim the overhead lights by two steps" },
            { "index": 2, "text": "did you sleep okay by the way" },
            { "index": 3, "text": "and bring the endoscope a little closer" },
            { "index": 4, "text": "wait no uh actually dim it less" }
        ]
    })
}

/// Expected states per utterance: dim -2, noise unchanged, zoom to 3,
/// self-correction to brightness 79.
fn states_payload() -> Value {
    let dimmed = state_json(|s| s.lighting.primary_intensity_pct = 78);
    let zoomed = state_json(|s| {
        s.lighting.primary_intensity_pct = 78;
        s.endoscope.zoom_level = 3;
    });
    let corrected = state_json(|s| {
        s.lighting.primary_intensity_pct = 79;
        s.endoscope.zoom_level = 3;
    });
    json!({
        "states": [
            { "index": 1, "room_state": dimmed },
            { "index": 2, "room_state": dimmed },
            { "index": 3, "room_state": zoomed },
            { "index": 4, "room_state": corrected }
        ]
    })
}

/// Combined single-call payload: transcript utterances plus chained states.
fn case_payload(mut payload: Value, states: Value) -> Value {
    let object = payload.as_object_mut().expect("object");
    for (key, value) in states.as_object().expect("object") {
        object.insert(key.clone(), value.clone());
    }
    payload
}

async fn mount_pipeline(server: &MockServer, payload: Value) {
    Mock::given(method("POST"))
        .and(BodyContains("speech-to-text transcripts"))
        .respond_with(Completion(payload))
        .mount(server)
        .await;
}

async fn generator(server: &MockServer) -> Generator {
    Generator::new(
        OpenRouterClient::new(server.uri(), "test-key", "test-model"),
        GeneratorConfig {
            utterances_per_case: 4,
            noise_ratio: 0.5,
        },
    )
}

#[tokio::test]
async fn generate_case_produces_expected_states() {
    let server = MockServer::start().await;
    mount_pipeline(
        &server,
        case_payload(transcript_payload(), states_payload()),
    )
    .await;
    let generator = generator(&server).await;
    let mut rng = StdRng::seed_from_u64(3);

    let case = generator
        .generate_case(&mut rng, "case_000", "cholecystectomy")
        .await
        .expect("case");

    assert_eq!(case.id, "case_000");
    assert_eq!(case.scenario, "cholecystectomy");
    assert_eq!(case.model, "test-model");
    assert_eq!(case.history.len(), 4);

    assert!(case.history[0].raw_utterance.contains("dim"));
    assert_eq!(
        case.history[0]
            .expected_output_state
            .lighting
            .primary_intensity_pct,
        78
    );
    // Noise entry: state unchanged.
    assert_eq!(
        case.history[1].expected_output_state,
        case.history[0].expected_output_state
    );
    assert_eq!(
        case.history[2].expected_output_state.endoscope.zoom_level,
        3
    );
    assert_eq!(
        case.history[3]
            .expected_output_state
            .lighting
            .primary_intensity_pct,
        79
    );
}

#[tokio::test]
async fn generate_case_clamps_predicted_states() {
    let server = MockServer::start().await;
    let out_of_range = state_json(|s| {
        s.insufflator.target_pressure_mmhg = 40;
        s.table.tilt_degrees = -30;
    });
    let states = json!({
        "states": [
            { "index": 1, "room_state": out_of_range },
            { "index": 2, "room_state": out_of_range },
            { "index": 3, "room_state": out_of_range },
            { "index": 4, "room_state": out_of_range }
        ]
    });
    mount_pipeline(&server, case_payload(transcript_payload(), states)).await;
    let generator = generator(&server).await;
    let mut rng = StdRng::seed_from_u64(3);

    let case = generator
        .generate_case(&mut rng, "case_000", "hernia_repair")
        .await
        .expect("case");

    for entry in &case.history {
        assert_eq!(
            entry.expected_output_state.insufflator.target_pressure_mmhg,
            25
        );
        assert_eq!(entry.expected_output_state.table.tilt_degrees, -15);
    }
}

#[tokio::test]
async fn generate_case_output_is_jsonl_ready() {
    let server = MockServer::start().await;
    mount_pipeline(
        &server,
        case_payload(transcript_payload(), states_payload()),
    )
    .await;
    let generator = generator(&server).await;
    let mut rng = StdRng::seed_from_u64(3);

    let case = generator
        .generate_case(&mut rng, "case_001", "hernia_repair")
        .await
        .expect("case");

    let line = serde_json::to_string(&case).expect("serialize line");
    assert!(!line.contains('\n'));
    let value: Value = serde_json::from_str(&line).expect("line is JSON");
    assert_eq!(value["model"], json!("test-model"));
    assert_eq!(value["scenario"], json!("hernia_repair"));
    assert!(value["initial_state"]["lighting"].is_object());
    let entry_object = value["history"][0].as_object().expect("entry object");
    assert_eq!(entry_object.len(), 2);
    assert!(entry_object.contains_key("raw_utterance"));
    assert!(entry_object.contains_key("expected_output_state"));
}

#[tokio::test]
async fn generate_case_sends_exactly_one_request() {
    let server = MockServer::start().await;
    mount_pipeline(
        &server,
        case_payload(transcript_payload(), states_payload()),
    )
    .await;
    let generator = generator(&server).await;
    let mut rng = StdRng::seed_from_u64(3);

    generator
        .generate_case(&mut rng, "case_010", "cholecystectomy")
        .await
        .expect("case");

    // Roundtrip optimization: transcript + states in a single completion call.
    let requests = server.received_requests().await.expect("request log");
    assert_eq!(requests.len(), 1, "one request per case expected");
}

#[tokio::test]
async fn transcript_length_mismatch_is_malformed() {
    let server = MockServer::start().await;
    let short = json!({ "utterances": [{ "index": 1, "text": "only one" }] });
    mount_pipeline(&server, case_payload(short, states_payload())).await;
    let generator = generator(&server).await;
    let mut rng = StdRng::seed_from_u64(3);

    let error = generator
        .generate_case(&mut rng, "case_002", "appendectomy")
        .await
        .expect_err("must fail");

    assert!(matches!(error, DatasetError::Malformed(_)), "got {error:?}");
}

#[tokio::test]
async fn state_out_of_order_index_is_malformed() {
    let server = MockServer::start().await;
    let mut states = states_payload();
    states["states"][1]["index"] = json!(9);
    mount_pipeline(&server, case_payload(transcript_payload(), states)).await;
    let generator = generator(&server).await;
    let mut rng = StdRng::seed_from_u64(3);

    let error = generator
        .generate_case(&mut rng, "case_003", "appendectomy")
        .await
        .expect_err("must fail");

    assert!(matches!(error, DatasetError::Malformed(_)), "got {error:?}");
}

#[tokio::test]
async fn http_error_maps_to_status() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(BodyContains("speech-to-text transcripts"))
        .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
        .mount(&server)
        .await;
    let generator = generator(&server).await;
    let mut rng = StdRng::seed_from_u64(3);

    let error = generator
        .generate_case(&mut rng, "case_004", "appendectomy")
        .await
        .expect_err("must fail");

    assert!(
        matches!(error, DatasetError::Status { .. }),
        "got {error:?}"
    );
}
