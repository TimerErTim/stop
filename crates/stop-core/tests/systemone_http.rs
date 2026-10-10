//! HTTP client tests: wiremock-canned `/v1/systemone` responses, request
//! shape, per-setting decode rules (`null` = leave as is, choice-encoded
//! operands, noul toggles), error mapping, timeout mapping, and one
//! `#[ignore]`d live test.

use std::time::Duration;

use serde_json::{Value, json};
use stop_core::systemone::question_catalog_json;
use stop_core::{InferenceInput, InferencePort, LightMode, ProviderError, RoomState, ValueChange};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn canned_answers() -> Value {
    json!({
        "answers": {
            "emergency_stop": { "type": "noul", "noul": 0.1 },
            "requires_sterile_confirm": { "type": "noul", "noul": 0.2 },
            "light_brightness": {
                "type": "choice",
                "choice": "SetBrightness:60",
                "probabilities": { "null": 0.05, "SetBrightness:60": 0.85, "DecreaseBrightness": 0.1 },
                "confidence": 0.85
            },
            "light_mode": {
                "type": "choice",
                "choice": "AmbientRed",
                "probabilities": { "null": 0.1, "Normal": 0.05, "CavityFocus": 0.05, "AmbientRed": 0.8 },
                "confidence": 0.8
            },
            "camera_zoom": {
                "type": "choice",
                "choice": "null",
                "probabilities": { "null": 0.7, "ZoomIn": 0.2, "ZoomOut": 0.1 },
                "confidence": 0.7
            },
            "camera_irrigation": { "type": "noul", "noul": 0.8, "confidence": 0.8 },
            "insufflator_pressure": {
                "type": "choice",
                "choice": "SetPressure:14",
                "probabilities": { "null": 0.1, "SetPressure:14": 0.85, "DecreasePressure": 0.05 },
                "confidence": 0.85
            },
            "insufflator_active": { "type": "noul", "noul": 0.1, "confidence": 0.9 },
            "table_tilt": {
                "type": "choice",
                "choice": "null",
                "probabilities": { "null": 0.9, "SetTilt:8": 0.05, "SetHeight:90": 0.05 },
                "confidence": 0.9
            },
            "table_height": {
                "type": "choice",
                "choice": "SetHeight:90",
                "probabilities": { "null": 0.1, "SetHeight:90": 0.85 },
                "confidence": 0.85
            },
        },
        "elapsed_ms": 21.5
    })
}

fn input<'a>(state: &'a RoomState) -> InferenceInput<'a> {
    InferenceInput {
        room_state: state,
        utterance: "dim the light",
    }
}

async fn mount_answers(
    server: &MockServer,
    answers: Value,
) -> stop_core::systemone::SystemOneClient {
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(answers))
        .expect(1..)
        .mount(server)
        .await;
    stop_core::systemone::SystemOneClient::new(server.uri(), "test-model")
}

#[tokio::test]
async fn decodes_canned_response_into_explicit_decisions() {
    let server = MockServer::start().await;
    let client = mount_answers(&server, canned_answers()).await;

    let state = RoomState::default();
    let outcome = client.single_pass(&input(&state)).await.expect("decode");

    assert!(!outcome.decision.emergency_stop);
    assert!(!outcome.decision.requires_sterile_confirm);

    // Light: brightness and mode are independent settings.
    assert_eq!(
        outcome.decision.light.brightness,
        Some(ValueChange::Absolute(60))
    );
    assert_eq!(
        outcome.decision.light.field_mode,
        Some(LightMode::AmbientRed)
    );

    // Camera: zoom unchanged, irrigation toggled on.
    assert_eq!(outcome.decision.camera.zoom, None);
    assert!(outcome.decision.camera.toggle_irrigation);

    // Insufflator: pressure set, insufflation unchanged.
    assert_eq!(
        outcome.decision.insufflator.pressure,
        Some(ValueChange::Absolute(14))
    );
    assert!(!outcome.decision.insufflator.toggle_insufflation);

    // Table: tilt unchanged, height set — both can coexist.
    assert_eq!(outcome.decision.table.tilt, None);
    assert_eq!(
        outcome.decision.table.height,
        Some(ValueChange::Absolute(90))
    );

    // Latency comes from the server's `elapsed_ms`, not wall-clock.
    assert_eq!(outcome.latency, Duration::from_micros(21_500));
}

#[tokio::test]
async fn json_null_choice_decodes_to_leave_as_is() {
    let server = MockServer::start().await;
    let mut answers = canned_answers();
    answers["answers"]["light_brightness"]["choice"] = Value::Null;
    let client = mount_answers(&server, answers).await;

    let state = RoomState::default();
    let outcome = client.single_pass(&input(&state)).await.expect("decode");

    assert_eq!(outcome.decision.light.brightness, None);
}

#[tokio::test]
async fn relative_choice_without_operand_defaults_to_one_step() {
    let server = MockServer::start().await;
    let mut answers = canned_answers();
    answers["answers"]["light_brightness"]["choice"] = json!("DecreaseBrightness");
    let client = mount_answers(&server, answers).await;

    let state = RoomState::default();
    let outcome = client.single_pass(&input(&state)).await.expect("decode");

    assert_eq!(
        outcome.decision.light.brightness,
        Some(ValueChange::Decrease(1))
    );
}

#[tokio::test]
async fn low_confidence_setting_is_leave_as_is() {
    let server = MockServer::start().await;
    let mut answers = canned_answers();
    answers["answers"]["light_brightness"]["confidence"] = json!(0.3);
    let client = mount_answers(&server, answers).await;

    let state = RoomState::default();
    let outcome = client.single_pass(&input(&state)).await.expect("decode");

    assert_eq!(outcome.decision.light.brightness, None);
}

#[tokio::test]
async fn absolute_choice_without_operand_maps_to_inconsistent_slots() {
    let server = MockServer::start().await;
    let mut answers = canned_answers();
    answers["answers"]["light_brightness"]["choice"] = json!("SetBrightness");
    let client = mount_answers(&server, answers).await;

    let state = RoomState::default();
    let error = client
        .single_pass(&input(&state))
        .await
        .expect_err("SetBrightness needs a value");

    assert!(
        matches!(error, ProviderError::InconsistentSlots(_)),
        "unexpected error: {error:?}"
    );
}

#[tokio::test]
async fn request_carries_ten_questions_and_minimal_state() {
    let server = MockServer::start().await;
    let client = mount_answers(&server, canned_answers()).await;

    let state = RoomState::default();
    let _ = client.single_pass(&input(&state)).await.expect("decode");

    let requests = server.received_requests().await.expect("request log");
    assert_eq!(requests.len(), 1);
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body is JSON");
    assert_eq!(body["model"], json!("test-model"));
    let questions = body["questions"].as_object().expect("questions object");
    assert_eq!(
        questions.len(),
        10,
        "expected 10 mapped questions: {questions:?}"
    );
    for name in [
        "emergency_stop",
        "requires_sterile_confirm",
        "light_brightness",
        "light_mode",
        "camera_zoom",
        "camera_irrigation",
        "insufflator_pressure",
        "insufflator_active",
        "table_tilt",
        "table_height",
    ] {
        assert!(questions.contains_key(name), "missing question {name}");
    }
    // Token minimization: room state + utterance only, no pass history.
    assert_eq!(body["state"]["utterance"], json!("dim the light"));
    assert!(body["state"]["room_state"]["lighting"].is_object());
    assert!(body["state"].get("history").is_none());
}

#[tokio::test]
async fn request_offers_uttered_numbers_as_choice_options() {
    let server = MockServer::start().await;
    let client = mount_answers(&server, canned_answers()).await;

    let state = RoomState::default();
    let numeric = InferenceInput {
        room_state: &state,
        utterance: "bump the light to eighty-five percent and raise the table by 12",
    };
    let _ = client.single_pass(&numeric).await.expect("decode");

    let requests = server.received_requests().await.expect("request log");
    let body: Value = serde_json::from_slice(&requests[0].body).expect("request body is JSON");
    let brightness = &body["questions"]["light_brightness"]["criteria"];
    assert_eq!(
        brightness["SetBrightness:85"],
        json!("Set brightness to 85 %")
    );
    assert_eq!(
        brightness["IncreaseBrightness:85"],
        json!("Increase brightness by 85 %")
    );
    let height = &body["questions"]["table_height"]["criteria"];
    assert_eq!(height["SetHeight:12"], json!("Set table height to 12 cm"));
    assert_eq!(
        height["IncreaseHeight:12"],
        json!("Increase table height by 12 cm")
    );
    // Generic fallbacks remain so a direction-only utterance still resolves.
    assert!(brightness.get("IncreaseBrightness").is_some());
}

#[tokio::test]
async fn question_catalog_covers_all_settings() {
    let catalog = question_catalog_json(&[8]);
    let questions = catalog.as_object().expect("catalog object");
    assert_eq!(questions.len(), 10);
    // Typed question forms match the System-One contract.
    assert_eq!(questions["emergency_stop"]["type"], json!("noul"));
    assert_eq!(questions["light_brightness"]["type"], json!("choice"));
    assert_eq!(questions["camera_irrigation"]["type"], json!("noul"));
    // Every choice offers the `null` leave-as-is option.
    for choice in [
        "light_brightness",
        "light_mode",
        "camera_zoom",
        "insufflator_pressure",
        "table_tilt",
        "table_height",
    ] {
        assert!(
            questions[choice]["criteria"].get("null").is_some(),
            "{choice} lacks null option"
        );
    }
    // Table tilt and height are separate questions.
    assert!(
        questions["table_tilt"]["criteria"]
            .get("SetTilt:8")
            .is_some()
    );
    assert!(
        questions["table_height"]["criteria"]
            .get("SetHeight:8")
            .is_some()
    );
}

#[tokio::test]
async fn http_error_status_maps_to_provider_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(401).set_body_string("unauthorized"))
        .mount(&server)
        .await;
    let client = stop_core::systemone::SystemOneClient::new(server.uri(), "test-model");

    let state = RoomState::default();
    let error = client.single_pass(&input(&state)).await.expect_err("401");

    match error {
        ProviderError::Status { status, body } => {
            assert_eq!(status, 401);
            assert!(body.contains("unauthorized"), "body: {body}");
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn malformed_response_maps_to_provider_error() {
    let server = MockServer::start().await;
    // Valid JSON but missing `answers`.
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({ "model": "x" })))
        .mount(&server)
        .await;
    let client = stop_core::systemone::SystemOneClient::new(server.uri(), "test-model");

    let state = RoomState::default();
    let error = client
        .single_pass(&input(&state))
        .await
        .expect_err("no answers");

    assert!(
        matches!(error, ProviderError::Malformed(_)),
        "unexpected error: {error:?}"
    );
}

#[tokio::test]
async fn unknown_choice_option_maps_to_inconsistent_slots() {
    let server = MockServer::start().await;
    let mut answers = canned_answers();
    answers["answers"]["light_brightness"]["choice"] = json!("Teleporter");
    let client = mount_answers(&server, answers).await;

    let state = RoomState::default();
    let error = client
        .single_pass(&input(&state))
        .await
        .expect_err("bad choice");

    assert!(
        matches!(error, ProviderError::InconsistentSlots(_)),
        "unexpected error: {error:?}"
    );
}

#[tokio::test]
async fn request_timeout_maps_to_timeout_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(canned_answers())
                .set_delay(Duration::from_millis(500)),
        )
        .mount(&server)
        .await;
    let client = stop_core::systemone::SystemOneClient::new(server.uri(), "test-model")
        .with_request_timeout(Duration::from_millis(50));

    let state = RoomState::default();
    let error = client
        .single_pass(&input(&state))
        .await
        .expect_err("must time out");

    assert!(
        matches!(error, ProviderError::Timeout(_)),
        "unexpected error: {error:?}"
    );
}

/// Live check against `SYSTEMONE_API_BASE_URL`; ignored by default (needs a
/// running JevK5 / System-One instance). Run with:
/// `cargo nextest run -p stop-core -- live_ --ignored --nocapture`
#[tokio::test]
#[ignore = "needs a live SYSTEMONE_API_BASE_URL instance"]
async fn live_systemone_round_trip() {
    let client = stop_core::systemone::SystemOneClient::from_env().expect("SYSTEMONE_API_BASE_URL");
    let state = RoomState::default();
    let input = input(&state);

    let outcome = client.single_pass(&input).await.expect("live round trip");

    eprintln!("live input: {:?}", input);
    eprintln!(
        "live decision: {:?} latency {:?}",
        outcome.decision, outcome.latency
    );
}
