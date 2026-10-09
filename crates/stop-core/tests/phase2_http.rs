//! HTTP client tests: wiremock-canned `/v1/systemone` responses, request
//! shape, decode rules, error mapping, and one `#[ignore]`d live test.

use std::time::Duration;

use serde_json::{Value, json};
use stop_core::systemone::question_catalog_json;
use stop_core::{
    ActionKind, InferenceInput, InferencePort, ProviderError, RoomState, StepValue, TargetDevice,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn canned_answers() -> Value {
    json!({
        "answers": {
            "further_action": { "type": "noul", "noul": 0.9 },
            "requires_sterile_confirm": { "type": "noul", "noul": 0.2 },
            "target_device": {
                "type": "choice",
                "choice": "SurgicalLight",
                "probabilities": { "None": 0.05, "SurgicalLight": 0.8, "EndoscopeCamera": 0.1, "Insufflator": 0.03, "OperatingTable": 0.02 },
                "confidence": 0.8
            },
            "action_kind": {
                "type": "choice",
                "choice": "DecreaseBrightness",
                "probabilities": { "Idle": 0.0, "DecreaseBrightness": 0.85, "ZoomIn": 0.05 },
                "confidence": 0.85
            },
            "step_value": {
                "type": "score",
                "score": 0.0,
                "probabilities": { "0": 0.1, "1": 0.2, "2": 0.5, "3": 0.15, "4": 0.05 },
                "confidence": 0.5
            },
            "brightness_target": {
                "type": "score",
                "score": 6.0,
                "probabilities": { "0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.02, "6": 0.9, "7": 0.05, "8": 0.03, "9": 0.0 },
                "confidence": 0.9
            },
            "zoom_target": {
                "type": "score",
                "score": 1.0,
                "probabilities": { "0": 0.1, "1": 0.6, "2": 0.2, "3": 0.05, "4": 0.05 },
                "confidence": 0.6
            },
            "pressure_target": {
                "type": "score",
                "score": 5.0,
                "probabilities": { "0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.85, "6": 0.05, "7": 0.05, "8": 0.03, "9": 0.02 },
                "confidence": 0.85
            },
            "tilt_target": {
                "type": "score",
                "score": 4.0,
                "probabilities": { "0": 0.02, "1": 0.03, "2": 0.05, "3": 0.1, "4": 0.7, "5": 0.06, "6": 0.03, "7": 0.01, "8": 0.0 },
                "confidence": 0.7
            },
            "light_mode_target": {
                "type": "choice",
                "choice": "Normal",
                "probabilities": { "Normal": 0.9, "CavityFocus": 0.05, "AmbientRed": 0.05 },
                "confidence": 0.9
            },
        },
        "elapsed_ms": 21.5
    })
}

fn input<'a>(state: &'a RoomState) -> InferenceInput<'a> {
    InferenceInput {
        room_state: state,
        utterance: "dim the light",
        history: &[],
    }
}

async fn spawn_client(server: &MockServer) -> stop_core::systemone::SystemOneClient {
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_answers()))
        .expect(1..)
        .mount(server)
        .await;
    stop_core::systemone::SystemOneClient::new(server.uri(), "test-model")
}

#[tokio::test]
async fn decodes_canned_response_into_action_decision() {
    let server = MockServer::start().await;
    let client = spawn_client(&server).await;

    let state = RoomState::default();
    let outcome = client.single_pass(&input(&state)).await.expect("decode");

    let decision = &outcome.decision;
    assert!(decision.further_action_needed);
    assert!(!decision.requires_sterile_confirm);
    assert_eq!(decision.target_device, TargetDevice::SurgicalLight);
    assert_eq!(decision.action_kind, ActionKind::DecreaseBrightness);
    // Brightness action takes the absolute-target answer (level 6 = 60%).
    assert_eq!(decision.step_value, StepValue::AbsoluteValue(60));
    // Latency comes from the server's `elapsed_ms`, not wall-clock.
    assert_eq!(outcome.latency, Duration::from_micros(21_500));
    assert_eq!(outcome.slot_confidences.target_device, 0.8);
    assert_eq!(outcome.slot_confidences.further_action_needed, 0.9);
}

#[tokio::test]
async fn relative_step_used_when_action_has_no_absolute_target() {
    let server = MockServer::start().await;
    // ToggleIrrigation has no absolute target: relative step (index 2 = 0).
    let mut answers = canned_answers();
    answers["answers"]["action_kind"]["choice"] = json!("ToggleIrrigation");
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(answers))
        .mount(&server)
        .await;
    let client = stop_core::systemone::SystemOneClient::new(server.uri(), "test-model");

    let state = RoomState::default();
    let outcome = client.single_pass(&input(&state)).await.expect("decode");

    assert_eq!(outcome.decision.action_kind, ActionKind::ToggleIrrigation);
    assert_eq!(outcome.decision.step_value, StepValue::Zero);
    assert_eq!(
        outcome.slot_confidences.absolute_target,
        outcome.slot_confidences.step_value
    );
}

#[tokio::test]
async fn request_carries_all_ten_questions_and_state() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(canned_answers()))
        .mount(&server)
        .await;
    let client = stop_core::systemone::SystemOneClient::new(server.uri(), "test-model");

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
        "further_action",
        "requires_sterile_confirm",
        "target_device",
        "action_kind",
        "step_value",
        "brightness_target",
        "zoom_target",
        "pressure_target",
        "tilt_target",
        "light_mode_target",
    ] {
        assert!(questions.contains_key(name), "missing question {name}");
    }
    assert_eq!(body["state"]["utterance"], json!("dim the light"));
    assert!(body["state"]["room_state"]["lighting"].is_object());
    assert!(body["state"]["history"].is_array());
}

#[tokio::test]
async fn question_catalog_covers_all_slots() {
    let catalog = question_catalog_json();
    let questions = catalog.as_object().expect("catalog object");
    assert_eq!(questions.len(), 10);
    // Typed question forms match the System-One contract.
    assert_eq!(questions["further_action"]["type"], json!("noul"));
    assert_eq!(questions["target_device"]["type"], json!("choice"));
    assert_eq!(questions["step_value"]["type"], json!("score"));
    // Choice criteria carry all target devices.
    let devices = questions["target_device"]["criteria"]
        .as_object()
        .expect("criteria");
    assert_eq!(devices.len(), 5);
    let actions = questions["action_kind"]["criteria"]
        .as_object()
        .expect("criteria");
    assert_eq!(actions.len(), 11);
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
    answers["answers"]["target_device"]["choice"] = json!("Teleporter");
    Mock::given(method("POST"))
        .and(path("/v1/systemone"))
        .respond_with(ResponseTemplate::new(200).set_body_json(answers))
        .mount(&server)
        .await;
    let client = stop_core::systemone::SystemOneClient::new(server.uri(), "test-model");

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

/// Live check against `SYSTEMONE_API_BASE_URL`; ignored by default (needs a
/// running JevK5 / System-One instance). Run with:
/// `cargo nextest run -p stop-core -- live_ --ignored --nocapture`
#[tokio::test]
#[ignore = "needs a live SYSTEMONE_API_BASE_URL instance"]
async fn live_systemone_round_trip() {
    let client = stop_core::systemone::SystemOneClient::from_env().expect("SYSTEMONE_API_BASE_URL");
    let state = RoomState::default();
    let input = input(&state);

    let outcome = client
        .single_pass(&input)
        .await
        .expect("live round trip");

    eprintln!("live input: {:?}", input);
    eprintln!(
        "live decision: {:?} latency {:?}",
        outcome.decision, outcome.latency
    );
    assert!(!format!("{:?}", outcome.decision.target_device).is_empty());
}
