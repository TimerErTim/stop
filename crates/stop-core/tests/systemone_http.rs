//! HTTP client tests: wiremock-canned `/v1/systemone` responses, request
//! shape, per-object decode rules (`null` = no change, optional targets),
//! error mapping, timeout mapping, and one `#[ignore]`d live test.

use std::time::Duration;

use serde_json::{Value, json};
use stop_core::systemone::question_catalog_json;
use stop_core::{
    ActionKind, InferenceInput, InferencePort, ProviderError, RoomState, TargetDevice,
};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn canned_answers() -> Value {
    json!({
        "answers": {
            "emergency_stop": { "type": "noul", "noul": 0.1 },
            "requires_sterile_confirm": { "type": "noul", "noul": 0.2 },
            "light_action": {
                "type": "choice",
                "choice": "DecreaseBrightness",
                "probabilities": { "null": 0.05, "DecreaseBrightness": 0.85, "SetLightMode": 0.1 },
                "confidence": 0.85
            },
            "brightness_target": {
                "type": "score",
                "score": 6.0,
                "probabilities": { "0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.02, "6": 0.9, "7": 0.05, "8": 0.03, "9": 0.0 },
                "confidence": 0.9
            },
            "light_mode_target": {
                "type": "choice",
                "choice": "Normal",
                "probabilities": { "Normal": 0.9, "CavityFocus": 0.05, "AmbientRed": 0.05 },
                "confidence": 0.9
            },
            "camera_action": {
                "type": "choice",
                "choice": "null",
                "probabilities": { "null": 0.7, "ZoomIn": 0.2, "ZoomOut": 0.1 },
                "confidence": 0.7
            },
            "zoom_target": {
                "type": "score",
                "score": 1.0,
                "probabilities": { "0": 0.1, "1": 0.6, "2": 0.2, "3": 0.05, "4": 0.05 },
                "confidence": 0.6
            },
            "insufflator_action": {
                "type": "choice",
                "choice": "AdjustPressure",
                "probabilities": { "null": 0.1, "AdjustPressure": 0.85, "ToggleInsufflation": 0.05 },
                "confidence": 0.85
            },
            "pressure_target": {
                "type": "score",
                "score": 5.0,
                "probabilities": { "0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 0.85, "6": 0.05, "7": 0.05, "8": 0.03, "9": 0.02 },
                "confidence": 0.85
            },
            "table_action": {
                "type": "choice",
                "choice": "null",
                "probabilities": { "null": 0.9, "TiltTable": 0.05, "SetTableHeight": 0.05 },
                "confidence": 0.9
            },
            "tilt_target": {
                "type": "score",
                "score": 4.0,
                "probabilities": { "0": 0.02, "1": 0.03, "2": 0.05, "3": 0.1, "4": 0.7, "5": 0.06, "6": 0.03, "7": 0.01, "8": 0.0 },
                "confidence": 0.7
            },
            "height_target": {
                "type": "score",
                "score": 3.0,
                "probabilities": { "0": 0.02, "1": 0.03, "2": 0.1, "3": 0.75, "4": 0.05, "5": 0.03, "6": 0.02 },
                "confidence": 0.75
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

fn device_at(
    outcome: &stop_core::InferenceOutcome,
    device: TargetDevice,
) -> &stop_core::DeviceDecision {
    outcome
        .decision
        .devices
        .iter()
        .find(|d| d.target_device == device)
        .expect("device group present")
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
async fn decodes_canned_response_into_device_decisions() {
    let server = MockServer::start().await;
    let client = mount_answers(&server, canned_answers()).await;

    let state = RoomState::default();
    let outcome = client.single_pass(&input(&state)).await.expect("decode");

    assert!(!outcome.decision.emergency_stop);
    assert!(!outcome.decision.requires_sterile_confirm);
    assert_eq!(outcome.decision.devices.len(), 4);

    // Light: brightness action takes the absolute-target answer (level 6 = 60%).
    let light = device_at(&outcome, TargetDevice::SurgicalLight);
    assert_eq!(light.action, Some(ActionKind::DecreaseBrightness));
    assert_eq!(light.absolute, Some(60));
    assert_eq!(light.confidence, 0.85);
    assert_eq!(light.absolute_confidence, 0.9);

    // Camera / table: model chose `null` = no change.
    assert_eq!(
        device_at(&outcome, TargetDevice::EndoscopeCamera).action,
        None
    );
    assert_eq!(
        device_at(&outcome, TargetDevice::OperatingTable).action,
        None
    );

    // Insufflator: pressure action with target (level 5 = 14 mmHg).
    let insufflator = device_at(&outcome, TargetDevice::Insufflator);
    assert_eq!(insufflator.action, Some(ActionKind::AdjustPressure));
    assert_eq!(insufflator.absolute, Some(14));

    // Latency comes from the server's `elapsed_ms`, not wall-clock.
    assert_eq!(outcome.latency, Duration::from_micros(21_500));
}

#[tokio::test]
async fn json_null_choice_decodes_to_no_change() {
    let server = MockServer::start().await;
    let mut answers = canned_answers();
    answers["answers"]["light_action"]["choice"] = Value::Null;
    let client = mount_answers(&server, answers).await;

    let state = RoomState::default();
    let outcome = client.single_pass(&input(&state)).await.expect("decode");

    assert_eq!(
        device_at(&outcome, TargetDevice::SurgicalLight).action,
        None
    );
}

#[tokio::test]
async fn toggle_action_carries_no_absolute_target() {
    let server = MockServer::start().await;
    let mut answers = canned_answers();
    answers["answers"]["camera_action"]["choice"] = json!("ToggleIrrigation");
    let client = mount_answers(&server, answers).await;

    let state = RoomState::default();
    let outcome = client.single_pass(&input(&state)).await.expect("decode");

    let camera = device_at(&outcome, TargetDevice::EndoscopeCamera);
    assert_eq!(camera.action, Some(ActionKind::ToggleIrrigation));
    assert_eq!(camera.absolute, None);
}

#[tokio::test]
async fn missing_target_answer_degrades_to_no_absolute() {
    let server = MockServer::start().await;
    let mut answers = canned_answers();
    answers["answers"]
        .as_object_mut()
        .expect("answers object")
        .remove("brightness_target");
    let client = mount_answers(&server, answers).await;

    let state = RoomState::default();
    let outcome = client.single_pass(&input(&state)).await.expect("decode");

    let light = device_at(&outcome, TargetDevice::SurgicalLight);
    assert_eq!(light.action, Some(ActionKind::DecreaseBrightness));
    assert_eq!(light.absolute, None);
    assert_eq!(light.absolute_confidence, 0.0);
}

#[tokio::test]
async fn request_carries_twelve_questions_and_minimal_state() {
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
        12,
        "expected 12 mapped questions: {questions:?}"
    );
    for name in [
        "emergency_stop",
        "requires_sterile_confirm",
        "light_action",
        "brightness_target",
        "light_mode_target",
        "camera_action",
        "zoom_target",
        "insufflator_action",
        "pressure_target",
        "table_action",
        "tilt_target",
        "height_target",
    ] {
        assert!(questions.contains_key(name), "missing question {name}");
    }
    // Token minimization: room state + utterance only, no pass history.
    assert_eq!(body["state"]["utterance"], json!("dim the light"));
    assert!(body["state"]["room_state"]["lighting"].is_object());
    assert!(body["state"].get("history").is_none());
}

#[tokio::test]
async fn question_catalog_covers_all_object_groups() {
    let catalog = question_catalog_json();
    let questions = catalog.as_object().expect("catalog object");
    assert_eq!(questions.len(), 12);
    // Typed question forms match the System-One contract.
    assert_eq!(questions["emergency_stop"]["type"], json!("noul"));
    assert_eq!(questions["light_action"]["type"], json!("choice"));
    assert_eq!(questions["brightness_target"]["type"], json!("score"));
    // Every object group offers the `null` no-change option.
    for group in [
        "light_action",
        "camera_action",
        "insufflator_action",
        "table_action",
    ] {
        let criteria = questions[group]["criteria"].as_object().expect("criteria");
        assert!(criteria.contains_key("null"), "{group} lacks null option");
    }
    // Closed object set: exactly four groups, table has height too.
    assert!(
        questions["table_action"]["criteria"]
            .as_object()
            .expect("criteria")
            .contains_key("SetTableHeight")
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
    answers["answers"]["light_action"]["choice"] = json!("Teleporter");
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
    assert!(!format!("{:?}", outcome.decision.devices).is_empty());
}
