//! Single-pass executor tests: scripted mock engine, exactly-one-call
//! invariant, per-object null semantics, confidence gating, emergency
//! stop precedence, error propagation.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Duration;

use stop_core::{
    ActionKind, DeviceDecision, ExecutionError, InferenceInput, InferenceOutcome, InferencePort,
    ProviderError, RoomState, SinglePassExecutor, TargetDevice, UtteranceDecision,
};

/// Recorded observation of one `single_pass` call.
#[derive(Debug, Clone)]
struct PassInput {
    utterance: String,
    room_state: RoomState,
}

type Recording = Rc<RefCell<Vec<PassInput>>>;

/// Tests-only mock: replays a scripted sequence of results and records
/// every input it receives.
struct MockDecisionEngine {
    script: RefCell<VecDeque<Result<UtteranceDecision, ProviderError>>>,
    recording: Recording,
}

impl MockDecisionEngine {
    fn new<I>(script: I) -> Self
    where
        I: IntoIterator<Item = Result<UtteranceDecision, ProviderError>>,
    {
        Self {
            script: RefCell::new(script.into_iter().collect()),
            recording: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn from_decisions<I>(decisions: I) -> Self
    where
        I: IntoIterator<Item = UtteranceDecision>,
    {
        Self::new(decisions.into_iter().map(Ok))
    }

    fn recording(&self) -> Recording {
        self.recording.clone()
    }
}

impl InferencePort for MockDecisionEngine {
    async fn single_pass(
        &self,
        input: &InferenceInput<'_>,
    ) -> Result<InferenceOutcome, ProviderError> {
        self.recording.borrow_mut().push(PassInput {
            utterance: input.utterance.to_string(),
            room_state: input.room_state.clone(),
        });

        let mut script = self.script.borrow_mut();
        let item = script
            .pop_front()
            .unwrap_or_else(|| Ok(decision(vec![], false, false)));
        let decision = item?;

        Ok(InferenceOutcome {
            decision,
            latency: Duration::from_millis(1),
        })
    }
}

fn device(
    target_device: TargetDevice,
    action: Option<ActionKind>,
    absolute: Option<i16>,
) -> DeviceDecision {
    DeviceDecision {
        target_device,
        action,
        absolute,
        confidence: 0.9,
        absolute_confidence: 0.9,
    }
}

fn decision(
    devices: Vec<DeviceDecision>,
    emergency_stop: bool,
    requires_sterile_confirm: bool,
) -> UtteranceDecision {
    UtteranceDecision {
        devices,
        emergency_stop,
        requires_sterile_confirm,
    }
}

// --- Exactly one call ---------------------------------------------------------

#[tokio::test]
async fn one_inference_call_per_utterance_applies_all_devices() {
    let engine = MockDecisionEngine::from_decisions([decision(
        vec![
            device(
                TargetDevice::SurgicalLight,
                Some(ActionKind::DecreaseBrightness),
                Some(60),
            ),
            device(
                TargetDevice::EndoscopeCamera,
                Some(ActionKind::ZoomIn),
                Some(3),
            ),
            device(
                TargetDevice::Insufflator,
                Some(ActionKind::ToggleInsufflation),
                None,
            ),
            device(TargetDevice::OperatingTable, None, None),
        ],
        false,
        false,
    )]);
    let recording = engine.recording();
    let executor = SinglePassExecutor::new(engine);

    let result = executor
        .process_utterance(&RoomState::default(), "dim to 60, zoom 3, stop the CO2")
        .await
        .expect("run");

    // Hard invariant: one utterance -> exactly one inference call.
    assert_eq!(recording.borrow().len(), 1);
    let inputs = recording.borrow();
    assert_eq!(inputs[0].utterance, "dim to 60, zoom 3, stop the CO2");
    // The pass sees the current room state, nothing else.
    assert_eq!(inputs[0].room_state, RoomState::default());

    // All three device actions resolved within that single pass.
    assert_eq!(result.report.applied.len(), 3);
    let room = &result.new_room;
    assert_eq!(room.lighting.primary_intensity_pct, 60);
    assert_eq!(room.endoscope.zoom_level, 3);
    assert!(!room.insufflator.is_active);
}

// --- Null / no-change semantics ----------------------------------------------

#[tokio::test]
async fn null_device_decisions_leave_state_untouched() {
    let engine = MockDecisionEngine::from_decisions([decision(
        vec![
            device(TargetDevice::SurgicalLight, None, None),
            device(TargetDevice::EndoscopeCamera, None, None),
            device(TargetDevice::Insufflator, None, None),
            device(TargetDevice::OperatingTable, None, None),
        ],
        false,
        false,
    )]);
    let executor = SinglePassExecutor::new(engine);
    let before = RoomState::default();

    let result = executor
        .process_utterance(&before, "how is the patient doing")
        .await
        .expect("run");

    assert_eq!(result.new_room, before);
    assert!(result.report.applied.is_empty());
}

#[tokio::test]
async fn low_action_confidence_is_treated_as_no_change() {
    let mut light = device(
        TargetDevice::SurgicalLight,
        Some(ActionKind::DecreaseBrightness),
        Some(60),
    );
    light.confidence = 0.4;
    let engine = MockDecisionEngine::from_decisions([decision(vec![light], false, false)]);
    let executor = SinglePassExecutor::new(engine);
    let before = RoomState::default();

    let result = executor
        .process_utterance(&before, "dim?")
        .await
        .expect("run");

    assert_eq!(result.new_room, before);
    assert!(result.report.applied.is_empty());
}

#[tokio::test]
async fn low_value_confidence_skips_value_setting_action() {
    let mut light = device(
        TargetDevice::SurgicalLight,
        Some(ActionKind::DecreaseBrightness),
        Some(60),
    );
    light.absolute_confidence = 0.3;
    let engine = MockDecisionEngine::from_decisions([decision(vec![light], false, false)]);
    let executor = SinglePassExecutor::new(engine);
    let before = RoomState::default();

    let result = executor
        .process_utterance(&before, "dim to 60")
        .await
        .expect("run");

    assert_eq!(result.new_room, before);
    assert!(result.report.applied.is_empty());
}

#[tokio::test]
async fn missing_absolute_target_skips_value_setting_action() {
    let engine = MockDecisionEngine::from_decisions([decision(
        vec![device(
            TargetDevice::SurgicalLight,
            Some(ActionKind::DecreaseBrightness),
            None,
        )],
        false,
        false,
    )]);
    let executor = SinglePassExecutor::new(engine);
    let before = RoomState::default();

    let result = executor
        .process_utterance(&before, "dim")
        .await
        .expect("run");

    assert_eq!(result.new_room, before);
    assert!(result.report.applied.is_empty());
}

#[tokio::test]
async fn toggle_actions_need_no_absolute_target() {
    let engine = MockDecisionEngine::from_decisions([decision(
        vec![device(
            TargetDevice::EndoscopeCamera,
            Some(ActionKind::ToggleIrrigation),
            None,
        )],
        false,
        false,
    )]);
    let executor = SinglePassExecutor::new(engine);

    let result = executor
        .process_utterance(&RoomState::default(), "irrigate")
        .await
        .expect("run");

    assert!(result.new_room.endoscope.irrigation_active);
    assert_eq!(result.report.applied.len(), 1);
}

// --- Emergency stop precedence ------------------------------------------------

#[tokio::test]
async fn emergency_stop_trips_interlock_and_wins_over_device_actions() {
    let mut initial = RoomState::default();
    initial.endoscope.irrigation_active = true;

    let engine = MockDecisionEngine::from_decisions([decision(
        vec![device(
            TargetDevice::Insufflator,
            Some(ActionKind::AdjustPressure),
            Some(20),
        )],
        true,
        true,
    )]);
    let executor = SinglePassExecutor::new(engine);

    let result = executor
        .process_utterance(&initial, "stop everything now")
        .await
        .expect("run");

    assert!(result.new_room.safety_interlock_active);
    assert!(!result.new_room.insufflator.is_active);
    assert!(!result.new_room.endoscope.irrigation_active);
    assert!(result.report.requires_sterile_confirm);
    // Emergency stop report first, then the device group reports.
    assert!(result.report.applied.len() >= 2);
}

// --- Error handling -----------------------------------------------------------

#[tokio::test]
async fn invalid_device_action_pair_surfaces_error() {
    let engine = MockDecisionEngine::from_decisions([decision(
        vec![device(
            TargetDevice::SurgicalLight,
            Some(ActionKind::TiltTable),
            Some(5),
        )],
        false,
        false,
    )]);
    let executor = SinglePassExecutor::new(engine);

    let error = executor
        .process_utterance(&RoomState::default(), "bad pair")
        .await
        .expect_err("must fail");

    assert!(
        matches!(
            error,
            ExecutionError::InvalidTargetAction {
                target_device: TargetDevice::SurgicalLight,
                action_kind: ActionKind::TiltTable,
            }
        ),
        "unexpected error: {error:?}"
    );
}

#[tokio::test]
async fn provider_error_propagates_as_execution_error() {
    let engine =
        MockDecisionEngine::new([Err(ProviderError::Transport("connection refused".into()))]);
    let executor = SinglePassExecutor::new(engine);

    let error = executor
        .process_utterance(&RoomState::default(), "any")
        .await
        .expect_err("must fail");

    assert!(
        matches!(error, ExecutionError::Provider(ProviderError::Transport(_))),
        "unexpected error: {error:?}"
    );
}

// --- Report payload -----------------------------------------------------------

#[tokio::test]
async fn report_carries_latency_flags_and_no_applied_for_noise() {
    let engine = MockDecisionEngine::from_decisions([decision(vec![], false, true)]);
    let executor = SinglePassExecutor::new(engine);

    let result = executor
        .process_utterance(&RoomState::default(), "team comment")
        .await
        .expect("run");

    assert_eq!(result.report.latency, Duration::from_millis(1));
    assert!(result.report.requires_sterile_confirm);
    assert!(result.report.applied.is_empty());
}
