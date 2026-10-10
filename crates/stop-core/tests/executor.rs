//! Single-pass executor tests: scripted mock engine, exactly-one-call
//! invariant, per-object leave-as-is semantics, independent settings,
//! emergency stop precedence, error propagation.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Duration;

use stop_core::{
    CameraDecision, ExecutionError, InferenceInput, InferenceOutcome, InferencePort,
    InsufflatorDecision, LightDecision, LightMode, ProviderError, RoomState, SinglePassExecutor,
    StepValue, TableDecision, UtteranceDecision, ValueChange,
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
            .unwrap_or_else(|| Ok(UtteranceDecision::default()));
        let decision = item?;

        Ok(InferenceOutcome {
            decision,
            latency: Duration::from_millis(1),
        })
    }
}

fn light_decision(
    brightness: Option<ValueChange>,
    field_mode: Option<LightMode>,
) -> UtteranceDecision {
    UtteranceDecision {
        light: LightDecision {
            brightness,
            field_mode,
        },
        ..UtteranceDecision::default()
    }
}

// --- Exactly one call ---------------------------------------------------------

#[tokio::test]
async fn one_inference_call_per_utterance_applies_all_devices() {
    let decision = UtteranceDecision {
        light: LightDecision {
            brightness: Some(ValueChange::Absolute(60)),
            field_mode: None,
        },
        camera: CameraDecision {
            zoom: Some(ValueChange::Absolute(3)),
            toggle_irrigation: false,
        },
        insufflator: InsufflatorDecision {
            pressure: None,
            toggle_insufflation: true,
        },
        table: TableDecision::default(),
        emergency_stop: false,
        requires_sterile_confirm: false,
    };
    let engine = MockDecisionEngine::from_decisions([decision]);
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
    assert_eq!(inputs[0].room_state, RoomState::default());

    // All three device actions resolved within that single pass.
    assert_eq!(result.report.applied.len(), 3);
    let room = &result.new_room;
    assert_eq!(room.lighting.primary_intensity_pct, 60);
    assert_eq!(room.endoscope.zoom_level, 3);
    assert!(!room.insufflator.is_active);
}

#[tokio::test]
async fn brightness_and_field_mode_change_in_one_utterance() {
    let engine = MockDecisionEngine::from_decisions([light_decision(
        Some(ValueChange::Increase(10)),
        Some(LightMode::AmbientRed),
    )]);
    let executor = SinglePassExecutor::new(engine);

    let result = executor
        .process_utterance(&RoomState::default(), "brighten by 10 and go red")
        .await
        .expect("run");

    assert_eq!(result.new_room.lighting.primary_intensity_pct, 90);
    assert_eq!(result.new_room.lighting.field_mode, LightMode::AmbientRed);
    assert_eq!(result.report.applied.len(), 2);
}

#[tokio::test]
async fn tilt_and_height_change_in_one_utterance() {
    let engine = MockDecisionEngine::from_decisions([UtteranceDecision {
        table: TableDecision {
            tilt: Some(ValueChange::Absolute(10)),
            height: Some(ValueChange::Increase(15)),
        },
        ..UtteranceDecision::default()
    }]);
    let executor = SinglePassExecutor::new(engine);

    let result = executor
        .process_utterance(&RoomState::default(), "tilt 10 and raise 15")
        .await
        .expect("run");

    assert_eq!(result.new_room.table.tilt_degrees, 10);
    assert_eq!(result.new_room.table.height_cm, 115);
}

// --- Relative vs absolute operands -------------------------------------------

#[tokio::test]
async fn relative_operand_offsets_the_current_value() {
    let engine =
        MockDecisionEngine::from_decisions([light_decision(Some(ValueChange::Increase(8)), None)]);
    let executor = SinglePassExecutor::new(engine);
    let before = RoomState::default(); // 80 % by default

    let result = executor
        .process_utterance(&before, "brighten by 8")
        .await
        .expect("run");

    assert_eq!(result.new_room.lighting.primary_intensity_pct, 88);
}

#[tokio::test]
async fn default_relative_step_is_one() {
    let engine = MockDecisionEngine::from_decisions([UtteranceDecision {
        camera: CameraDecision {
            zoom: Some(ValueChange::Increase(1)),
            toggle_irrigation: false,
        },
        ..UtteranceDecision::default()
    }]);
    let executor = SinglePassExecutor::new(engine);
    let before = RoomState::default(); // zoom 2 by default

    let result = executor
        .process_utterance(&before, "zoom in")
        .await
        .expect("run");

    assert_eq!(result.new_room.endoscope.zoom_level, 3);
}

// --- Leave-as-is semantics ----------------------------------------------------

#[tokio::test]
async fn all_none_decisions_leave_state_untouched() {
    let engine = MockDecisionEngine::from_decisions([UtteranceDecision::default()]);
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
async fn toggle_actions_need_no_operand() {
    let engine = MockDecisionEngine::from_decisions([UtteranceDecision {
        camera: CameraDecision {
            zoom: None,
            toggle_irrigation: true,
        },
        ..UtteranceDecision::default()
    }]);
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

    let engine = MockDecisionEngine::from_decisions([UtteranceDecision {
        insufflator: InsufflatorDecision {
            pressure: Some(ValueChange::Absolute(20)),
            toggle_insufflation: false,
        },
        emergency_stop: true,
        requires_sterile_confirm: true,
        ..UtteranceDecision::default()
    }]);
    let executor = SinglePassExecutor::new(engine);

    let result = executor
        .process_utterance(&initial, "stop everything now")
        .await
        .expect("run");

    assert!(result.new_room.safety_interlock_active);
    assert!(!result.new_room.insufflator.is_active);
    assert!(!result.new_room.endoscope.irrigation_active);
    assert!(result.report.requires_sterile_confirm);
    // Emergency stop report first, then the object reports.
    assert!(result.report.applied.len() >= 2);
}

// --- Error handling -----------------------------------------------------------

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

#[test]
fn value_change_maps_to_action_and_step() {
    assert_eq!(
        StepValue::from(ValueChange::Absolute(60)),
        StepValue::AbsoluteValue(60)
    );
    assert_eq!(
        StepValue::from(ValueChange::Increase(8)),
        StepValue::Plus(8)
    );
    assert_eq!(
        StepValue::from(ValueChange::Decrease(3)),
        StepValue::Minus(3)
    );
}

// --- Report payload -----------------------------------------------------------

#[tokio::test]
async fn report_carries_latency_flags_and_no_applied_for_noise() {
    let engine = MockDecisionEngine::from_decisions([UtteranceDecision {
        requires_sterile_confirm: true,
        ..UtteranceDecision::default()
    }]);
    let executor = SinglePassExecutor::new(engine);

    let result = executor
        .process_utterance(&RoomState::default(), "team comment")
        .await
        .expect("run");

    assert_eq!(result.report.latency, Duration::from_millis(1));
    assert!(result.report.requires_sterile_confirm);
    assert!(result.report.applied.is_empty());
}
