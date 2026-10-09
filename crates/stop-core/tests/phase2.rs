//! Phase 2 acceptance tests: scripted mock engine, multi-pass termination,
//! safety-limit guard, history growth, emergency stop, error propagation.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Duration;

use stop_core::{
    ActionDecision, ActionKind, ExecutionError, InferenceInput, InferenceOutcome, InferencePort,
    MultiPassExecutor, ProviderError, RoomState, SlotConfidences, StepValue, TargetDevice,
};

/// Recorded observation of one `single_pass` call.
#[derive(Debug, Clone)]
struct PassInput {
    history_len: usize,
    utterance: String,
    room_state: RoomState,
}

type Recording = Rc<RefCell<Vec<PassInput>>>;

/// Tests-only mock: replays a scripted sequence of results and records
/// every input it receives. When the script runs dry, the last decision
/// repeats (models loop on `further_action_needed == true`).
struct MockDecisionEngine {
    script: RefCell<VecDeque<Result<ActionDecision, ProviderError>>>,
    recording: Recording,
}

impl MockDecisionEngine {
    fn new<I>(script: I) -> Self
    where
        I: IntoIterator<Item = Result<ActionDecision, ProviderError>>,
    {
        Self {
            script: RefCell::new(script.into_iter().collect()),
            recording: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn from_decisions<I>(decisions: I) -> Self
    where
        I: IntoIterator<Item = ActionDecision>,
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
            history_len: input.history.len(),
            utterance: input.utterance.to_string(),
            room_state: input.room_state.clone(),
        });

        let mut script = self.script.borrow_mut();
        // Play the script in order; the final entry repeats forever when the
        // script is exhausted (models loop on `further_action_needed`).
        let item = if script.len() > 1 {
            script.pop_front().expect("len > 1 checked above")
        } else {
            script.back().cloned().unwrap_or_else(|| Ok(idle(false)))
        };
        let decision = item?;

        Ok(InferenceOutcome {
            decision,
            slot_confidences: SlotConfidences {
                further_action_needed: 0.9,
                requires_sterile_confirm: 0.8,
                target_device: 0.95,
                action_kind: 0.9,
                step_value: 0.7,
                absolute_target: 0.7,
            },
            latency: Duration::from_millis(1),
        })
    }
}

fn decision(
    device: TargetDevice,
    action: ActionKind,
    step: StepValue,
    further: bool,
) -> ActionDecision {
    ActionDecision {
        further_action_needed: further,
        target_device: device,
        action_kind: action,
        step_value: step,
        requires_sterile_confirm: false,
    }
}

fn idle(further: bool) -> ActionDecision {
    decision(
        TargetDevice::None,
        ActionKind::Idle,
        StepValue::Zero,
        further,
    )
}

// --- Termination ------------------------------------------------------------

#[tokio::test]
async fn terminates_after_single_pass_when_further_is_false() {
    let engine = MockDecisionEngine::from_decisions([decision(
        TargetDevice::SurgicalLight,
        ActionKind::DecreaseBrightness,
        StepValue::MinusTwo,
        false,
    )]);
    let recording = engine.recording();
    let executor = MultiPassExecutor::new(engine);

    let result = executor
        .process_utterance(&RoomState::default(), "dim the light")
        .await
        .expect("run");

    assert_eq!(result.passes.len(), 1);
    assert_eq!(result.passes[0].pass_index.get(), 1);
    assert_eq!(result.new_room.lighting.primary_intensity_pct, 78);
    assert!(result.passes[0].applied.is_some());
    assert_eq!(recording.borrow().len(), 1);
}

#[tokio::test]
async fn runs_full_sequence_and_accumulates_history() {
    let engine = MockDecisionEngine::from_decisions([
        decision(
            TargetDevice::SurgicalLight,
            ActionKind::DecreaseBrightness,
            StepValue::MinusTwo,
            true,
        ),
        decision(
            TargetDevice::EndoscopeCamera,
            ActionKind::ZoomIn,
            StepValue::PlusOne,
            true,
        ),
        decision(
            TargetDevice::Insufflator,
            ActionKind::AdjustPressure,
            StepValue::AbsoluteValue(14),
            false,
        ),
    ]);
    let recording = engine.recording();
    let executor = MultiPassExecutor::new(engine);

    let result = executor
        .process_utterance(
            &RoomState::default(),
            "dim two steps, get closer, set pressure to 14",
        )
        .await
        .expect("run");

    assert_eq!(result.passes.len(), 3);
    let room = &result.new_room;
    assert_eq!(room.lighting.primary_intensity_pct, 78);
    assert_eq!(room.endoscope.zoom_level, 3);
    assert_eq!(room.insufflator.target_pressure_mmhg, 14);

    // History grows one report per applied pass: 0, 1, 2.
    let inputs = recording.borrow();
    let lens: Vec<usize> = inputs.iter().map(|i| i.history_len).collect();
    assert_eq!(lens, vec![0, 1, 2]);
    // Pass 2 sees the state updated by pass 1.
    assert_eq!(inputs[1].room_state.lighting.primary_intensity_pct, 78);
    assert!(
        inputs
            .iter()
            .all(|i| i.utterance == "dim two steps, get closer, set pressure to 14")
    );
}

// --- Safety guard -----------------------------------------------------------

#[tokio::test]
async fn max_passes_guard_stops_endless_loop() {
    // Always `further_action_needed == true`; must stop at max_passes.
    let engine = MockDecisionEngine::from_decisions([decision(
        TargetDevice::SurgicalLight,
        ActionKind::DecreaseBrightness,
        StepValue::MinusTwo,
        true,
    )]);
    let executor = MultiPassExecutor::new(engine);

    let result = executor
        .process_utterance(&RoomState::default(), "keep going forever")
        .await
        .expect("run");

    assert_eq!(executor.max_passes(), 4);
    assert_eq!(result.passes.len(), 4);
    assert_eq!(result.new_room.lighting.primary_intensity_pct, 72);
}

#[tokio::test]
async fn target_device_none_terminates_despite_further_true() {
    let engine = MockDecisionEngine::from_decisions([decision(
        TargetDevice::None,
        ActionKind::IncreaseBrightness,
        StepValue::PlusTwo,
        true,
    )]);
    let executor = MultiPassExecutor::new(engine);
    let before = RoomState::default();

    let result = executor
        .process_utterance(&before, "nothing to do")
        .await
        .expect("run");

    assert_eq!(result.passes.len(), 1);
    assert!(result.passes[0].applied.is_none());
    assert_eq!(result.new_room, before);
}

// --- Safety semantics through the loop -------------------------------------

#[tokio::test]
async fn emergency_stop_via_executor_trips_interlock() {
    let mut initial = RoomState::default();
    initial.endoscope.irrigation_active = true;

    let engine = MockDecisionEngine::from_decisions([
        decision(
            TargetDevice::Insufflator,
            ActionKind::EmergencyStop,
            StepValue::Zero,
            true,
        ),
        idle(false),
    ]);
    let executor = MultiPassExecutor::new(engine);

    let result = executor
        .process_utterance(&initial, "stop everything now")
        .await
        .expect("run");

    assert!(result.new_room.safety_interlock_active);
    assert!(!result.new_room.insufflator.is_active);
    assert!(!result.new_room.endoscope.irrigation_active);
    assert_eq!(result.passes.len(), 2);
}

#[tokio::test]
async fn invalid_device_action_pair_surfaces_error() {
    let engine = MockDecisionEngine::from_decisions([decision(
        TargetDevice::SurgicalLight,
        ActionKind::TiltTable,
        StepValue::PlusTwo,
        false,
    )]);
    let executor = MultiPassExecutor::new(engine);

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
    let executor = MultiPassExecutor::new(engine);

    let error = executor
        .process_utterance(&RoomState::default(), "any")
        .await
        .expect_err("must fail");

    assert!(
        matches!(error, ExecutionError::Provider(ProviderError::Transport(_))),
        "unexpected error: {error:?}"
    );
}

// --- Report payload ---------------------------------------------------------

#[tokio::test]
async fn pass_reports_carry_latency_and_confidences() {
    let engine = MockDecisionEngine::from_decisions([idle(false)]);
    let executor = MultiPassExecutor::new(engine);

    let result = executor
        .process_utterance(&RoomState::default(), "nothing")
        .await
        .expect("run");

    let pass = &result.passes[0];
    assert_eq!(pass.latency, Duration::from_millis(1));
    assert_eq!(pass.slot_confidences.target_device, 0.95);
    assert_eq!(pass.slot_confidences.further_action_needed, 0.9);
}
