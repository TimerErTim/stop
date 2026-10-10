//! Single-pass decision execution (spec `docs/INSTRUCTIONS.md` section 3.3).
//!
//! Exactly one inference call per utterance: the response carries an
//! explicit decision for every room object (`None` = leave as is), so no
//! re-evaluation loop exists and no safety guard is needed. All inference
//! runs through the async [`InferencePort`], so high System-One latency
//! never blocks the caller's executor.

use std::time::Duration;

use crate::decision::{
    ActionDecision, ActionKind, CameraDecision, InsufflatorDecision, LightDecision, StepValue,
    TableDecision, TargetDevice, UtteranceDecision, ValueChange,
};
use crate::delta::{AppliedActionReport, apply_action_to_state};
use crate::engine::{InferenceInput, InferencePort};
use crate::error::ExecutionError;
use crate::state::RoomState;

/// Minimum confidence for a decoded setting to take effect; below it the
/// setting is treated as "leave as is" (safe default). Applied during
/// System-One answer decoding.
pub const MIN_ACTION_CONFIDENCE: f32 = 0.5;

/// Report of one utterance execution (feeds the GUI HUD telemetry line and
/// the benchmark raw output).
#[derive(Debug, Clone)]
pub struct UtteranceReport {
    /// Applied state deltas in device order; empty when everything was
    /// `None` (noise semantics).
    pub applied: Vec<AppliedActionReport>,
    /// Model flagged the utterance as needing sterile confirmation.
    pub requires_sterile_confirm: bool,
    /// Latency of the single inference pass.
    pub latency: Duration,
    /// Raw prediction of the pass: the decoded decision of every slot, as
    /// returned by the inference response (persisted in benchmark raw
    /// output).
    pub decision: UtteranceDecision,
}

/// Result of [`SinglePassExecutor::process_utterance`]: the applied room
/// state plus the per-utterance report.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub new_room: RoomState,
    pub report: UtteranceReport,
}

/// Single-pass orchestrator: one utterance -> exactly one inference call.
pub struct SinglePassExecutor<I> {
    inference: I,
}

impl<I: InferencePort> SinglePassExecutor<I> {
    pub fn new(inference: I) -> Self {
        Self { inference }
    }

    /// Runs one utterance against `current_state` with exactly one
    /// inference call. Emergency stop is applied first (it overrides the
    /// device groups), then each object's independent settings; the object
    /// set is closed and small, so several actions per utterance resolve in
    /// this single pass without ordering hazards.
    pub async fn process_utterance(
        &self,
        current_state: &RoomState,
        utterance: &str,
    ) -> Result<ExecutionResult, ExecutionError> {
        let mut room = current_state.clone();
        let mut applied = Vec::new();

        let outcome = self
            .inference
            .single_pass(&InferenceInput {
                room_state: &room,
                utterance,
            })
            .await?;
        let decision = outcome.decision;

        // Emergency stop first: it shuts down insufflation/irrigation and
        // must not be overridden by object actions.
        if decision.emergency_stop {
            let report = apply_action_to_state(
                &mut room,
                &ActionDecision {
                    target_device: TargetDevice::None,
                    action_kind: ActionKind::EmergencyStop,
                    step_value: StepValue::Zero,
                },
            )?;
            applied.push(report);
        }

        apply_light(&mut room, &decision.light, &mut applied)?;
        apply_camera(&mut room, &decision.camera, &mut applied)?;
        apply_insufflator(&mut room, &decision.insufflator, &mut applied)?;
        apply_table(&mut room, &decision.table, &mut applied)?;

        Ok(ExecutionResult {
            new_room: room,
            report: UtteranceReport {
                applied,
                requires_sterile_confirm: decision.requires_sterile_confirm,
                latency: outcome.latency,
                decision,
            },
        })
    }
}

/// Applies a light decision: brightness and field mode are independent.
fn apply_light(
    room: &mut RoomState,
    decision: &LightDecision,
    applied: &mut Vec<AppliedActionReport>,
) -> Result<(), ExecutionError> {
    if let Some(change) = decision.brightness {
        let action = brightness_action(change);
        apply_change(room, TargetDevice::SurgicalLight, action, change, applied)?;
    }
    if let Some(mode) = &decision.field_mode {
        apply_action_to_state(
            room,
            &ActionDecision {
                target_device: TargetDevice::SurgicalLight,
                action_kind: ActionKind::SetLightMode,
                step_value: StepValue::AbsoluteValue(light_mode_code(mode)),
            },
        )
        .map(|report| applied.push(report))?;
    }
    Ok(())
}

/// Applies a camera decision: zoom plus an optional irrigation toggle.
fn apply_camera(
    room: &mut RoomState,
    decision: &CameraDecision,
    applied: &mut Vec<AppliedActionReport>,
) -> Result<(), ExecutionError> {
    if let Some(change) = decision.zoom {
        let action = zoom_action(change);
        apply_change(room, TargetDevice::EndoscopeCamera, action, change, applied)?;
    }
    if decision.toggle_irrigation {
        apply_toggle(
            room,
            TargetDevice::EndoscopeCamera,
            ActionKind::ToggleIrrigation,
            applied,
        )?;
    }
    Ok(())
}

/// Applies an insufflator decision: pressure plus an optional insufflation
/// toggle.
fn apply_insufflator(
    room: &mut RoomState,
    decision: &InsufflatorDecision,
    applied: &mut Vec<AppliedActionReport>,
) -> Result<(), ExecutionError> {
    if let Some(change) = decision.pressure {
        let action = pressure_action(change);
        apply_change(room, TargetDevice::Insufflator, action, change, applied)?;
    }
    if decision.toggle_insufflation {
        apply_toggle(
            room,
            TargetDevice::Insufflator,
            ActionKind::ToggleInsufflation,
            applied,
        )?;
    }
    Ok(())
}

/// Applies a table decision: tilt and height are independent.
fn apply_table(
    room: &mut RoomState,
    decision: &TableDecision,
    applied: &mut Vec<AppliedActionReport>,
) -> Result<(), ExecutionError> {
    if let Some(change) = decision.tilt {
        let action = tilt_action(change);
        apply_change(room, TargetDevice::OperatingTable, action, change, applied)?;
    }
    if let Some(change) = decision.height {
        let action = height_action(change);
        apply_change(room, TargetDevice::OperatingTable, action, change, applied)?;
    }
    Ok(())
}

fn apply_change(
    room: &mut RoomState,
    device: TargetDevice,
    action: ActionKind,
    change: ValueChange,
    applied: &mut Vec<AppliedActionReport>,
) -> Result<(), ExecutionError> {
    apply_action_to_state(
        room,
        &ActionDecision {
            target_device: device,
            action_kind: action,
            step_value: change.into(),
        },
    )
    .map(|report| applied.push(report))
}

fn apply_toggle(
    room: &mut RoomState,
    device: TargetDevice,
    action: ActionKind,
    applied: &mut Vec<AppliedActionReport>,
) -> Result<(), ExecutionError> {
    apply_action_to_state(
        room,
        &ActionDecision {
            target_device: device,
            action_kind: action,
            step_value: StepValue::Zero,
        },
    )
    .map(|report| applied.push(report))
}

fn brightness_action(change: ValueChange) -> ActionKind {
    match change {
        ValueChange::Absolute(_) => ActionKind::SetBrightness,
        ValueChange::Increase(_) => ActionKind::IncreaseBrightness,
        ValueChange::Decrease(_) => ActionKind::DecreaseBrightness,
    }
}

fn zoom_action(change: ValueChange) -> ActionKind {
    match change {
        ValueChange::Absolute(_) => ActionKind::SetZoom,
        ValueChange::Increase(_) => ActionKind::ZoomIn,
        ValueChange::Decrease(_) => ActionKind::ZoomOut,
    }
}

fn pressure_action(change: ValueChange) -> ActionKind {
    match change {
        ValueChange::Absolute(_) => ActionKind::SetPressure,
        ValueChange::Increase(_) => ActionKind::IncreasePressure,
        ValueChange::Decrease(_) => ActionKind::DecreasePressure,
    }
}

fn tilt_action(change: ValueChange) -> ActionKind {
    match change {
        ValueChange::Absolute(_) => ActionKind::SetTilt,
        ValueChange::Increase(_) => ActionKind::IncreaseTilt,
        ValueChange::Decrease(_) => ActionKind::DecreaseTilt,
    }
}

fn height_action(change: ValueChange) -> ActionKind {
    match change {
        ValueChange::Absolute(_) => ActionKind::SetHeight,
        ValueChange::Increase(_) => ActionKind::IncreaseHeight,
        ValueChange::Decrease(_) => ActionKind::DecreaseHeight,
    }
}

fn light_mode_code(mode: &crate::state::LightMode) -> i16 {
    match mode {
        crate::state::LightMode::Normal => 0,
        crate::state::LightMode::CavityFocus => 1,
        crate::state::LightMode::AmbientRed => 2,
    }
}
