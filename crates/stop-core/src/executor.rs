//! Single-pass decision execution (spec `docs/INSTRUCTIONS.md` section 3.3).
//!
//! Exactly one inference call per utterance: the response already carries
//! every object's decision (`null` = no change), so no re-evaluation loop
//! exists and no safety guard is needed. All inference runs through the
//! async [`InferencePort`], so high System-One latency never blocks the
//! caller's executor.

use std::time::Duration;

use crate::decision::{ActionDecision, ActionKind, DeviceDecision, StepValue, TargetDevice};
use crate::delta::{AppliedActionReport, apply_action_to_state};
use crate::engine::{InferenceInput, InferencePort};
use crate::error::ExecutionError;
use crate::state::RoomState;

/// Minimum confidence to apply a device action at all.
pub const MIN_ACTION_CONFIDENCE: f32 = 0.5;
/// Minimum confidence to apply the absolute target of an action; below it
/// the device sees no change (safe default).
pub const MIN_VALUE_CONFIDENCE: f32 = 0.5;

/// Report of one utterance execution (feeds the GUI HUD telemetry line and
/// the benchmark raw output).
#[derive(Debug, Clone)]
pub struct UtteranceReport {
    /// Applied state deltas in device order; empty when everything was
    /// `null` / below confidence (noise semantics).
    pub applied: Vec<AppliedActionReport>,
    /// Model flagged the utterance as needing sterile confirmation.
    pub requires_sterile_confirm: bool,
    /// Latency of the single inference pass.
    pub latency: Duration,
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
    /// device groups), then each object's decision independently; the
    /// object set is closed and small, so several actions per utterance
    /// resolve in this single pass without ordering hazards.
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
        // must not be overridden by group actions.
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

        for device in &decision.devices {
            if let Some(report) = apply_device_decision(&mut room, device)? {
                applied.push(report);
            }
        }

        Ok(ExecutionResult {
            new_room: room,
            report: UtteranceReport {
                applied,
                requires_sterile_confirm: decision.requires_sterile_confirm,
                latency: outcome.latency,
            },
        })
    }
}

/// Applies one object's decision with confidence gating; `None` means no
/// state change (model `null`, low confidence, or missing target value).
fn apply_device_decision(
    room: &mut RoomState,
    device: &DeviceDecision,
) -> Result<Option<AppliedActionReport>, ExecutionError> {
    let Some(action) = device.action else {
        return Ok(None); // model chose null: no change to this object
    };
    if device.confidence < MIN_ACTION_CONFIDENCE {
        tracing::warn!(
            device = ?device.target_device,
            confidence = device.confidence,
            "low action confidence, treating as no change"
        );
        return Ok(None);
    }

    let step_value = match absolute_action(action) {
        true => match device.absolute {
            Some(value) if device.absolute_confidence >= MIN_VALUE_CONFIDENCE => {
                StepValue::AbsoluteValue(value)
            }
            _ => {
                tracing::warn!(
                    device = ?device.target_device,
                    action = ?action,
                    "value-setting action without a confident target, treating as no change"
                );
                return Ok(None);
            }
        },
        false => StepValue::Zero,
    };

    let report = apply_action_to_state(
        room,
        &ActionDecision {
            target_device: device.target_device,
            action_kind: action,
            step_value,
        },
    )?;
    Ok(Some(report))
}

/// Value-setting actions take their value from the absolute target;
/// toggles flip state and carry no value.
fn absolute_action(action: ActionKind) -> bool {
    matches!(
        action,
        ActionKind::IncreaseBrightness
            | ActionKind::DecreaseBrightness
            | ActionKind::SetLightMode
            | ActionKind::ZoomIn
            | ActionKind::ZoomOut
            | ActionKind::AdjustPressure
            | ActionKind::TiltTable
            | ActionKind::SetTableHeight
    )
}
