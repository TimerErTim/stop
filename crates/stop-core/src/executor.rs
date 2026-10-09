//! Multi-pass decision loop (spec `docs/INSTRUCTIONS.md` section 3.3).
//!
//! One utterance can imply several simultaneous actions; the executor
//! re-evaluates the updated state while the model reports
//! `further_action_needed`, capped by a safety `max_passes` guard. All
//! inference runs through the async [`InferencePort`], so high System-One
//! latency never blocks the caller's executor.

use std::num::NonZeroUsize;
use std::time::Duration;

use crate::decision::{ActionDecision, TargetDevice};
use crate::delta::{AppliedActionReport, apply_action_to_state};
use crate::engine::{InferenceInput, InferencePort, SlotConfidences};
use crate::error::ExecutionError;
use crate::state::RoomState;

/// Default safety guard: hard cap on passes per utterance.
pub const DEFAULT_MAX_PASSES: usize = 4;

/// One executed pass of the loop (feeds the GUI HUD telemetry line
/// `Pass 1: Light -> Dim (-2) [19ms]` and the benchmark raw output).
#[derive(Debug, Clone)]
pub struct PassReport {
    /// 1-based pass index.
    pub pass_index: NonZeroUsize,
    pub decision: ActionDecision,
    /// `None` when the pass was a no-op (`TargetDevice::None` / `Idle`).
    pub applied: Option<AppliedActionReport>,
    pub slot_confidences: SlotConfidences,
    pub latency: Duration,
}

/// Result of [`MultiPassExecutor::process_utterance`]: the applied room
/// state plus the per-pass reports in execution order.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub new_room: RoomState,
    pub passes: Vec<PassReport>,
}

/// Cyclic multi-pass orchestrator.
pub struct MultiPassExecutor<I> {
    inference: I,
    max_passes: usize,
}

impl<I: InferencePort> MultiPassExecutor<I> {
    pub fn new(inference: I) -> Self {
        Self {
            inference,
            max_passes: DEFAULT_MAX_PASSES,
        }
    }

    /// Overrides the safety guard (tests, benchmarks).
    pub fn with_max_passes(mut self, max_passes: usize) -> Self {
        self.max_passes = max_passes.max(1);
        self
    }

    pub fn max_passes(&self) -> usize {
        self.max_passes
    }

    /// Runs the loop for one utterance against `current_state`.
    ///
    /// Terminates when the model reports `further_action_needed == false`,
    /// when `target_device == None`, or when `max_passes` is reached (warn,
    /// not an error — spec safety guard against infinite loops).
    pub async fn process_utterance(
        &self,
        current_state: &RoomState,
        utterance: &str,
    ) -> Result<ExecutionResult, ExecutionError> {
        let mut room = current_state.clone();
        let mut history: Vec<AppliedActionReport> = Vec::new();
        let mut passes = Vec::new();

        loop {
            if passes.len() >= self.max_passes {
                tracing::warn!(
                    max_passes = self.max_passes,
                    "Max passes reached, breaking cycle"
                );
                break;
            }

            // 1. Inference payload: state + prompt + history (spec 3.3).
            let outcome = self
                .inference
                .single_pass(&InferenceInput {
                    room_state: &room,
                    utterance,
                    history: &history,
                })
                .await?;

            // 2. Apply the state delta deterministically (safety caps live
            //    in the state setters: see section 3.1).
            let decision = outcome.decision;
            let applied = if decision.target_device != TargetDevice::None {
                let report = apply_action_to_state(&mut room, &decision)?;
                history.push(report.clone());
                Some(report)
            } else {
                None
            };

            passes.push(PassReport {
                pass_index: NonZeroUsize::new(passes.len() + 1)
                    .expect("passes.len() + 1 is never zero"),
                decision,
                applied,
                slot_confidences: outcome.slot_confidences,
                latency: outcome.latency,
            });

            // 3. Stop condition.
            let last = passes.last().expect("pushed above");
            if !last.decision.further_action_needed
                || last.decision.target_device == TargetDevice::None
            {
                break;
            }
        }

        Ok(ExecutionResult {
            new_room: room,
            passes,
        })
    }
}
