//! Provider abstraction for a single decision pass (Phase 2).
//!
//! `stop` talks to a JevK5 / System-One typed-decision endpoint through
//! [`InferencePort`]; tests substitute a scripted mock. System-One inference
//! carries high latency, so the trait is `async` end to end and nothing in
//! this layer blocks.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::decision::ActionDecision;
use crate::delta::AppliedActionReport;
use crate::state::RoomState;

/// Transport / protocol failures of a decision provider.
///
/// Slot-level inconsistencies from the model surface as
/// [`ProviderError::InconsistentSlots`] and are re-mapped to
/// [`crate::ExecutionError::InconsistentSlots`] by the executor.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ProviderError {
    #[error("transport error: {0}")]
    Transport(String),

    #[error("request timed out: {0}")]
    Timeout(String),

    #[error("HTTP status {status}: {body}")]
    Status { status: u16, body: String },

    #[error("malformed system-one response: {0}")]
    Malformed(String),

    #[error("model returned inconsistent decision slots: {0}")]
    InconsistentSlots(String),
}

/// Everything one pass needs: current room state, the utterance, and the
/// action reports already applied by earlier passes of the same utterance.
#[derive(Debug, Clone)]
pub struct InferenceInput<'a> {
    pub room_state: &'a RoomState,
    pub utterance: &'a str,
    pub history: &'a [AppliedActionReport],
}

/// Per-slot confidences from one pass; benchmark ROC/AUC (Phase 4) reads
/// these from the persisted raw output.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct SlotConfidences {
    pub further_action_needed: f32,
    pub requires_sterile_confirm: f32,
    pub target_device: f32,
    pub action_kind: f32,
    pub step_value: f32,
    /// Confidence of the absolute-target answer actually used this pass
    /// (relative passes report the `step_value` confidence here).
    pub absolute_target: f32,
}

/// Decoded result of one inference pass.
#[derive(Debug, Clone)]
pub struct InferenceOutcome {
    pub decision: ActionDecision,
    pub slot_confidences: SlotConfidences,
    /// Inference latency (server-reported when the response carries it,
    /// wall-clock otherwise).
    pub latency: Duration,
}

/// One parallel evaluation of all decision slots against the current state.
///
/// Native `async fn` in trait: the executor is generic over the port, never
/// `dyn`, so no `async-trait` boxing is needed. `Send`ness of the executor's
/// future leaks from the concrete port (SystemOneClient is `Send`, the test
/// mock intentionally is not), which is why the lint is allowed here.
#[allow(async_fn_in_trait)]
pub trait InferencePort {
    async fn single_pass(
        &self,
        input: &InferenceInput<'_>,
    ) -> Result<InferenceOutcome, ProviderError>;
}
