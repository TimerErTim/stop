//! Error type for state-delta application and the multi-pass executor.

use crate::decision::{ActionKind, StepValue, TargetDevice};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ExecutionError {
    #[error("action {action_kind:?} is not valid for target device {target_device:?}")]
    InvalidTargetAction {
        target_device: TargetDevice,
        action_kind: ActionKind,
    },

    #[error("step {step_value:?} is not valid for action {action_kind:?}")]
    InvalidStep {
        action_kind: ActionKind,
        step_value: StepValue,
    },

    #[error("unknown light mode code {0}")]
    UnknownLightMode(i16),

    #[error("model returned inconsistent decision slots: {0}")]
    InconsistentSlots(String),

    #[error("decision provider failed: {0}")]
    Provider(#[from] crate::engine::ProviderError),
}
