//! `stop-core`: domain model, System-One API client, single-pass engine, events.
//!
//! The core delivers the room state, decision types, the deterministic
//! state-delta application with safety caps, the async inference port, the
//! single-pass executor, and the System-One HTTP client.
//! See `docs/INSTRUCTIONS.md` section 3.

pub mod decision;
pub mod delta;
pub mod engine;
pub mod error;
pub mod executor;
pub mod state;
pub mod systemone;

pub use decision::{
    ActionDecision, ActionKind, CameraDecision, InsufflatorDecision, LightDecision, StepValue,
    TableDecision, TargetDevice, UtteranceDecision, ValueChange,
};
pub use delta::{AppliedActionReport, apply_action_to_state};
pub use engine::{InferenceInput, InferenceOutcome, InferencePort, ProviderError};
pub use error::ExecutionError;
pub use executor::{ExecutionResult, SinglePassExecutor, UtteranceReport};
pub use state::{
    BRIGHTNESS_MAX, BRIGHTNESS_MIN, EndoscopeState, HEIGHT_MAX, HEIGHT_MIN, InsufflatorState,
    LightMode, LightingState, MAX_PRESSURE_MMHG, RoomState, TILT_MAX, TILT_MIN, TableState,
    ZOOM_MAX, ZOOM_MIN,
};
