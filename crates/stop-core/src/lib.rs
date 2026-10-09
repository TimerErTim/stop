//! `stop-core`: domain model, Jev API client, multi-pass engine, events.
//!
//! Phase 1 delivers the room state, decision slot types, and the deterministic
//! state-delta application with safety caps. See `docs/INSTRUCTIONS.md` section 3.

pub mod decision;
pub mod delta;
pub mod error;
pub mod state;

pub use decision::{ActionDecision, ActionKind, StepValue, TargetDevice};
pub use delta::{AppliedActionReport, apply_action_to_state};
pub use error::ExecutionError;
pub use state::{
    BRIGHTNESS_MAX, BRIGHTNESS_MIN, EndoscopeState, InsufflatorState, LightMode, LightingState,
    MAX_PRESSURE_MMHG, RoomState, TILT_MAX, TILT_MIN, TableState, ZOOM_MAX, ZOOM_MIN,
};
