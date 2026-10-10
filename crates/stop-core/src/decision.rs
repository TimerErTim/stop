//! Decision schema emitted by the JevK5-2B typed parallel heads.
//!
//! Single-pass schema: one inference pass returns one [`DeviceDecision`] per
//! room object plus the global flags ([`UtteranceDecision`]). Every object
//! carries its own choice set with a `null` option (`action == None` = no
//! change to that object); value-setting actions carry absolute targets
//! (brightness %, zoom level, mmHg, degrees, cm, light-mode code).

use serde::{Deserialize, Serialize};

/// Deterministic state-delta unit applied by `crate::delta`: one device,
/// one action, one step (relative delta or absolute target).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ActionDecision {
    /// Target device of the action.
    pub target_device: TargetDevice,

    /// Concrete operation on the target device.
    pub action_kind: ActionKind,

    /// Relative step / absolute target for the operation.
    pub step_value: StepValue,
}

/// One object's decision from a single inference pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeviceDecision {
    pub target_device: TargetDevice,
    /// `None` when the model chose `null`: no change to this object.
    pub action: Option<ActionKind>,
    /// Absolute target of a value-setting action (brightness %, zoom level,
    /// mmHg, degrees, cm, light-mode code). `None` for toggles or no-change.
    pub absolute: Option<i16>,
    /// Confidence of the action choice (0.0 when unavailable).
    pub confidence: f32,
    /// Confidence of the absolute target (0.0 when unavailable).
    pub absolute_confidence: f32,
}

/// Complete decision set of one inference pass: all room objects at once.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UtteranceDecision {
    /// The closed object set of the room, one entry per device (light,
    /// endoscope camera, insufflator, table). Order is not semantic.
    pub devices: Vec<DeviceDecision>,
    /// Trigger the safety interlock, shut down insufflation and irrigation.
    pub emergency_stop: bool,
    /// Safety-relevant verification needed (overpressure, table tilt).
    pub requires_sterile_confirm: bool,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TargetDevice {
    None,
    SurgicalLight,
    EndoscopeCamera,
    Insufflator,
    OperatingTable,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ActionKind {
    Idle,
    IncreaseBrightness,
    DecreaseBrightness,
    SetLightMode,
    ZoomIn,
    ZoomOut,
    ToggleIrrigation,
    AdjustPressure,
    ToggleInsufflation,
    TiltTable,
    SetTableHeight,
    EmergencyStop,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum StepValue {
    Zero,
    PlusOne,
    PlusTwo,
    MinusOne,
    MinusTwo,
    AbsoluteValue(i16),
}

impl StepValue {
    /// Numeric interpretation of the step: relative delta for `Zero..MinusTwo`,
    /// absolute target for `AbsoluteValue`.
    pub fn as_i16(&self) -> i16 {
        match self {
            StepValue::Zero => 0,
            StepValue::PlusOne => 1,
            StepValue::PlusTwo => 2,
            StepValue::MinusOne => -1,
            StepValue::MinusTwo => -2,
            StepValue::AbsoluteValue(v) => *v,
        }
    }

    pub fn is_absolute(&self) -> bool {
        matches!(self, StepValue::AbsoluteValue(_))
    }
}
