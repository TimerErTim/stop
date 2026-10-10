//! Decision schema emitted by the JevK5-2B typed parallel heads.
//!
//! Single-pass schema: one inference pass returns one [`UtteranceDecision`]
//! carrying a fully explicit decision per room object. Every object lists
//! its settings separately and every field is optional — `None` means "leave
//! as is" (the model chose `null` / the toggle is off). Value-setting
//! settings come in a `Set` (absolute) and `Increase` / `Decrease`
//! (relative) flavour and carry their numeric operand; toggles are booleans.

use serde::{Deserialize, Serialize};

use crate::state::LightMode;

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

/// Complete decision set of one inference pass: every room object at once.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct UtteranceDecision {
    /// Surgical light decision (brightness and field mode are independent).
    pub light: LightDecision,
    /// Endoscope camera decision.
    pub camera: CameraDecision,
    /// Insufflator decision.
    pub insufflator: InsufflatorDecision,
    /// Operating table decision (tilt and height are independent).
    pub table: TableDecision,
    /// Trigger the safety interlock, shut down insufflation and irrigation.
    pub emergency_stop: bool,
    /// Safety-relevant verification needed (overpressure, table tilt).
    pub requires_sterile_confirm: bool,
}

/// Surgical light: brightness and field mode can change in the same utterance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct LightDecision {
    /// `None` = leave brightness as is.
    pub brightness: Option<ValueChange>,
    /// `None` = leave the field mode as is.
    pub field_mode: Option<LightMode>,
}

/// Endoscope camera: zoom and irrigation can change in the same utterance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct CameraDecision {
    /// `None` = leave zoom as is.
    pub zoom: Option<ValueChange>,
    /// `true` flips irrigation, `false` leaves it as is.
    pub toggle_irrigation: bool,
}

/// CO2 insufflator: target pressure and insufflation can change together.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct InsufflatorDecision {
    /// `None` = leave target pressure as is.
    pub pressure: Option<ValueChange>,
    /// `true` flips insufflation, `false` leaves it as is.
    pub toggle_insufflation: bool,
}

/// Operating table: tilt and height can change in the same utterance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TableDecision {
    /// `None` = leave tilt as is.
    pub tilt: Option<ValueChange>,
    /// `None` = leave height as is.
    pub height: Option<ValueChange>,
}

/// One numeric setting change: absolute target or relative step.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValueChange {
    /// Set the setting to an absolute value.
    Absolute(i16),
    /// Raise the setting by the given amount.
    Increase(i16),
    /// Lower the setting by the given amount.
    Decrease(i16),
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
    SetBrightness,
    IncreaseBrightness,
    DecreaseBrightness,
    SetLightMode,
    SetZoom,
    ZoomIn,
    ZoomOut,
    ToggleIrrigation,
    SetPressure,
    IncreasePressure,
    DecreasePressure,
    ToggleInsufflation,
    SetTilt,
    IncreaseTilt,
    DecreaseTilt,
    SetHeight,
    IncreaseHeight,
    DecreaseHeight,
    EmergencyStop,
}

impl ActionKind {
    /// `Set*` actions take their operand as an absolute target.
    pub fn is_absolute(self) -> bool {
        matches!(
            self,
            Self::SetBrightness
                | Self::SetLightMode
                | Self::SetZoom
                | Self::SetPressure
                | Self::SetTilt
                | Self::SetHeight
        )
    }

    /// Relative actions that raise the current value by their operand.
    pub fn is_relative_increase(self) -> bool {
        matches!(
            self,
            Self::IncreaseBrightness
                | Self::ZoomIn
                | Self::IncreasePressure
                | Self::IncreaseTilt
                | Self::IncreaseHeight
        )
    }

    /// Relative actions that lower the current value by their operand.
    pub fn is_relative_decrease(self) -> bool {
        matches!(
            self,
            Self::DecreaseBrightness
                | Self::ZoomOut
                | Self::DecreasePressure
                | Self::DecreaseTilt
                | Self::DecreaseHeight
        )
    }
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum StepValue {
    Zero,
    Plus(i16),
    Minus(i16),
    AbsoluteValue(i16),
}

impl StepValue {
    /// Numeric interpretation of the step: relative delta for `Plus` /
    /// `Minus` / `Zero`, absolute target for `AbsoluteValue`.
    pub fn as_i16(&self) -> i16 {
        match self {
            StepValue::Zero => 0,
            StepValue::Plus(v) => *v,
            StepValue::Minus(v) => -*v,
            StepValue::AbsoluteValue(v) => *v,
        }
    }

    pub fn is_absolute(&self) -> bool {
        matches!(self, StepValue::AbsoluteValue(_))
    }
}

impl From<ValueChange> for StepValue {
    fn from(change: ValueChange) -> Self {
        match change {
            ValueChange::Absolute(v) => StepValue::AbsoluteValue(v),
            ValueChange::Increase(v) => StepValue::Plus(v),
            ValueChange::Decrease(v) => StepValue::Minus(v),
        }
    }
}
