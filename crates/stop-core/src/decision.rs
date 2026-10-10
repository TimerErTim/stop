//! Decision slot schema emitted by the JevK5-2B typed parallel heads.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionDecision {
    /// Signalisiert dem Orchestrator, ob ein weiterer Durchlauf nötig ist
    pub further_action_needed: bool,

    /// Zielgerät der aktuellen Teilaktion
    pub target_device: TargetDevice,

    /// Konkrete Operation auf dem Zielgerät
    pub action_kind: ActionKind,

    /// Relativer Schrittwert / Diskrete Anpassung
    pub step_value: StepValue,

    /// Sicherheitsrelevante Verifikation nötig (z. B. Überdruck oder OP-Tischneigung)
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
