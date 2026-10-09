//! Deterministic application of decision slots to the `RoomState`.
//!
//! Every action clamps to the physical safety envelope of the device:
//! brightness 0-100 %, zoom 1-5, pressure capped at 25 mmHg, table tilt
//! -15..+15 degrees. `EmergencyStop` forces the safety interlock and shuts
//! down insufflation and irrigation.

use crate::decision::{ActionDecision, ActionKind, StepValue, TargetDevice};
use crate::error::ExecutionError;
use crate::state::{LightMode, RoomState};

/// Report of a single applied state delta (feeds the GUI HUD telemetry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppliedActionReport {
    pub target_device: TargetDevice,
    pub action_kind: ActionKind,
    /// Human-readable HUD line, e.g. `Light -> Dim (-2)`.
    pub detail: String,
    /// True when a safety clamp changed the requested value.
    pub clamped: bool,
}

/// Applies one pass of decision slots to `current_state`.
///
/// `TargetDevice::None` and `ActionKind::Idle` are no-ops and return an
/// `Idle` report. Action/device combinations outside the supported matrix
/// return [`ExecutionError::InvalidTargetAction`].
pub fn apply_action_to_state(
    current_state: &mut RoomState,
    prediction: &ActionDecision,
) -> Result<AppliedActionReport, ExecutionError> {
    let step = prediction.step_value;
    let device = &prediction.target_device;
    let action = &prediction.action_kind;

    // Idle is a no-op.
    if *action == ActionKind::Idle {
        return Ok(idle_report(device, action));
    }

    // EmergencyStop is device-agnostic: it always takes effect.
    if *action == ActionKind::EmergencyStop && *device != TargetDevice::None {
        current_state.safety_interlock_active = true;
        current_state.insufflator.is_active = false;
        current_state.endoscope.irrigation_active = false;
        return Ok(AppliedActionReport {
            target_device: *device,
            action_kind: *action,
            detail: "EmergencyStop -> interlock ON, insufflation OFF".to_string(),
            clamped: false,
        });
    }

    match device {
        TargetDevice::None => Ok(idle_report(device, action)),
        TargetDevice::SurgicalLight => match action {
            ActionKind::IncreaseBrightness | ActionKind::DecreaseBrightness => {
                adjust_brightness(current_state, *action, step)
            }
            ActionKind::SetLightMode => set_light_mode(current_state, step),
            _ => Err(ExecutionError::InvalidTargetAction {
                target_device: *device,
                action_kind: *action,
            }),
        },
        TargetDevice::EndoscopeCamera => match action {
            ActionKind::ZoomIn | ActionKind::ZoomOut => adjust_zoom(current_state, *action, step),
            ActionKind::ToggleIrrigation => {
                current_state.endoscope.irrigation_active =
                    !current_state.endoscope.irrigation_active;
                Ok(AppliedActionReport {
                    target_device: *device,
                    action_kind: *action,
                    detail: format!(
                        "Irrigation -> {}",
                        on_off(current_state.endoscope.irrigation_active)
                    ),
                    clamped: false,
                })
            }
            _ => Err(ExecutionError::InvalidTargetAction {
                target_device: *device,
                action_kind: *action,
            }),
        },
        TargetDevice::Insufflator => match action {
            ActionKind::AdjustPressure => adjust_pressure(current_state, step),
            ActionKind::ToggleInsufflation => {
                current_state.insufflator.is_active = !current_state.insufflator.is_active;
                Ok(AppliedActionReport {
                    target_device: *device,
                    action_kind: *action,
                    detail: format!(
                        "Insufflation -> {}",
                        on_off(current_state.insufflator.is_active)
                    ),
                    clamped: false,
                })
            }
            _ => Err(ExecutionError::InvalidTargetAction {
                target_device: *device,
                action_kind: *action,
            }),
        },
        TargetDevice::OperatingTable => match action {
            ActionKind::TiltTable => adjust_tilt(current_state, step),
            _ => Err(ExecutionError::InvalidTargetAction {
                target_device: *device,
                action_kind: *action,
            }),
        },
    }
}

fn idle_report(device: &TargetDevice, action: &ActionKind) -> AppliedActionReport {
    AppliedActionReport {
        target_device: *device,
        action_kind: *action,
        detail: "Idle -> no state change".to_string(),
        clamped: false,
    }
}

fn on_off(v: bool) -> &'static str {
    if v { "ON" } else { "OFF" }
}

/// Applies brightness delta or absolute value, clamped to 0-100 %.
fn adjust_brightness(
    state: &mut RoomState,
    action: ActionKind,
    step: StepValue,
) -> Result<AppliedActionReport, ExecutionError> {
    let current = i16::from(state.lighting.primary_intensity_pct);
    let requested = if step.is_absolute() {
        step.as_i16()
    } else {
        current + step.as_i16()
    };
    let (applied, clamped) = state.lighting.set_intensity_pct(requested);
    let applied = i16::from(applied);
    Ok(AppliedActionReport {
        target_device: TargetDevice::SurgicalLight,
        action_kind: action,
        detail: format!(
            "Light -> Brightness {}% (requested {}%)",
            applied, requested
        ),
        clamped,
    })
}

/// Sets the light field mode. Mode codes: 0=Normal, 1=CavityFocus, 2=AmbientRed.
fn set_light_mode(
    state: &mut RoomState,
    step: StepValue,
) -> Result<AppliedActionReport, ExecutionError> {
    let mode = match step.as_i16() {
        0 => LightMode::Normal,
        1 => LightMode::CavityFocus,
        2 => LightMode::AmbientRed,
        other => return Err(ExecutionError::UnknownLightMode(other)),
    };
    state.lighting.field_mode = mode.clone();
    Ok(AppliedActionReport {
        target_device: TargetDevice::SurgicalLight,
        action_kind: ActionKind::SetLightMode,
        detail: format!("Light -> Mode {:?}", mode),
        clamped: false,
    })
}

/// Applies zoom delta or absolute value, clamped to levels 1-5.
fn adjust_zoom(
    state: &mut RoomState,
    action: ActionKind,
    step: StepValue,
) -> Result<AppliedActionReport, ExecutionError> {
    let current = i16::from(state.endoscope.zoom_level);
    let requested = if step.is_absolute() {
        step.as_i16()
    } else {
        current + step.as_i16()
    };
    let (applied, clamped) = state.endoscope.set_zoom_level(requested);
    let applied = i16::from(applied);
    Ok(AppliedActionReport {
        target_device: TargetDevice::EndoscopeCamera,
        action_kind: action,
        detail: format!("Camera -> Zoom {} (requested {})", applied, requested),
        clamped,
    })
}

/// Applies pressure delta or absolute value, hard-capped at 25 mmHg.
fn adjust_pressure(
    state: &mut RoomState,
    step: StepValue,
) -> Result<AppliedActionReport, ExecutionError> {
    let current = i16::from(state.insufflator.target_pressure_mmhg);
    let requested = if step.is_absolute() {
        step.as_i16()
    } else {
        current + step.as_i16()
    };
    let (applied, clamped) = state.insufflator.set_target_pressure_mmhg(requested);
    let applied = i16::from(applied);
    Ok(AppliedActionReport {
        target_device: TargetDevice::Insufflator,
        action_kind: ActionKind::AdjustPressure,
        detail: format!(
            "Insufflator -> {} mmHg (requested {} mmHg)",
            applied, requested
        ),
        clamped,
    })
}

/// Applies table tilt delta or absolute value, clamped to -15..+15 degrees.
fn adjust_tilt(
    state: &mut RoomState,
    step: StepValue,
) -> Result<AppliedActionReport, ExecutionError> {
    let current = i16::from(state.table.tilt_degrees);
    let requested = if step.is_absolute() {
        step.as_i16()
    } else {
        current + step.as_i16()
    };
    let (applied, clamped) = state.table.set_tilt_degrees(requested);
    let applied = i16::from(applied);
    Ok(AppliedActionReport {
        target_device: TargetDevice::OperatingTable,
        action_kind: ActionKind::TiltTable,
        detail: format!(
            "Table -> Tilt {} deg (requested {} deg)",
            applied, requested
        ),
        clamped,
    })
}
