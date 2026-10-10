//! State and delta acceptance tests: serde round-trips, safety clamps,
//! emergency stop.

use stop_core::{
    ActionDecision, ActionKind, ExecutionError, LightMode, RoomState, StepValue, TargetDevice,
    apply_action_to_state,
};

fn slots(device: TargetDevice, action: ActionKind, step: StepValue) -> ActionDecision {
    ActionDecision {
        target_device: device,
        action_kind: action,
        step_value: step,
    }
}

// --- Serde -------------------------------------------------------------------

#[test]
fn room_state_round_trips() {
    let state = RoomState::default();
    let json = serde_json::to_string(&state).expect("serialize");
    let back: RoomState = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(state, back);
}

#[test]
fn room_state_json_is_compact() {
    // The Jev model runs on a 16k context; the serialized state must stay tiny.
    let json = serde_json::to_string(&RoomState::default()).expect("serialize");
    assert!(
        json.len() < 1024,
        "state JSON too large: {} bytes",
        json.len()
    );
}

#[test]
fn decision_slots_round_trip() {
    let slots = slots(
        TargetDevice::Insufflator,
        ActionKind::SetPressure,
        StepValue::AbsoluteValue(14),
    );
    let json = serde_json::to_string(&slots).expect("serialize");
    let back: ActionDecision = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(back.target_device, TargetDevice::Insufflator);
    assert_eq!(back.step_value, StepValue::AbsoluteValue(14));
}

#[test]
fn dataset_example_state_round_trips() {
    // Shape from docs/INSTRUCTIONS.md section 4.2.
    let json = r#"{
        "lighting": { "primary_intensity_pct": 80, "field_mode": "Normal" },
        "endoscope": { "zoom_level": 2, "white_balance_locked": true, "irrigation_active": false },
        "insufflator": { "target_pressure_mmhg": 12, "gas_flow_l_min": 10, "is_active": true },
        "table": { "tilt_degrees": 0, "height_cm": 100 },
        "safety_interlock_active": false
    }"#;
    let state: RoomState = serde_json::from_str(json).expect("parse dataset state");
    assert_eq!(state.lighting.field_mode, LightMode::Normal);
    assert_eq!(state.insufflator.target_pressure_mmhg, 12);
}

// --- Safety clamps -----------------------------------------------------------

#[test]
fn pressure_never_exceeds_safety_cap() {
    let mut state = RoomState::default();
    // Request 40 mmHg absolute — must be clamped to the 25 mmHg cap.
    let report = apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::Insufflator,
            ActionKind::SetPressure,
            StepValue::AbsoluteValue(40),
        ),
    )
    .expect("apply");
    assert_eq!(state.insufflator.target_pressure_mmhg, 25);
    assert!(report.clamped);
}

#[test]
fn pressure_relative_step_clamps_at_cap() {
    let mut state = RoomState::default();
    state.insufflator.target_pressure_mmhg = 24;
    apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::Insufflator,
            ActionKind::IncreasePressure,
            StepValue::Plus(2),
        ),
    )
    .expect("apply");
    assert_eq!(state.insufflator.target_pressure_mmhg, 25);
}

#[test]
fn brightness_clamps_to_0_100() {
    let mut state = RoomState::default();
    state.lighting.primary_intensity_pct = 99;
    let report = apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::SurgicalLight,
            ActionKind::IncreaseBrightness,
            StepValue::Plus(2),
        ),
    )
    .expect("apply");
    assert_eq!(state.lighting.primary_intensity_pct, 100);
    assert!(report.clamped);

    let report = apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::SurgicalLight,
            ActionKind::DecreaseBrightness,
            StepValue::AbsoluteValue(-5),
        ),
    )
    .expect("apply");
    assert_eq!(state.lighting.primary_intensity_pct, 0);
    assert!(report.clamped);
}

#[test]
fn zoom_clamps_to_1_5() {
    let mut state = RoomState::default();
    state.endoscope.zoom_level = 5;
    apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::EndoscopeCamera,
            ActionKind::ZoomIn,
            StepValue::Plus(1),
        ),
    )
    .expect("apply");
    assert_eq!(state.endoscope.zoom_level, 5);

    apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::EndoscopeCamera,
            ActionKind::ZoomOut,
            StepValue::Minus(2),
        ),
    )
    .expect("apply");
    apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::EndoscopeCamera,
            ActionKind::ZoomOut,
            StepValue::Minus(2),
        ),
    )
    .expect("apply");
    assert_eq!(state.endoscope.zoom_level, 1);
}

#[test]
fn tilt_clamps_to_plus_minus_15_degrees() {
    let mut state = RoomState::default();
    state.table.tilt_degrees = 14;
    apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::OperatingTable,
            ActionKind::IncreaseTilt,
            StepValue::Plus(2),
        ),
    )
    .expect("apply");
    assert_eq!(state.table.tilt_degrees, 15);

    apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::OperatingTable,
            ActionKind::SetTilt,
            StepValue::AbsoluteValue(-40),
        ),
    )
    .expect("apply");
    assert_eq!(state.table.tilt_degrees, -15);
}

#[test]
fn height_clamps_to_70_130_cm() {
    let mut state = RoomState::default();
    state.table.height_cm = 100;
    let report = apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::OperatingTable,
            ActionKind::SetHeight,
            StepValue::AbsoluteValue(200),
        ),
    )
    .expect("apply");
    assert_eq!(state.table.height_cm, 130);
    assert!(report.clamped);

    let report = apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::OperatingTable,
            ActionKind::SetHeight,
            StepValue::AbsoluteValue(10),
        ),
    )
    .expect("apply");
    assert_eq!(state.table.height_cm, 70);
    assert!(report.clamped);
}

#[test]
fn height_set_applies_within_range() {
    let mut state = RoomState::default();
    let report = apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::OperatingTable,
            ActionKind::SetHeight,
            StepValue::AbsoluteValue(90),
        ),
    )
    .expect("apply");
    assert_eq!(state.table.height_cm, 90);
    assert!(!report.clamped);
    assert!(report.detail.contains("Height 90 cm"));
}

// --- Emergency stop ----------------------------------------------------------

#[test]
fn emergency_stop_forces_interlock_and_shuts_down() {
    let mut state = RoomState::default();
    state.endoscope.irrigation_active = true;
    apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::Insufflator,
            ActionKind::EmergencyStop,
            StepValue::Zero,
        ),
    )
    .expect("apply");
    assert!(state.safety_interlock_active);
    assert!(!state.insufflator.is_active);
    assert!(!state.endoscope.irrigation_active);
}

#[test]
fn emergency_stop_is_device_agnostic() {
    // Global flag: must take effect even without a target device.
    let mut state = RoomState::default();
    state.endoscope.irrigation_active = true;
    apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::None,
            ActionKind::EmergencyStop,
            StepValue::Zero,
        ),
    )
    .expect("apply");
    assert!(state.safety_interlock_active);
    assert!(!state.insufflator.is_active);
    assert!(!state.endoscope.irrigation_active);
}

// --- No-ops and error matrix -------------------------------------------------

#[test]
fn target_device_none_is_noop() {
    let mut state = RoomState::default();
    let before = state.clone();
    let report = apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::None,
            ActionKind::IncreaseBrightness,
            StepValue::Plus(2),
        ),
    )
    .expect("apply");
    assert_eq!(state, before);
    assert_eq!(report.action_kind, ActionKind::IncreaseBrightness);
}

#[test]
fn idle_action_is_noop() {
    let mut state = RoomState::default();
    let before = state.clone();
    let report = apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::SurgicalLight,
            ActionKind::Idle,
            StepValue::Zero,
        ),
    )
    .expect("apply");
    assert_eq!(state, before);
    assert_eq!(report.detail, "Idle -> no state change");
}

#[test]
fn mismatched_device_action_is_rejected() {
    let mut state = RoomState::default();
    let err = apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::OperatingTable,
            ActionKind::IncreaseBrightness,
            StepValue::Plus(1),
        ),
    )
    .expect_err("must reject");
    assert!(matches!(err, ExecutionError::InvalidTargetAction { .. }));
}

#[test]
fn unknown_light_mode_is_rejected() {
    let mut state = RoomState::default();
    let err = apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::SurgicalLight,
            ActionKind::SetLightMode,
            StepValue::AbsoluteValue(7),
        ),
    )
    .expect_err("must reject");
    assert!(matches!(err, ExecutionError::UnknownLightMode(7)));
}

#[test]
fn light_mode_set_applies() {
    let mut state = RoomState::default();
    apply_action_to_state(
        &mut state,
        &slots(
            TargetDevice::SurgicalLight,
            ActionKind::SetLightMode,
            StepValue::AbsoluteValue(2),
        ),
    )
    .expect("apply");
    assert_eq!(state.lighting.field_mode, LightMode::AmbientRed);
}
