//! `RoomState`: compact serializable OR room snapshot for the Jev 16k context.

use serde::{Deserialize, Serialize};

/// Safety cap for insufflator target pressure (medical upper bound, mmHg).
pub const MAX_PRESSURE_MMHG: u8 = 25;
/// Upper bound for insufflator gas flow (l/min).
pub const MAX_GAS_FLOW_L_MIN: i16 = 45;
/// Brightness range in percent.
pub const BRIGHTNESS_MIN: i16 = 0;
pub const BRIGHTNESS_MAX: i16 = 100;
/// Endoscope zoom levels.
pub const ZOOM_MIN: i8 = 1;
pub const ZOOM_MAX: i8 = 5;
/// Table tilt range in degrees (-15 Trendelenburg, +15 anti-Trendelenburg).
pub const TILT_MIN: i8 = -15;
pub const TILT_MAX: i8 = 15;
/// Table height range in cm.
pub const HEIGHT_MIN: u8 = 70;
pub const HEIGHT_MAX: u8 = 130;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RoomState {
    pub lighting: LightingState,
    pub endoscope: EndoscopeState,
    pub insufflator: InsufflatorState,
    pub table: TableState,
    pub safety_interlock_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LightingState {
    pub primary_intensity_pct: u8, // 0 - 100
    pub field_mode: LightMode,     // Normal, CavityFocus, AmbientRed
}

impl LightingState {
    /// Sets the intensity, clamped to 0-100 %. Returns `(applied, clamped)`.
    pub fn set_intensity_pct(&mut self, requested: i16) -> (u8, bool) {
        let applied = requested.clamp(BRIGHTNESS_MIN, BRIGHTNESS_MAX);
        let clamped = applied != requested;
        self.primary_intensity_pct = applied as u8;
        (applied as u8, clamped)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LightMode {
    Normal,
    CavityFocus,
    AmbientRed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EndoscopeState {
    pub zoom_level: i8, // 1 bis 5
    pub white_balance_locked: bool,
    pub irrigation_active: bool,
}

impl EndoscopeState {
    /// Sets the zoom level, clamped to 1-5. Returns `(applied, clamped)`.
    pub fn set_zoom_level(&mut self, requested: i16) -> (i8, bool) {
        let applied = requested.clamp(i16::from(ZOOM_MIN), i16::from(ZOOM_MAX));
        let clamped = applied != requested;
        self.zoom_level = applied as i8;
        (applied as i8, clamped)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InsufflatorState {
    pub target_pressure_mmhg: u8, // typisch 12 - 15 mmHg
    pub gas_flow_l_min: u8,
    pub is_active: bool,
}

impl InsufflatorState {
    /// Sets the target pressure, hard-capped at [`MAX_PRESSURE_MMHG`].
    /// Returns `(applied, clamped)`.
    pub fn set_target_pressure_mmhg(&mut self, requested: i16) -> (u8, bool) {
        let applied = requested.clamp(0, i16::from(MAX_PRESSURE_MMHG));
        let clamped = applied != requested;
        self.target_pressure_mmhg = applied as u8;
        (applied as u8, clamped)
    }

    /// Sets the gas flow, clamped to 0-45 l/min. Returns `(applied, clamped)`.
    pub fn set_gas_flow_l_min(&mut self, requested: i16) -> (u8, bool) {
        let applied = requested.clamp(0, MAX_GAS_FLOW_L_MIN);
        let clamped = applied != requested;
        self.gas_flow_l_min = applied as u8;
        (applied as u8, clamped)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TableState {
    pub tilt_degrees: i8, // -15 (Trendelenburg) bis +15 (Anti-Trendelenburg)
    pub height_cm: u8,
}

impl TableState {
    /// Sets the tilt, clamped to -15..+15 degrees. Returns `(applied, clamped)`.
    pub fn set_tilt_degrees(&mut self, requested: i16) -> (i8, bool) {
        let applied = requested.clamp(i16::from(TILT_MIN), i16::from(TILT_MAX));
        let clamped = applied != requested;
        self.tilt_degrees = applied as i8;
        (applied as i8, clamped)
    }

    /// Sets the height, clamped to 70-130 cm. Returns `(applied, clamped)`.
    pub fn set_height_cm(&mut self, requested: i16) -> (u8, bool) {
        let applied = requested.clamp(i16::from(HEIGHT_MIN), i16::from(HEIGHT_MAX));
        let clamped = applied != requested;
        self.height_cm = applied as u8;
        (applied as u8, clamped)
    }
}

impl Default for RoomState {
    fn default() -> Self {
        Self {
            lighting: LightingState {
                primary_intensity_pct: 80,
                field_mode: LightMode::Normal,
            },
            endoscope: EndoscopeState {
                zoom_level: 2,
                white_balance_locked: true,
                irrigation_active: false,
            },
            insufflator: InsufflatorState {
                target_pressure_mmhg: 12,
                gas_flow_l_min: 10,
                is_active: true,
            },
            table: TableState {
                tilt_degrees: 0,
                height_cm: 100,
            },
            safety_interlock_active: false,
        }
    }
}
