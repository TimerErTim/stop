//! Pure tiny-skia scene renderer for the schematic OR room.
//!
//! `render_scene` draws the complete room into a pixmap from a [`RoomState`]
//! snapshot plus an animation clock. No GPU, no egui, no I/O: deterministic
//! and unit-testable on raw pixels.

use tiny_skia::{Color, FillRule, Paint, Path, PathBuilder, Pixmap, Rect, Transform};

use stop_core::state::{
    HEIGHT_MAX, HEIGHT_MIN, LightMode, MAX_GAS_FLOW_L_MIN, MAX_PRESSURE_MMHG, RoomState,
};

/// Pressure above which the insufflator bar turns to the warn color.
const PRESSURE_WARN_MMHG: u8 = 15;

// Palette (schematic OR look, dark room). u8 triplets; tiny-skia colors
// are built at draw time with `palette()`.

/// u8 palette entry to tiny-skia color.
fn palette(r: u8, g: u8, b: u8) -> Color {
    Color::from_rgba8(r, g, b, 255)
}

const BG_NORMAL: (u8, u8, u8) = (23, 26, 33);
const BG_AMBIENT_RED: (u8, u8, u8) = (61, 15, 15);
const TABLE_MATTRESS: (u8, u8, u8) = (209, 214, 222);
const TABLE_PEDESTAL: (u8, u8, u8) = (89, 94, 105);
const LAMP_HOUSING: (u8, u8, u8) = (69, 74, 84);
const CONE_WARM: (u8, u8, u8) = (255, 240, 191);
const CONE_RED: (u8, u8, u8) = (255, 89, 69);
const MONITOR_BEZEL: (u8, u8, u8) = (20, 23, 26);
const MONITOR_SCREEN: (u8, u8, u8) = (13, 33, 20);
const TISSUE: (u8, u8, u8) = (189, 97, 92);
const TISSUE_DARK: (u8, u8, u8) = (150, 69, 69);
const DROPLET: (u8, u8, u8) = (120, 181, 255);
const INSUFFLATOR_PANEL: (u8, u8, u8) = (31, 33, 41);
const BAR_OK: (u8, u8, u8) = (79, 199, 120);
const BAR_WARN: (u8, u8, u8) = (230, 89, 61);
const GAUGE_TRACK: (u8, u8, u8) = (56, 60, 72);
const INTERLOCK_RED: (u8, u8, u8) = (217, 28, 28);
const LED_OFF: (u8, u8, u8) = (52, 56, 66);
const LED_ON: (u8, u8, u8) = (79, 199, 120);
const LED_STANDBY: (u8, u8, u8) = (230, 160, 61);

/// Builds an opaque paint from a palette triplet.
fn paint_of(color: (u8, u8, u8)) -> Paint<'static> {
    let mut paint = Paint::default();
    paint.set_color(palette(color.0, color.1, color.2));
    paint
}

/// Background fill color from a palette triplet.
fn palette_bg(color: (u8, u8, u8)) -> Color {
    palette(color.0, color.1, color.2)
}

/// Geometry derived from the pixmap size (relative coordinates). Public
/// because the egui overlay in [`crate::app`] maps annotation anchors from
/// scene space onto the displayed image.
pub struct Layout {
    width: f32,
    height: f32,
}

impl Layout {
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    fn lamp(&self) -> (f32, f32) {
        (self.width * 0.5, self.height * 0.10)
    }

    /// Cone apex sits just below the lamp housing.
    fn cone_apex(&self) -> (f32, f32) {
        (self.width * 0.5, self.height * 0.14)
    }

    /// Cone base lands on the mattress top surface.
    fn cone_base_y(&self) -> f32 {
        self.mattress_top()
    }

    fn cone_base_half_width(&self) -> f32 {
        self.width * 0.20
    }

    /// Top edge of the mattress (the table surface).
    fn mattress_top(&self) -> f32 {
        let (cy, hh) = (self.table_center_y(), self.height * 0.035);
        cy - hh
    }

    fn table_center(&self) -> (f32, f32) {
        (self.width * 0.5, self.table_center_y())
    }

    fn table_center_y(&self) -> f32 {
        self.height * 0.70
    }

    fn mattress_half(&self) -> (f32, f32) {
        (self.width * 0.14, self.height * 0.035)
    }

    /// Full-height pedestal column (the table leg). The visible height is
    /// scaled by `room.table.height_cm` in [`draw_table`], anchored at the
    /// floor line.
    fn pedestal(&self) -> Rect {
        let (cx, cy) = self.table_center();
        Rect::from_xywh(
            cx - self.width * 0.02,
            cy + self.height * 0.035,
            self.width * 0.04,
            self.height * 0.16,
        )
        .unwrap()
    }

    /// Floor line the pedestal stands on.
    fn floor_y(&self) -> f32 {
        self.pedestal().y() + self.pedestal().height()
    }

    fn monitor(&self) -> Rect {
        Rect::from_xywh(
            self.width * 0.67,
            self.height * 0.15,
            self.width * 0.27,
            self.height * 0.32,
        )
        .unwrap()
    }

    fn monitor_screen(&self) -> Rect {
        let m = self.monitor();
        Rect::from_xywh(
            m.x() + m.width() * 0.06,
            m.y() + m.height() * 0.08,
            m.width() * 0.88,
            m.height() * 0.84,
        )
        .unwrap()
    }

    /// The two insufflator gauges: identical size and vertical span,
    /// side by side, top-aligned on one baseline. `index` 0 = pressure,
    /// 1 = flow.
    fn insufflator_gauge(&self, index: usize) -> Rect {
        let bar_w = self.width * 0.024;
        let bar_h = self.height * 0.34;
        let bar_y = self.height * 0.22;
        let left_x = self.width * 0.055;
        let gap = bar_w * 0.6;
        Rect::from_xywh(
            left_x + (bar_w + gap) * index as f32,
            bar_y,
            bar_w,
            bar_h,
        )
        .unwrap()
    }

    /// Pressure gauge bar (left).
    fn pressure_bar(&self) -> Rect {
        self.insufflator_gauge(0)
    }

    /// Gas-flow gauge bar (right of pressure, same height and baseline).
    fn flow_bar(&self) -> Rect {
        self.insufflator_gauge(1)
    }

    /// Insufflator panel enclosing both gauge bars plus a value header row.
    fn insufflator_panel(&self) -> Rect {
        let pressure = self.pressure_bar();
        let flow = self.flow_bar();
        Rect::from_ltrb(
            pressure.x() - self.width * 0.015,
            pressure.y() - self.height * 0.08,
            flow.x() + flow.width() + self.width * 0.015,
            pressure.y() + pressure.height() + self.height * 0.045,
        )
        .expect("insufflator panel rect is finite")
    }

    /// Top row inside the insufflator panel: LED + (overlay) value labels.
    fn insufflator_header_y(&self) -> f32 {
        self.insufflator_panel().y() + self.height * 0.035
    }

    fn interlock_border(&self) -> f32 {
        self.width * 0.008
    }

    /// Anchor point inside the device each annotation labels, in scene
    /// pixel coordinates. Consumed by the egui overlay in [`crate::app`].
    pub fn annotation_anchor(&self, annotation: Annotation) -> (f32, f32) {
        match annotation {
            Annotation::Light => self.lamp(),
            Annotation::Camera => {
                let screen = self.monitor_screen();
                (screen.x() + screen.width() * 0.5, screen.y() - self.height * 0.01)
            }
            Annotation::InsufflatorPressure => {
                let bar = self.pressure_bar();
                (bar.x() + bar.width() * 0.5, self.insufflator_header_y())
            }
            Annotation::InsufflatorFlow => {
                let bar = self.flow_bar();
                (
                    bar.x() + bar.width() * 0.5,
                    self.insufflator_header_y() + self.height * 0.05,
                )
            }
            Annotation::Table => {
                let pedestal = self.pedestal();
                (
                    pedestal.x() + pedestal.width() * 0.5,
                    self.floor_y() - self.height * 0.01,
                )
            }
        }
    }
}

/// Scene objects the UI labels with text annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Annotation {
    /// Surgical light (cone + lamp).
    Light,
    /// Endoscope camera monitor.
    Camera,
    /// Insufflator pressure gauge.
    InsufflatorPressure,
    /// Insufflator gas-flow gauge.
    InsufflatorFlow,
    /// Operating table.
    Table,
}

impl Annotation {
    /// All annotations in draw order.
    pub const ALL: [Annotation; 5] = [
        Annotation::Light,
        Annotation::Camera,
        Annotation::InsufflatorPressure,
        Annotation::InsufflatorFlow,
        Annotation::Table,
    ];

    /// Short device name for the on-scene label.
    pub fn label(self) -> &'static str {
        match self {
            Annotation::Light => "light",
            Annotation::Camera => "camera",
            Annotation::InsufflatorPressure => "pressure",
            Annotation::InsufflatorFlow => "flow",
            Annotation::Table => "table",
        }
    }
}

/// Renders the full OR room into `pixmap` (the background is painted first).
pub fn render_scene(pixmap: &mut Pixmap, room: &RoomState, anim_time_secs: f32) {
    let layout = Layout::new(pixmap.width() as f32, pixmap.height() as f32);
    let ambient_red = room.lighting.field_mode == LightMode::AmbientRed;

    let bg = palette_bg(if ambient_red {
        BG_AMBIENT_RED
    } else {
        BG_NORMAL
    });
    pixmap.fill(bg);

    if room.safety_interlock_active {
        draw_interlock_border(pixmap, &layout);
    }

    draw_light(pixmap, &layout, room, ambient_red);
    draw_table(pixmap, &layout, room);
    draw_monitor(pixmap, &layout, room, anim_time_secs);
    draw_insufflator(pixmap, &layout, room);
}

fn fill_path(pixmap: &mut Pixmap, path: &Path, paint: &Paint) {
    pixmap.fill_path(path, paint, FillRule::Winding, Transform::identity(), None);
}

fn fill_rect(pixmap: &mut Pixmap, rect: &Rect, paint: &Paint) {
    pixmap.fill_rect(*rect, paint, Transform::identity(), None);
}

fn polygon(points: &[(f32, f32)]) -> Option<Path> {
    let mut builder = PathBuilder::new();
    let (first_x, first_y) = points.first().copied()?;
    builder.move_to(first_x, first_y);
    for &(x, y) in &points[1..] {
        builder.line_to(x, y);
    }
    builder.close();
    builder.finish()
}

fn circle(cx: f32, cy: f32, radius: f32) -> Option<Path> {
    let mut builder = PathBuilder::new();
    builder.push_circle(cx, cy, radius);
    builder.finish()
}

fn draw_interlock_border(pixmap: &mut Pixmap, layout: &Layout) {
    let border = layout.interlock_border();
    let w = layout.width;
    let h = layout.height;
    let edges = [
        Rect::from_xywh(0.0, 0.0, w, border).unwrap(),
        Rect::from_xywh(0.0, h - border, w, border).unwrap(),
        Rect::from_xywh(0.0, 0.0, border, h).unwrap(),
        Rect::from_xywh(w - border, 0.0, border, h).unwrap(),
    ];
    let paint = paint_of(INTERLOCK_RED);
    for edge in &edges {
        fill_rect(pixmap, edge, &paint);
    }
}

fn draw_light(pixmap: &mut Pixmap, layout: &Layout, room: &RoomState, ambient_red: bool) {
    // Cone: opacity scales with brightness; tint follows the field mode.
    let (apex_x, apex_y) = layout.cone_apex();
    let base_y = layout.cone_base_y();
    let half = layout.cone_base_half_width();
    if let Some(cone) = polygon(&[
        (apex_x, apex_y),
        (apex_x - half, base_y),
        (apex_x + half, base_y),
    ]) {
        let intensity = f32::from(room.lighting.primary_intensity_pct) / 100.0;
        let alpha = intensity * 0.75;
        let tint = if ambient_red { CONE_RED } else { CONE_WARM };
        let mut paint = paint_of(tint);
        paint.set_color(
            Color::from_rgba(
                f32::from(tint.0) / 255.0,
                f32::from(tint.1) / 255.0,
                f32::from(tint.2) / 255.0,
                alpha,
            )
            .expect("tint components are normalized"),
        );
        fill_path(pixmap, &cone, &paint);
    }

    // Lamp housing above the apex.
    let (lamp_x, lamp_y) = layout.lamp();
    let housing = Rect::from_xywh(
        lamp_x - layout.width * 0.06,
        lamp_y - layout.height * 0.03,
        layout.width * 0.12,
        layout.height * 0.05,
    )
    .unwrap();
    fill_rect(pixmap, &housing, &paint_of(LAMP_HOUSING));
}

fn draw_table(pixmap: &mut Pixmap, layout: &Layout, room: &RoomState) {
    let pedestal = layout.pedestal();
    let full_h = pedestal.height();
    let floor_y = pedestal.y() + full_h;

    // Height scales the pedestal column: taller table -> taller pedestal.
    // Anchored at the floor line, growing upward.
    let height_fraction = f32::from(room.table.height_cm - HEIGHT_MIN)
        / f32::from(HEIGHT_MAX - HEIGHT_MIN);
    let pedestal_h = full_h * (0.45 + 0.55 * height_fraction);
    let pedestal_top = floor_y - pedestal_h;
    let pedestal_scaled =
        Rect::from_xywh(pedestal.x(), pedestal_top, pedestal.width(), pedestal_h).unwrap();
    fill_rect(pixmap, &pedestal_scaled, &paint_of(TABLE_PEDESTAL));

    // Mattress rests exactly on the pedestal top; tilt pivots around the
    // mattress/pedestal contact point so top and leg never separate.
    let (cx, _) = layout.table_center();
    let (hw, hh) = layout.mattress_half();
    let mattress_cy = pedestal_top - hh;
    let mattress =
        PathBuilder::from_rect(Rect::from_xywh(cx - hw, mattress_cy - hh, hw * 2.0, hh * 2.0)
            .unwrap());
    let tilt = Transform::from_rotate_at(f32::from(room.table.tilt_degrees), cx, mattress_cy);
    if let Some(rotated) = mattress.transform(tilt) {
        fill_path(pixmap, &rotated, &paint_of(TABLE_MATTRESS));
    }
}

fn draw_monitor(pixmap: &mut Pixmap, layout: &Layout, room: &RoomState, anim_time_secs: f32) {
    let monitor = layout.monitor();
    let screen = layout.monitor_screen();
    fill_rect(pixmap, &monitor, &paint_of(MONITOR_BEZEL));
    fill_rect(pixmap, &screen, &paint_of(MONITOR_SCREEN));

    // Laparoscopic target: tissue disc scales with the zoom level.
    let cx = screen.x() + screen.width() * 0.5;
    let cy = screen.y() + screen.height() * 0.5;
    let radius = layout.width * 0.018 * f32::from(room.endoscope.zoom_level);
    if let Some(outer) = circle(cx, cy, radius) {
        fill_path(pixmap, &outer, &paint_of(TISSUE));
    }
    let inner_radius = radius * 0.55;
    let inner_cx = cx + radius * 0.15;
    let inner_cy = cy + radius * 0.10;
    if let Some(inner) = circle(inner_cx, inner_cy, inner_radius) {
        fill_path(pixmap, &inner, &paint_of(TISSUE_DARK));
    }

    // White-balance lock LED in the top-right corner of the bezel.
    let led_r = layout.width * 0.006;
    let led_cx = monitor.x() + monitor.width() - led_r * 3.0;
    let led_cy = monitor.y() + led_r * 3.0;
    let color = if room.endoscope.white_balance_locked {
        LED_ON
    } else {
        LED_OFF
    };
    if let Some(led) = circle(led_cx, led_cy, led_r) {
        fill_path(pixmap, &led, &paint_of(color));
    }

    // Irrigation droplets fall across the screen while active.
    if room.endoscope.irrigation_active {
        let h = layout.height;
        for i in 0..6u32 {
            let x = screen.x() + screen.width() * (0.12 + 0.15 * i as f32);
            let fall = (anim_time_secs * 90.0 + i as f32 * 37.0) % screen.height();
            let y = screen.y() + fall;
            if let Some(drop) = circle(x, y, h * 0.008) {
                fill_path(pixmap, &drop, &paint_of(DROPLET));
            }
        }
    }
}

fn draw_insufflator(pixmap: &mut Pixmap, layout: &Layout, room: &RoomState) {
    let panel = layout.insufflator_panel();
    fill_rect(pixmap, &panel, &paint_of(INSUFFLATOR_PANEL));

    draw_pressure_bar(pixmap, layout, room);
    draw_flow_bar(pixmap, layout, room);

    // Insufflation activity LED in the panel header row.
    let led_r = layout.width * 0.007;
    let led_cx = panel.x() + panel.width() * 0.5;
    let led_cy = layout.insufflator_header_y();
    let color = if room.insufflator.is_active {
        LED_ON
    } else {
        LED_STANDBY
    };
    if let Some(led) = circle(led_cx, led_cy, led_r) {
        fill_path(pixmap, &led, &paint_of(color));
    }
}

/// One vertical gauge: dark track, bottom-up fill, warn color, threshold
/// marker. Identical rendering for both insufflator gauges.
fn draw_gauge(pixmap: &mut Pixmap, bar: &Rect, fill_fraction: f32, warn_fraction: f32, warn: bool) {
    // Full track behind the fill so the gauge reads as one bar.
    fill_rect(pixmap, bar, &paint_of(GAUGE_TRACK));

    // Fill from the bottom, proportional to the value.
    if fill_fraction > 0.0 {
        let fill_h = bar.height() * fill_fraction.min(1.0);
        let fill = Rect::from_xywh(
            bar.x(),
            bar.y() + bar.height() - fill_h,
            bar.width(),
            fill_h,
        )
        .unwrap();
        let color = if warn { BAR_WARN } else { BAR_OK };
        fill_rect(pixmap, &fill, &paint_of(color));
    }

    // Threshold marker line at the warn boundary.
    let marker_y = bar.y() + bar.height() * (1.0 - warn_fraction);
    let marker = Rect::from_xywh(bar.x(), marker_y - 2.0, bar.width(), 4.0).unwrap();
    fill_rect(pixmap, &marker, &paint_of(BAR_WARN));
}

/// Pressure gauge: 0..MAX_PRESSURE_MMHG with warn above 15 mmHg.
fn draw_pressure_bar(pixmap: &mut Pixmap, layout: &Layout, room: &RoomState) {
    let bar = layout.pressure_bar();
    let warn_fraction = f32::from(PRESSURE_WARN_MMHG) / f32::from(MAX_PRESSURE_MMHG);
    let fill_fraction = f32::from(room.insufflator.target_pressure_mmhg) / f32::from(MAX_PRESSURE_MMHG);
    let warn = room.insufflator.target_pressure_mmhg > PRESSURE_WARN_MMHG;
    draw_gauge(pixmap, &bar, fill_fraction, warn_fraction, warn);
}

/// Gas-flow gauge: 0..MAX_GAS_FLOW_L_MIN with warn above 30 l/min.
fn draw_flow_bar(pixmap: &mut Pixmap, layout: &Layout, room: &RoomState) {
    const FLOW_WARN_L_MIN: u8 = 30;
    let bar = layout.flow_bar();
    let warn_fraction = f32::from(FLOW_WARN_L_MIN) / f32::from(MAX_GAS_FLOW_L_MIN);
    let fill_fraction =
        f32::from(room.insufflator.gas_flow_l_min) / f32::from(MAX_GAS_FLOW_L_MIN);
    let warn = room.insufflator.gas_flow_l_min > FLOW_WARN_L_MIN;
    draw_gauge(pixmap, &bar, fill_fraction, warn_fraction, warn);
}

#[cfg(test)]
mod tests {
    use super::*;
    use stop_core::state::{EndoscopeState, InsufflatorState, LightingState, TableState};

    const W: u32 = 1280;
    const H: u32 = 800;

    fn base_room() -> RoomState {
        RoomState {
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

    fn render(room: &RoomState) -> Pixmap {
        let mut pixmap = Pixmap::new(W, H).unwrap();
        render_scene(&mut pixmap, room, 0.5);
        pixmap
    }

    fn pixel(pixmap: &Pixmap, x: u32, y: u32) -> (u8, u8, u8) {
        let p = pixmap.pixel(x, y).unwrap();
        // Premultiplied; with alpha 255 the channels are plain values.
        assert_eq!(p.alpha(), 255, "background must stay opaque");
        (p.red(), p.green(), p.blue())
    }

    fn diff_count(a: &Pixmap, b: &Pixmap) -> usize {
        a.data()
            .iter()
            .zip(b.data().iter())
            .filter(|(x, y)| x != y)
            .count()
    }

    #[test]
    fn brightness_changes_cone_intensity() {
        let mut dark = base_room();
        dark.lighting.primary_intensity_pct = 0;
        let mut bright = base_room();
        bright.lighting.primary_intensity_pct = 100;

        let dark_px = pixel(&render(&dark), W / 2 - 30, (H as f32 * 0.35) as u32);
        let bright_px = pixel(&render(&bright), W / 2 - 30, (H as f32 * 0.35) as u32);
        let sum = |(r, g, b): (u8, u8, u8)| u32::from(r) + u32::from(g) + u32::from(b);
        assert!(
            sum(bright_px) > sum(dark_px) + 100,
            "cone pixel must brighten: {dark_px:?} vs {bright_px:?}"
        );
    }

    #[test]
    fn ambient_red_tints_room_and_cone() {
        let mut room = base_room();
        room.lighting.field_mode = LightMode::AmbientRed;
        room.lighting.primary_intensity_pct = 0;

        let (r, g, b) = pixel(&render(&room), 20, H - 20);
        assert!(
            r > g + 20 && r > b + 20,
            "background red-dominant: {r} {g} {b}"
        );

        room.lighting.primary_intensity_pct = 100;
        let (r, g, _b) = pixel(&render(&room), W / 2 - 30, (H as f32 * 0.35) as u32);
        assert!(r > g, "cone red-dominant in AmbientRed: {r} {g}");
    }

    #[test]
    fn tilt_rotates_mattress() {
        let level = render(&base_room());
        let mut tilted = base_room();
        tilted.table.tilt_degrees = 15;
        let tilted = render(&tilted);
        assert!(
            diff_count(&level, &tilted) > 2000,
            "tilt must visibly rotate the mattress"
        );
    }

    #[test]
    fn zoom_scales_tissue_disc() {
        let mut low = base_room();
        low.endoscope.zoom_level = 1;
        let mut high = base_room();
        high.endoscope.zoom_level = 3;

        let screen = Layout::new(W as f32, H as f32).monitor_screen();
        let cx = (screen.x() + screen.width() * 0.5) as u32;
        let cy = (screen.y() + screen.height() * 0.5) as u32;
        // Radius zoom1 ~= 23px, zoom3 ~= 69px: sample the annulus between.
        let sample_x = cx + 45;

        let low_px = pixel(&render(&low), sample_x, cy);
        let high_px = pixel(&render(&high), sample_x, cy);
        assert!(
            high_px.0 > low_px.0 + 40,
            "tissue disc must grow with zoom: {low_px:?} vs {high_px:?}"
        );
    }

    #[test]
    fn irrigation_animates_droplets() {
        let dry = render(&base_room());
        let mut wet = base_room();
        wet.endoscope.irrigation_active = true;
        let wet = render(&wet);
        assert!(diff_count(&dry, &wet) > 50, "droplets must appear");
    }

    #[test]
    fn pressure_bar_shows_warn_color_above_threshold() {
        let layout = Layout::new(W as f32, H as f32);
        let bar = layout.pressure_bar();
        // Sample 38% down from the top of the bar: below the 60% warn
        // marker line, inside the fill when pressure exceeds 15 mmHg.
        let sample_y = (bar.y() + bar.height() * 0.38) as u32;
        let sample_x = (bar.x() + bar.width() * 0.5) as u32;

        let mut ok = base_room();
        ok.insufflator.target_pressure_mmhg = 12;
        let ok_px = pixel(&render(&ok), sample_x, sample_y);

        let mut warn = base_room();
        warn.insufflator.target_pressure_mmhg = 20;
        let warn_px = pixel(&render(&warn), sample_x, sample_y);

        // 12 mmHg: fill ends at 48%, sample point is panel/warn-zone mix.
        assert!(
            ok_px.0 < warn_px.0,
            "warn fill must be brighter in red: {ok_px:?} vs {warn_px:?}"
        );
        assert!(
            warn_px.0 > warn_px.1 + 40,
            "warn fill red-dominant: {warn_px:?}"
        );
    }

    #[test]
    fn flow_bar_is_visible_and_separate_from_pressure() {
        let layout = Layout::new(W as f32, H as f32);
        let pressure = layout.pressure_bar();
        let flow = layout.flow_bar();
        // Bars must not overlap: visual merge was the old bug.
        assert!(
            flow.x() >= pressure.x() + pressure.width(),
            "flow bar must start right of the pressure bar"
        );

        // Low flow: fill stays green, no warn marker overlap.
        let mut room = base_room();
        room.insufflator.gas_flow_l_min = 10;
        let low_px = pixel(
            &render(&room),
            (flow.x() + flow.width() * 0.5) as u32,
            (flow.y() + flow.height() * 0.9) as u32,
        );
        room.insufflator.gas_flow_l_min = 40;
        let high_px = pixel(
            &render(&room),
            (flow.x() + flow.width() * 0.5) as u32,
            (flow.y() + flow.height() * 0.9) as u32,
        );
        assert!(
            high_px.0 > low_px.0,
            "high flow fill must be red-dominant: {low_px:?} vs {high_px:?}"
        );
    }

    #[test]
    fn table_height_scales_pedestal() {
        let layout = Layout::new(W as f32, H as f32);
        let pedestal = layout.pedestal();
        let sample_x = (pedestal.x() + pedestal.width() * 0.5) as u32;
        // The pedestal is bottom-anchored and shrinks from the top, so
        // sample near the top of the full column.
        let sample_y = (pedestal.y() + pedestal.height() * 0.1) as u32;

        let mut low = base_room();
        low.table.height_cm = HEIGHT_MIN;
        let low_px = pixel(&render(&low), sample_x, sample_y);

        let mut high = base_room();
        high.table.height_cm = HEIGHT_MAX;
        let high_px = pixel(&render(&high), sample_x, sample_y);

        // High table: pedestal covers the sample point. Low table: pedestal
        // shrinks, sample falls on background.
        let pedestalish = |p: (u8, u8, u8)| p.0 > 60 && p.1 > 60;
        assert!(
            pedestalish(high_px),
            "high table pedestal must cover sample: {high_px:?}"
        );
        assert!(
            !pedestalish(low_px),
            "low table pedestal must shrink past sample: {low_px:?}"
        );
    }

    #[test]
    fn insufflator_led_reflects_activity() {
        let layout = Layout::new(W as f32, H as f32);
        let panel = layout.insufflator_panel();
        let led_cx = (panel.x() + panel.width() * 0.5) as u32;
        let led_cy = layout.insufflator_header_y() as u32;

        let mut active = base_room();
        active.insufflator.is_active = true;
        let on = pixel(&render(&active), led_cx, led_cy);

        let mut idle = base_room();
        idle.insufflator.is_active = false;
        let off = pixel(&render(&idle), led_cx, led_cy);

        assert!(on.1 > on.0, "active LED green-dominant: {on:?}");
        assert!(off.0 > off.1 + 40, "standby LED amber-dominant: {off:?}");
    }

    #[test]
    fn white_balance_led_reflects_lock_state() {
        let layout = Layout::new(W as f32, H as f32);
        let monitor = layout.monitor();
        let led_r = layout.width * 0.006;
        let led_cx = (monitor.x() + monitor.width() - led_r * 3.0) as u32;
        let led_cy = (monitor.y() + led_r * 3.0) as u32;

        let mut locked = base_room();
        locked.endoscope.white_balance_locked = true;
        let on = pixel(&render(&locked), led_cx, led_cy);

        let mut unlocked = base_room();
        unlocked.endoscope.white_balance_locked = false;
        let off = pixel(&render(&unlocked), led_cx, led_cy);

        assert!(on.1 > on.0 + 40, "locked LED green-dominant: {on:?}");
        assert!(off.0 < 90, "unlocked LED grey/off: {off:?}");
    }

    #[test]
    fn insufflator_gauges_share_baseline_and_height() {
        let layout = Layout::new(W as f32, H as f32);
        let pressure = layout.pressure_bar();
        let flow = layout.flow_bar();
        assert_eq!(pressure.y(), flow.y(), "gauge tops must align");
        assert_eq!(pressure.height(), flow.height(), "gauge heights must match");
        assert_eq!(
            pressure.y() + pressure.height(),
            flow.y() + flow.height(),
            "gauge bottoms must align"
        );
        assert!(
            flow.x() >= pressure.x() + pressure.width(),
            "gauges must not overlap"
        );
    }

    #[test]
    fn mattress_rests_on_pedestal_top() {
        let layout = Layout::new(W as f32, H as f32);
        let pedestal = layout.pedestal();
        let (cx, cy) = layout.table_center();
        let (_, hh) = layout.mattress_half();
        // Default height: pedestal fills its full column, mattress bottom
        // touches the pedestal top.
        let pedestal_top = pedestal.y();
        let mattress_bottom = cy + hh;
        assert!(
            (mattress_bottom - pedestal_top).abs() < 2.0,
            "mattress bottom {mattress_bottom} must touch pedestal top {pedestal_top}"
        );
        // Pedestal is centered under the mattress.
        assert!(
            (pedestal.x() + pedestal.width() * 0.5 - cx).abs() < 1.0,
            "pedestal must sit centered under the mattress"
        );
    }

    #[test]
    fn cone_base_lands_on_mattress_top() {
        let layout = Layout::new(W as f32, H as f32);
        let mattress_top = layout.mattress_top();
        assert!(
            (layout.cone_base_y() - mattress_top).abs() < 2.0,
            "cone base {} must land on mattress top {mattress_top}",
            layout.cone_base_y()
        );
    }

    #[test]
    fn annotation_anchors_stay_inside_scene() {
        let layout = Layout::new(W as f32, H as f32);
        for annotation in Annotation::ALL {
            let (x, y) = layout.annotation_anchor(annotation);
            assert!(
                x >= 0.0 && x <= W as f32 && y >= 0.0 && y <= H as f32,
                "{annotation:?} anchor ({x}, {y}) outside scene"
            );
        }
    }

    #[test]
    fn interlock_draws_red_border() {
        let mut room = base_room();
        room.safety_interlock_active = true;
        let (r, g, b) = pixel(&render(&room), 3, 3);
        assert!(r > g + 80 && r > b + 80, "border red-dominant: {r} {g} {b}");
    }

    #[test]
    fn render_is_deterministic() {
        let room = base_room();
        let a = render(&room);
        let b = render(&room);
        assert_eq!(diff_count(&a, &b), 0);
    }
}
