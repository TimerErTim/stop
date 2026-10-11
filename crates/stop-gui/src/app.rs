//! eframe application: scene texture, HUD, event draining.
//!
//! eframe 0.36 hands the app a root [`egui::Ui`] in `App::ui`; panels are
//! shown inside that root Ui.

use eframe::egui;
use tiny_skia::Pixmap;

use crate::events::{GuiEvent, MicPhase, format_applied};

/// How many HUD status lines stay visible.
const HUD_HISTORY: usize = 8;

/// Repaint cadence for the animation clock (about 30 fps).
const REPAINT_MS: u64 = 33;

/// Env var set by [`run_window`] when it re-execs itself onto X11, so the
/// retry only happens once (and callers can opt out via `STOP_GUI_FORCE_X11`).
const FORCE_X11_ENV: &str = "STOP_GUI_FORCE_X11";

pub struct StopApp {
    scene: Pixmap,
    scene_texture: Option<egui::TextureHandle>,
    events_rx: tokio::sync::mpsc::UnboundedReceiver<GuiEvent>,
    commands_tx: tokio::sync::broadcast::Sender<String>,
    room: stop_core::RoomState,
    mic_phase: Option<MicPhase>,
    deciding: bool,
    latest_command: String,
    pending_input: String,
    hud_lines: Vec<String>,
    logs_open: bool,
    anim_start: std::time::Instant,
}

impl StopApp {
    pub fn new(
        events_rx: tokio::sync::mpsc::UnboundedReceiver<GuiEvent>,
        commands_tx: tokio::sync::broadcast::Sender<String>,
        room: stop_core::RoomState,
    ) -> Self {
        let scene = Pixmap::new(1280, 800).expect("scene pixmap allocates");
        Self {
            scene,
            scene_texture: None,
            events_rx,
            commands_tx,
            room,
            mic_phase: None,
            deciding: false,
            latest_command: String::new(),
            pending_input: String::new(),
            hud_lines: Vec::new(),
            logs_open: true,
            anim_start: std::time::Instant::now(),
        }
    }

    fn push_hud(&mut self, line: String) {
        self.hud_lines.push(line);
        if self.hud_lines.len() > HUD_HISTORY {
            self.hud_lines.remove(0);
        }
    }

    fn drain_events(&mut self) {
        while let Ok(event) = self.events_rx.try_recv() {
            match event {
                GuiEvent::PromptReceived(text) => {
                    self.deciding = true;
                    self.latest_command = text.clone();
                    self.push_hud(format!("> {text}"));
                }
                GuiEvent::SttStatus(status) => self.push_hud(format!("[stt] {status}")),
                GuiEvent::MicPhase(phase) => self.mic_phase = Some(phase),
                GuiEvent::ExecutionFinished {
                    new_room, report, ..
                } => {
                    self.deciding = false;
                    self.room = new_room;
                    for applied in &report.applied {
                        self.push_hud(format_applied(applied, report.latency));
                    }
                    if report.applied.is_empty() {
                        self.push_hud("No change (noise)".to_string());
                    }
                    if report.requires_sterile_confirm {
                        self.push_hud("Sterile confirm required".to_string());
                    }
                }
                GuiEvent::ExecutionFailed { utterance, error } => {
                    self.deciding = false;
                    self.push_hud(format!("Failed: {utterance} ({error})"));
                }
            }
        }
    }

    /// Corner status text: what the pipeline is doing right now.
    fn status_label(&self) -> &'static str {
        if self.deciding {
            "deciding"
        } else if let Some(phase) = self.mic_phase {
            phase.label()
        } else {
            "ready"
        }
    }

    fn upload_scene(&mut self, ctx: &egui::Context) {
        let width = self.scene.width() as usize;
        let height = self.scene.height() as usize;
        // tiny-skia premultiplied RGBA == straight alpha for an opaque scene.
        let image = egui::ColorImage::from_rgba_unmultiplied([width, height], self.scene.data());
        let texture = ctx.load_texture("or_scene", image, egui::TextureOptions::LINEAR);
        self.scene_texture = Some(texture);
    }
}

impl eframe::App for StopApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();

        self.drain_events();

        // Render the room scene from the current state + animation clock.
        let anim_time = self.anim_start.elapsed().as_secs_f32();
        crate::scene::render_scene(&mut self.scene, &self.room, anim_time);
        self.upload_scene(&ctx);

        // HUD first (bottom panel), scene fills the rest.
        // Layout top to bottom: header, latest command, input, logs.
        egui::Panel::bottom("hud")
            .min_size(if self.logs_open { 110.0 } else { 90.0 })
            .show(ui, |ui| {
                let status = self.status_label();
                ui.horizontal(|ui| {
                    ui.monospace(self.latest_command.clone());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.weak(status);
                    });
                });
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.pending_input)
                        .hint_text("Type a command and press Enter")
                        .desired_width(f32::INFINITY),
                );
                let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                if response.lost_focus() && enter {
                    let text = std::mem::take(&mut self.pending_input).trim().to_string();
                    if !text.is_empty() {
                        let _ = self.commands_tx.send(text);
                    }
                    response.request_focus();
                }
                if self.logs_open {
                    ui.separator();
                    let label = "▾ logs";
                    ui.horizontal(|ui| {
                        if ui.small_button(label).clicked() {
                            self.logs_open = false;
                        }
                    });
                    egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                        for line in &self.hud_lines {
                            ui.monospace(line.clone());
                        }
                    });
                } else {
                    ui.separator();
                    if ui.small_button("▸ logs").clicked() {
                        self.logs_open = true;
                    }
                }
            });

        egui::CentralPanel::no_frame().show(ui, |ui| {
            let available = ui.available_size();
            let Some(texture) = &self.scene_texture else {
                return;
            };
            let image = egui::Image::new((texture.id(), available)).maintain_aspect_ratio(true);
            ui.add(image);

            // Device annotations mapped from scene pixel space onto the
            // displayed image rect (aspect-ratio letterbox aware): the
            // image fits inside `available` keeping 1280x800, centered.
            let scene_w = self.scene.width() as f32;
            let scene_h = self.scene.height() as f32;
            let ratio = (available.x / scene_w).min(available.y / scene_h).min(1.0);
            let draw_size = egui::vec2(scene_w * ratio, scene_h * ratio);
            let top_left = egui::pos2(
                ui.max_rect().left() + (available.x - draw_size.x) * 0.5,
                ui.max_rect().top(),
            );
            let image_rect = egui::Rect::from_min_size(top_left, draw_size);
            let layout = crate::scene::Layout::new(scene_w, scene_h);
            let font = egui::FontId::proportional((image_rect.height() * 0.022).max(11.0));
            let painter = ui.painter();
            for annotation in crate::scene::Annotation::ALL {
                let (ax, ay) = layout.annotation_anchor(annotation);
                let pos = egui::pos2(
                    image_rect.min.x + ax / scene_w * image_rect.width(),
                    image_rect.min.y + ay / scene_h * image_rect.height(),
                );
                painter.text(
                    pos,
                    egui::Align2::CENTER_CENTER,
                    annotation.label(),
                    font.clone(),
                    egui::Color32::from_rgba_unmultiplied(220, 226, 235, 230),
                );
            }
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(REPAINT_MS));
    }
}

/// Launches the eframe window; blocks until the user closes it.
///
/// WSLg quirk: the Wayland event loop dies with `Broken pipe` on the RDP
/// socket right after startup (compositor-side connection reset; winit
/// treats any dispatch error as fatal). winit 0.30 forbids creating a second
/// event loop within one process, so the retry re-execs the same binary with
/// `STOP_GUI_FORCE_X11=1`; on Linux the re-exec replaces this process. The
/// X11 backend is stable under WSLg. Callers can pre-set the env var to skip
/// the doomed Wayland attempt entirely.
pub fn run_window(
    events_rx: tokio::sync::mpsc::UnboundedReceiver<GuiEvent>,
    commands_tx: tokio::sync::broadcast::Sender<String>,
    room: stop_core::RoomState,
) -> Result<(), String> {
    let already_x11 = std::env::var_os(FORCE_X11_ENV).is_some();
    let result = launch(events_rx, commands_tx, room, already_x11);
    match result {
        Ok(()) => Ok(()),
        Err(first) if !already_x11 => {
            tracing::warn!("window run failed ({first}); re-executing on X11");
            reexec_on_x11()
        }
        Err(e) => Err(e.to_string()),
    }
}

/// One window launch attempt; `x11` forces winit onto the X11 backend.
fn launch(
    events_rx: tokio::sync::mpsc::UnboundedReceiver<GuiEvent>,
    commands_tx: tokio::sync::broadcast::Sender<String>,
    room: stop_core::RoomState,
    x11: bool,
) -> Result<(), eframe::Error> {
    let mut options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("stop — Smart-OP System-One Controller"),
        ..Default::default()
    };
    if x11 {
        options.event_loop_builder = Some(Box::new(|builder| {
            #[cfg(target_os = "linux")]
            {
                use winit::platform::x11::EventLoopBuilderExtX11 as _;
                builder.with_x11();
            }
        }));
    }
    eframe::run_native(
        "stop",
        options,
        Box::new(move |_cc| {
            Ok(Box::new(StopApp::new(events_rx, commands_tx, room)) as Box<dyn eframe::App>)
        }),
    )
}

/// Re-execs the current binary with `STOP_GUI_FORCE_X11=1`, replacing this
/// process on success. Unusable without a Linux `execvp`; falls back to an
/// error message on other platforms.
fn reexec_on_x11() -> Result<(), String> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt as _;
        let exe = std::env::current_exe().map_err(|e| format!("cannot locate own binary: {e}"))?;
        let err = std::process::Command::new(exe)
            .args(std::env::args_os().skip(1))
            .env(FORCE_X11_ENV, "1")
            .exec();
        Err(format!("X11 re-exec failed: {err}"))
    }
    #[cfg(not(target_os = "linux"))]
    {
        Err("Wayland window failed and X11 fallback is Linux-only".to_string())
    }
}
