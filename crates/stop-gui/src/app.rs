//! eframe application: scene texture, HUD, event draining.
//!
//! eframe 0.36 hands the app a root [`egui::Ui`] in `App::ui`; panels are
//! shown inside that root Ui.

use eframe::egui;
use tiny_skia::Pixmap;

use crate::events::{GuiEvent, format_applied};

/// How many HUD status lines stay visible.
const HUD_HISTORY: usize = 8;

/// Repaint cadence for the animation clock (about 30 fps).
const REPAINT_MS: u64 = 33;

pub struct StopApp {
    scene: Pixmap,
    scene_texture: Option<egui::TextureHandle>,
    events_rx: tokio::sync::mpsc::UnboundedReceiver<GuiEvent>,
    commands_tx: tokio::sync::broadcast::Sender<String>,
    room: stop_core::RoomState,
    prompt: Option<String>,
    pending_input: String,
    hud_lines: Vec<String>,
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
            prompt: None,
            pending_input: String::new(),
            hud_lines: Vec::new(),
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
                    self.prompt = Some(text.clone());
                    self.push_hud(format!("> {text}"));
                }
                GuiEvent::SttStatus(status) => self.push_hud(format!("[stt] {status}")),
                GuiEvent::ExecutionFinished {
                    new_room, report, ..
                } => {
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
                    self.push_hud(format!("Failed: {utterance} ({error})"));
                }
            }
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
        egui::Panel::bottom("hud").min_size(150.0).show(ui, |ui| {
            ui.heading("stop — OR control demo");
            if let Some(prompt) = &self.prompt {
                ui.monospace(format!("prompt: {prompt}"));
            }
            ui.separator();
            for line in &self.hud_lines {
                ui.monospace(line.clone());
            }
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Type a command and press Enter:");
                let response = ui.text_edit_singleline(&mut self.pending_input);
                let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                if response.lost_focus() && enter {
                    let text = std::mem::take(&mut self.pending_input).trim().to_string();
                    if !text.is_empty() {
                        let _ = self.commands_tx.send(text);
                    }
                    response.request_focus();
                }
            });
        });

        egui::CentralPanel::no_frame().show(ui, |ui| {
            let available = ui.available_size();
            let Some(texture) = &self.scene_texture else {
                return;
            };
            let image = egui::Image::new((texture.id(), available)).maintain_aspect_ratio(true);
            ui.add(image);
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(REPAINT_MS));
    }
}

/// Launches the eframe window; blocks until the user closes it.
pub fn run_window(
    events_rx: tokio::sync::mpsc::UnboundedReceiver<GuiEvent>,
    commands_tx: tokio::sync::broadcast::Sender<String>,
    room: stop_core::RoomState,
) -> Result<(), String> {
    let app = StopApp::new(events_rx, commands_tx, room);
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_title("stop — Smart-OP System-One Controller"),
        ..Default::default()
    };
    eframe::run_native(
        "stop",
        native,
        Box::new(|_cc| Ok(Box::new(app) as Box<dyn eframe::App>)),
    )
    .map_err(|e| e.to_string())
}
