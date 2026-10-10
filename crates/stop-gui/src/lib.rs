//! `stop-gui`: egui/tiny-skia driving adapter for the System-One controller.
//!
//! Windowed demo per `docs/INSTRUCTIONS.md` section 6, input seam per
//! section 7 (hosted here as a driving-adapter concern).

pub mod app;
pub mod events;
pub mod pipeline;
pub mod scene;
pub mod source;

#[cfg(feature = "mic")]
pub mod stt;
