//! GUI event types: executor-side updates pushed to the render thread.

use std::time::Duration;

use stop_core::{AppliedActionReport, RoomState, UtteranceReport};

/// Event flow from the tokio executor task into the eframe repaint loop.
///
/// One utterance produces a fixed sequence:
/// `PromptReceived` -> `ExecutionFinished` (single pass, so deltas ride on
/// the finished event instead of separate `StateDeltaApplied` messages).
#[derive(Debug, Clone)]
pub enum GuiEvent {
    /// Raw utterance as received from any input source.
    PromptReceived(String),
    /// Status line for the mic pipeline (model download, warmup, errors).
    SttStatus(String),
    /// One utterance fully processed: new state plus the pass report.
    ExecutionFinished {
        utterance: String,
        new_room: RoomState,
        report: UtteranceReport,
    },
    /// Inference failed for this utterance.
    ExecutionFailed { utterance: String, error: String },
}

/// Latency humanized for the HUD, e.g. `19ms`.
pub fn format_latency(latency: Duration) -> String {
    let ms = latency.as_secs_f64() * 1000.0;
    if ms < 10.0 {
        format!("{ms:.1}ms")
    } else {
        format!("{ms:.0}ms")
    }
}

/// One `applied` entry formatted HUD-style: `Light -> Dim (-2) [19ms]`.
pub fn format_applied(report: &AppliedActionReport, latency: Duration) -> String {
    format!("{} [{}]", report.detail, format_latency(latency))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn latency_formats() {
        assert_eq!(format_latency(Duration::from_millis(19)), "19ms");
        assert_eq!(format_latency(Duration::from_millis(3)), "3.0ms");
        assert_eq!(format_latency(Duration::from_millis(1234)), "1234ms");
    }

    #[test]
    fn applied_formats_with_latency() {
        let report = AppliedActionReport {
            target_device: stop_core::TargetDevice::SurgicalLight,
            action_kind: stop_core::ActionKind::DecreaseBrightness,
            detail: "Light -> Brightness 60% (requested 60%)".to_string(),
            clamped: false,
        };
        let line = format_applied(&report, Duration::from_millis(19));
        assert_eq!(line, "Light -> Brightness 60% (requested 60%) [19ms]");
    }
}
