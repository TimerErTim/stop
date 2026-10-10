//! Microphone STT pipeline (feature `mic`): cpal capture, energy VAD,
//! memo-stt transcription on a dedicated worker thread.
//!
//! Layout: the cpal input stream pushes f32 frames into an mpsc channel;
//! the VAD thread converts to i16 mono and slices speech chunks; finished
//! clips go to the STT worker thread owning the blocking
//! [`memo_stt::SttEngine`]; transcriptions are published on the command
//! broadcast for the executor and echoed as HUD status events.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use thiserror::Error;

/// Frame length for VAD analysis (30 ms).
const FRAME_MS: u32 = 30;
/// Silence that closes an open speech chunk.
const DEFAULT_HANGOVER_MS: u32 = 700;
/// Speech energy threshold (linear RMS, f32 scale mapped to i16).
const DEFAULT_RMS_THRESHOLD: i16 = 900;
/// Chunks shorter than this are dropped (memo-stt minimum is 1 s).
const MIN_CLIP_MS: u32 = 1000;

/// Failures of the mic pipeline.
#[derive(Debug, Error)]
pub enum SttError {
    #[error("no audio input device found")]
    NoDevice,

    #[error("audio stream setup failed: {0}")]
    Stream(String),

    #[error("stt engine failed: {0}")]
    Engine(String),
}

/// Tunables for the VAD chunker.
#[derive(Debug, Clone)]
pub struct VadConfig {
    pub rms_threshold: i16,
    pub hangover: Duration,
    pub min_clip: Duration,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            rms_threshold: DEFAULT_RMS_THRESHOLD,
            hangover: Duration::from_millis(u64::from(DEFAULT_HANGOVER_MS)),
            min_clip: Duration::from_millis(u64::from(MIN_CLIP_MS)),
        }
    }
}

/// Spawn flags for the pipeline threads.
#[derive(Debug, Clone, Default)]
pub struct SttPipelineConfig {
    pub vad: VadConfig,
}

/// Slices a continuous sample stream into speech clips: energy-gated with
/// hangover. Feed via [`VadChunker::feed`], drain via
/// [`VadChunker::take_ready`].
#[derive(Debug)]
pub struct VadChunker {
    config: VadConfig,
    sample_rate: u32,
    current: Vec<f32>,
    pending: Vec<f32>,
    ready: Vec<Vec<f32>>,
    silence_run: u32,
}

impl VadChunker {
    pub fn new(sample_rate: u32) -> Self {
        Self::with_config(sample_rate, VadConfig::default())
    }

    pub fn with_config(sample_rate: u32, config: VadConfig) -> Self {
        Self {
            config,
            sample_rate,
            current: Vec::new(),
            pending: Vec::new(),
            ready: Vec::new(),
            silence_run: 0,
        }
    }

    /// RMS energy of a frame on the i16 scale.
    fn frame_rms(frame: &[f32]) -> i16 {
        if frame.is_empty() {
            return 0;
        }
        let sum_sq: f32 = frame.iter().map(|s| s * s).sum();
        let rms = (sum_sq / frame.len() as f32).sqrt();
        (rms * i16::MAX as f32).clamp(0.0, i16::MAX as f32) as i16
    }

    /// Pushes one buffer of mono f32 samples (-1..1) into the chunker.
    pub fn feed(&mut self, samples: &[f32]) {
        let frame_len = self.frame_samples() as usize;
        let mut rest = samples;
        while !rest.is_empty() {
            let take = frame_len.min(rest.len());
            self.pending.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.pending.len() < frame_len {
                break;
            }
            let frame: Vec<f32> = self.pending.drain(..frame_len).collect();
            self.process_frame(&frame);
        }
    }

    fn process_frame(&mut self, frame: &[f32]) {
        let speech = Self::frame_rms(frame) >= self.config.rms_threshold;
        let frame_samples = frame.len() as u32;
        if speech {
            self.current.extend_from_slice(frame);
            self.silence_run = 0;
        } else if !self.current.is_empty() {
            // Hangover: keep the tail, count silence until the clip closes.
            self.current.extend_from_slice(frame);
            self.silence_run += frame_samples;
            if self.silence_run >= self.hangover_samples() {
                self.close_clip();
            }
        }
    }

    fn frame_samples(&self) -> u32 {
        (self.sample_rate * FRAME_MS) / 1000
    }

    fn hangover_samples(&self) -> u32 {
        (self.sample_rate * self.config.hangover.as_millis() as u32) / 1000
    }

    fn min_clip_samples(&self) -> u32 {
        (self.sample_rate * self.config.min_clip.as_millis() as u32) / 1000
    }

    fn close_clip(&mut self) {
        if self.current.len() as u32 >= self.min_clip_samples() {
            self.ready.push(std::mem::take(&mut self.current));
        } else {
            self.current.clear();
        }
        self.silence_run = 0;
    }

    /// Drains completed speech clips (mono f32, -1..1).
    pub fn take_ready(&mut self) -> Vec<Vec<f32>> {
        std::mem::take(&mut self.ready)
    }

    /// True while a clip is being accumulated.
    pub fn is_capturing(&self) -> bool {
        !self.current.is_empty()
    }

    /// Drops any open clip (called on pipeline shutdown).
    pub fn reset(&mut self) {
        self.current.clear();
        self.pending.clear();
        self.silence_run = 0;
    }
}

/// Converts f32 samples (-1..1) to i16 PCM with hard clipping.
pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

/// Surgical vocabulary for the STT initial prompt: better recognition of
/// domain terms.
pub const STT_VOCABULARY_PROMPT: &str = "Operating room command: surgical light brightness, \
endoscope, camera zoom, irrigation, insufflator, CO2 pressure mmHg, operating table, \
Trendelenburg, laparoscopy, dim, emergency stop.";

/// Message produced by the STT worker (transcriptions and status).
#[derive(Debug)]
pub enum SttOutput {
    Transcription(String),
    Status(String),
}

/// Runs the STT worker: builds the engine (downloading the model on first
/// use), warms it up, then transcribes clips until the channel closes.
pub fn stt_worker(
    sample_rate: u32,
    output: tokio::sync::mpsc::UnboundedSender<SttOutput>,
    clips_rx: std::sync::mpsc::Receiver<Vec<f32>>,
    shutdown: Arc<AtomicBool>,
) -> Result<(), SttError> {
    let mut engine = memo_stt::SttEngine::new_default(sample_rate)
        .map_err(|e| SttError::Engine(e.to_string()))?;
    engine.set_prompt(Some(STT_VOCABULARY_PROMPT.to_string()));
    engine
        .warmup()
        .map_err(|e| SttError::Engine(e.to_string()))?;
    let _ = output.send(SttOutput::Status("STT model loaded and warmed up".into()));

    while !shutdown.load(Ordering::Relaxed) {
        // Blocking wait with periodic shutdown checks.
        match clips_rx.recv_timeout(Duration::from_millis(250)) {
            Ok(clip) => {
                let samples = f32_to_i16(&clip);
                match engine.transcribe(&samples) {
                    Ok(text) if !text.trim().is_empty() => {
                        let text = text.trim().to_string();
                        let _ = output.send(SttOutput::Transcription(text));
                    }
                    Ok(_) => {} // empty transcription: no speech detected
                    Err(e) => {
                        let _ = output.send(SttOutput::Status(format!("stt error: {e}")));
                    }
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

/// Opens the best available input device: PulseAudio host when present
/// (WSLg, most desktops), ALSA default otherwise.
fn default_input_config() -> Result<(cpal::Device, cpal::StreamConfig), SttError> {
    let host = pick_host();
    let Some(device) = host.default_input_device() else {
        return Err(SttError::NoDevice);
    };
    if let Ok(config) = default_config_of(&device) {
        return Ok((device, config));
    }
    // WSLg quirk: the default sink exists but its source is unusable; fall
    // back to any listed input device that opens.
    let mut tried = String::new();
    for dev in host
        .input_devices()
        .map_err(|e| SttError::Stream(e.to_string()))?
    {
        let name = dev.to_string();
        if let Ok(config) = default_config_of(&dev) {
            tracing::info!("mic input device: {name} @ {} Hz", config.sample_rate);
            return Ok((dev, config));
        }
        tried.push_str(&format!("\n  {name}"));
    }
    Err(SttError::Stream(format!(
        "no working input device; default failed, alternatives:{tried}"
    )))
}

fn default_config_of(device: &cpal::Device) -> Result<cpal::StreamConfig, cpal::Error> {
    Ok(device.default_input_config()?.into())
}

/// Prefers the PulseAudio host (WSLg exposes no raw ALSA cards), falls
/// back to the platform default host.
fn pick_host() -> cpal::Host {
    for id in cpal::available_hosts() {
        if id == cpal::HostId::PulseAudio
            && let Ok(host) = cpal::host_from_id(id)
        {
            tracing::info!("cpal host: PulseAudio");
            return host;
        }
    }
    tracing::info!("cpal host: default");
    cpal::default_host()
}

/// Starts the mic pipeline: capture stream, VAD thread, STT worker, and a
/// forwarder publishing transcriptions on the command broadcast.
///
/// Returns the shutdown flag; flip it to `true` to stop everything.
pub fn spawn_stt_pipeline(
    events_tx: tokio::sync::mpsc::UnboundedSender<crate::events::GuiEvent>,
    commands_tx: tokio::sync::broadcast::Sender<String>,
) -> Result<Arc<AtomicBool>, SttError> {
    spawn_stt_pipeline_with(events_tx, commands_tx, SttPipelineConfig::default())
}

/// Like [`spawn_stt_pipeline`] with explicit VAD tuning.
pub fn spawn_stt_pipeline_with(
    events_tx: tokio::sync::mpsc::UnboundedSender<crate::events::GuiEvent>,
    commands_tx: tokio::sync::broadcast::Sender<String>,
    pipeline_config: SttPipelineConfig,
) -> Result<Arc<AtomicBool>, SttError> {
    use crate::events::GuiEvent;

    let (device, config) = default_input_config()?;
    let sample_rate = config.sample_rate;
    tracing::info!("mic input: {sample_rate} Hz, {} ch", config.channels);

    // cpal callback -> clip channel feeding the VAD thread.
    let (raw_tx, raw_rx) = std::sync::mpsc::sync_channel::<Vec<f32>>(256);
    let error_flag = Arc::new(AtomicBool::new(false));

    // Channels between threads.
    let (clips_tx, clips_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let (stt_out_tx, mut stt_out_rx) = tokio::sync::mpsc::unbounded_channel::<SttOutput>();
    let shutdown = Arc::new(AtomicBool::new(false));

    // VAD thread: raw f32 frames -> speech clips.
    {
        let clips_tx = clips_tx.clone();
        let error_flag = Arc::clone(&error_flag);
        let channels = config.channels.max(1);
        let vad_config = pipeline_config.vad;
        std::thread::Builder::new()
            .name("stop-vad".into())
            .spawn(move || {
                let mut chunker = VadChunker::with_config(sample_rate, vad_config);
                while let Ok(frame) = raw_rx.recv() {
                    // Down-mix to mono by averaging channel groups.
                    let mono: Vec<f32> = if channels > 1 {
                        frame
                            .chunks(channels as usize)
                            .map(|group| group.iter().sum::<f32>() / channels as f32)
                            .collect()
                    } else {
                        frame
                    };
                    chunker.feed(&mono);
                    for clip in chunker.take_ready() {
                        if clips_tx.send(clip).is_err() {
                            break;
                        }
                    }
                }
                let _ = error_flag;
            })
            .map_err(|e| SttError::Stream(format!("vad thread spawn: {e}")))?;
    }

    // STT worker thread (blocking memo-stt engine).
    {
        let shutdown = Arc::clone(&shutdown);
        std::thread::Builder::new()
            .name("stop-stt".into())
            .spawn(move || {
                if let Err(e) = stt_worker(sample_rate, stt_out_tx.clone(), clips_rx, shutdown) {
                    let _ = stt_out_tx.send(SttOutput::Status(format!("stt init failed: {e}")));
                }
            })
            .map_err(|e| SttError::Stream(format!("stt thread spawn: {e}")))?;
    }

    // Forwarder: STT output -> command broadcast + HUD events.
    {
        let commands_tx = commands_tx.clone();
        let events_tx = events_tx.clone();
        std::thread::Builder::new()
            .name("stop-stt-fwd".into())
            .spawn(move || {
                // Bridge the async mpsc into a blocking loop.
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("forwarder runtime builds");
                runtime.block_on(async move {
                    while let Some(msg) = stt_out_rx.recv().await {
                        match msg {
                            SttOutput::Transcription(text) => {
                                let _ = events_tx.send(GuiEvent::PromptReceived(text.clone()));
                                let _ = commands_tx.send(text);
                            }
                            SttOutput::Status(status) => {
                                let _ = events_tx.send(GuiEvent::SttStatus(status));
                            }
                        }
                    }
                });
            })
            .map_err(|e| SttError::Stream(format!("forwarder thread spawn: {e}")))?;
    }

    // cpal input stream: f32 interleaved frames.
    let stream = device
        .build_input_stream(
            config,
            move |data: &[f32], _| {
                if raw_tx.send(data.to_vec()).is_err() {
                    // VAD thread gone: drop audio.
                }
            },
            move |err| {
                tracing::error!("cpal stream error: {err}");
                error_flag.store(true, Ordering::Relaxed);
            },
            None,
        )
        .map_err(|e| SttError::Stream(e.to_string()))?;
    stream.play().map_err(|e| SttError::Stream(e.to_string()))?;

    // Keep the stream alive for the process lifetime.
    std::mem::forget(stream);
    Ok(shutdown)
}

/// Runtime handle bundle for `test-mic`: it drains raw STT output directly.
pub struct SttPipelineHandle {
    pub shutdown: Arc<AtomicBool>,
    pub output_rx: tokio::sync::mpsc::UnboundedReceiver<SttOutput>,
}

/// CLI variant used by `test-mic`: builds the pipeline and prints every
/// transcription and status line to stdout as it arrives.
pub fn spawn_stt_cli_pipeline_with(
    pipeline_config: SttPipelineConfig,
) -> Result<SttPipelineHandle, SttError> {
    let (events_tx, mut events_rx) = tokio::sync::mpsc::unbounded_channel();
    let (commands_tx, _commands_rx) = tokio::sync::broadcast::channel(64);
    let shutdown = spawn_stt_pipeline_with(events_tx, commands_tx, pipeline_config)?;
    // CLI presentation: drain the GUI event channel and print.
    std::thread::Builder::new()
        .name("stop-stt-cli".into())
        .spawn(move || {
            while let Some(event) = events_rx.blocking_recv() {
                match event {
                    crate::events::GuiEvent::PromptReceived(text) => {
                        println!("[stt] {text}");
                    }
                    crate::events::GuiEvent::SttStatus(status) => {
                        println!("[stt:status] {status}");
                    }
                    _ => {}
                }
            }
        })
        .map_err(|e| SttError::Stream(format!("cli printer thread: {e}")))?;
    Ok(SttPipelineHandle {
        shutdown,
        // The GUI event channel is the presentation path; no separate raw
        // receiver is exposed. The handle exists for symmetry and shutdown.
        output_rx: tokio::sync::mpsc::unbounded_channel().1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16_000;

    /// Sine burst generator (0.5 s at 440 Hz, amplitude 0.5).
    fn sine_burst(duration_ms: u32, rate: u32, amplitude: f32) -> Vec<f32> {
        let n = (rate * duration_ms / 1000) as usize;
        (0..n)
            .map(|i| {
                amplitude * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / rate as f32).sin()
            })
            .collect()
    }

    fn silence(duration_ms: u32, rate: u32) -> Vec<f32> {
        vec![0.0; (rate * duration_ms / 1000) as usize]
    }

    #[test]
    fn rms_threshold_classifies_energy() {
        assert_eq!(VadChunker::frame_rms(&silence(30, RATE)), 0);
        let burst = sine_burst(30, RATE, 0.5);
        assert!(VadChunker::frame_rms(&burst) > DEFAULT_RMS_THRESHOLD);
        // Quiet speech (amplitude 0.02) stays below the default threshold.
        let quiet = sine_burst(30, RATE, 0.02);
        assert!(VadChunker::frame_rms(&quiet) < DEFAULT_RMS_THRESHOLD);
    }

    #[test]
    fn f32_conversion_clips_and_scales() {
        let out = f32_to_i16(&[0.0, 0.5, 2.0, -2.0]);
        // -1.0 maps to i16::MIN + 1 via the `as` cast truncation; -2.0 clamps
        // to the same extreme. Only the relative extremes matter here.
        assert_eq!(out[0], 0);
        assert_eq!(out[1], 16383);
        assert_eq!(out[2], i16::MAX);
        assert_eq!(out[3], i16::MIN + 1);
        assert!(out[3] < -32000, "negative extreme clamps hard");
    }

    #[test]
    fn chunker_splits_bursts_and_enforces_min_length() {
        let mut chunker = VadChunker::new(RATE);
        // Two 1.2 s bursts separated by 2 s silence (hangover 0.7 s closes
        // the first clip inside the gap).
        chunker.feed(&sine_burst(1200, RATE, 0.5));
        chunker.feed(&silence(2000, RATE));
        chunker.feed(&sine_burst(1200, RATE, 0.5));
        chunker.feed(&silence(2000, RATE));

        let clips = chunker.take_ready();
        assert_eq!(clips.len(), 2, "two bursts yield two clips");
        for clip in &clips {
            let ms = clip.len() as u64 * 1000 / RATE as u64;
            assert!(
                ms >= u64::from(MIN_CLIP_MS),
                "clip length {ms} ms below minimum"
            );
        }
    }

    #[test]
    fn chunker_drops_short_noise_bursts() {
        let mut chunker = VadChunker::new(RATE);
        chunker.feed(&sine_burst(200, RATE, 0.5));
        chunker.feed(&silence(2000, RATE));
        assert!(chunker.take_ready().is_empty(), "200 ms clip is dropped");
    }

    #[test]
    fn chunker_survives_partial_frames() {
        let mut chunker = VadChunker::new(RATE);
        // Feed in irregular sizes; internal framing must absorb remainders.
        chunker.feed(&sine_burst(997, RATE, 0.5));
        chunker.feed(&silence(2001, RATE));
        assert_eq!(chunker.take_ready().len(), 1);
    }

    #[test]
    fn silence_converts_to_zero_samples() {
        let clip = silence(1200, RATE);
        let samples = f32_to_i16(&clip);
        assert!(samples.iter().all(|s| *s == 0));
    }
}
