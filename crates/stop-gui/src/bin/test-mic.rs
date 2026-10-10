//! Standalone microphone STT test binary: prints live transcriptions to
//! the CLI. No Jev, no GUI, no executor.
//!
//! Requires the `mic` feature (default on): this binary is the mic smoke
//! test itself.

#[cfg(feature = "mic")]
use clap::Parser;

#[cfg(feature = "mic")]
use std::time::Duration;

#[cfg(feature = "mic")]
use stop_gui::stt::{SttPipelineConfig, VadConfig};

#[cfg(feature = "mic")]
#[derive(Debug, Parser)]
#[command(
    name = "test-mic",
    about = "Microphone STT smoke test: prints transcriptions"
)]
struct Args {
    /// List input device names and exit.
    #[arg(long)]
    list_devices: bool,

    /// Input device name (see --list-devices); default: system default.
    #[arg(long)]
    device: Option<String>,

    /// VAD speech energy threshold (linear RMS, i16 scale).
    #[arg(long, default_value_t = VadConfig::default().rms_threshold)]
    vad_threshold: i16,

    /// Silence that closes a speech clip, in milliseconds.
    #[arg(long, default_value_t = VadConfig::default().hangover.as_millis() as u64)]
    hangover_ms: u64,
}

#[cfg(feature = "mic")]
fn main() {
    let args = Args::parse();

    if args.list_devices {
        println!("Input devices (first = system default):");
        match stop_gui::stt::list_input_devices() {
            Ok(names) => {
                for name in names {
                    println!("  {name}");
                }
            }
            Err(e) => {
                eprintln!("device listing failed: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    println!("test-mic: speak commands; Ctrl+C exits.");
    match &args.device {
        Some(device) => println!("Device: {device}"),
        None => println!("Device: system default"),
    }
    println!(
        "VAD: threshold {}, hangover {} ms",
        args.vad_threshold, args.hangover_ms
    );
    println!("First run downloads the STT model (~500 MB) to the cache directory.");

    let config = SttPipelineConfig {
        vad: VadConfig {
            rms_threshold: args.vad_threshold,
            hangover: Duration::from_millis(args.hangover_ms),
            ..VadConfig::default()
        },
        device: args.device,
    };

    match stop_gui::stt::spawn_stt_cli_pipeline_with(config) {
        Ok(_handle) => {
            // The CLI printer thread owns output; park until Ctrl+C.
            loop {
                std::thread::park();
            }
        }
        Err(e) => {
            eprintln!("mic pipeline failed: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(not(feature = "mic"))]
fn main() {
    eprintln!("test-mic requires the `mic` cargo feature (build with --features mic)");
    std::process::exit(1);
}
