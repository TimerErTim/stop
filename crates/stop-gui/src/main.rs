//! `stop-gui` entry point: Winit/eframe event loop, input sources, executor.

use clap::Parser;
use stop_core::{RoomState, SinglePassExecutor, systemone::SystemOneClient};

use stop_gui::pipeline::Pipeline;

#[derive(Debug, Parser)]
#[command(name = "stop-gui", about = "Smart-OP System-One controller demo")]
struct Args {
    /// Input source: `stdin` line input, `mic` continuous speech, or `both`.
    #[arg(long, default_value = "stdin")]
    input: String,

    /// List input device names and exit (mic only).
    #[arg(long)]
    list_devices: bool,

    /// Input device name (see --list-devices); default: system default.
    #[arg(long)]
    device: Option<String>,

    /// VAD speech energy threshold (linear RMS, i16 scale). Feature `mic`.
    #[arg(long, default_value_t = 900)]
    vad_threshold: i16,

    /// VAD silence that closes a speech clip, in milliseconds. Feature `mic`.
    #[arg(long, default_value_t = 700)]
    hangover_ms: u64,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    #[cfg(feature = "mic")]
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

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("tokio runtime builds");

    runtime.block_on(run(args));
}

async fn run(args: Args) {
    let client = SystemOneClient::from_env().unwrap_or_else(|e| {
        eprintln!("System-One client not configured: {e}");
        eprintln!("Set SYSTEMONE_API_BASE_URL (e.g. http://localhost:8080) and restart.");
        std::process::exit(1);
    });
    tracing::info!("System-One model: {}", client.model());

    let (pipeline, events_rx) = Pipeline::new();
    let commands_tx = pipeline.commands_tx.clone();

    let executor = SinglePassExecutor::new(client);
    pipeline.spawn_executor(executor);

    let use_mic = args.input == "mic" || args.input == "both";
    let use_stdin = args.input == "stdin" || args.input == "both";

    #[cfg(feature = "mic")]
    if use_mic {
        let config = stop_gui::stt::SttPipelineConfig {
            vad: stop_gui::stt::VadConfig {
                rms_threshold: args.vad_threshold,
                hangover: std::time::Duration::from_millis(args.hangover_ms),
                ..stop_gui::stt::VadConfig::default()
            },
            device: args.device,
        };
        if let Err(e) = stop_gui::stt::spawn_stt_pipeline_with(
            pipeline.events_tx.clone(),
            commands_tx.clone(),
            config,
        ) {
            eprintln!("mic pipeline failed: {e}");
            std::process::exit(1);
        }
    }
    #[cfg(not(feature = "mic"))]
    if use_mic {
        eprintln!("mic input requires the `mic` cargo feature (build with --features mic)");
        std::process::exit(1);
    }

    if use_stdin {
        pipeline.spawn_stdin();
    }

    let room = RoomState::default();
    if let Err(e) = stop_gui::app::run_window(events_rx, commands_tx, room) {
        eprintln!("gui error: {e}");
        std::process::exit(1);
    }
}
