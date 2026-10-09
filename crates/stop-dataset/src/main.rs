//! `generate-data` (package main binary): writes synthetic ground-truth
//! scenarios to a JSONL file.
//!
//! Usage (see `tasks/misc.toml`):
//! ```bash
//! cargo run -p stop-dataset -- \
//!   --count 250 --output data/test_suite.jsonl \
//!   --scenarios "cholecystectomy,hernia_repair,appendectomy"
//! ```
//! Requires `OPENROUTER_API_KEY`; optional `OPENROUTER_MODEL` and
//! `OPENROUTER_BASE_URL` overrides.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use rand::SeedableRng;
use rand::rngs::StdRng;
use stop_dataset::generator::{Generator, GeneratorConfig};
use stop_dataset::openrouter::OpenRouterClient;

#[derive(Parser, Debug)]
#[command(
    name = "generate-data",
    about = "Generate the synthetic ground-truth dataset (incl. noise transcripts)"
)]
struct Args {
    /// Number of scenario cases (one JSONL line each).
    #[arg(long, default_value_t = 250)]
    count: usize,

    /// Output JSONL path.
    #[arg(long, default_value = "data/test_suite.jsonl")]
    output: PathBuf,

    /// Comma-separated scenario names, round-robined over the cases.
    #[arg(long, default_value = "cholecystectomy,hernia_repair,appendectomy")]
    scenarios: String,

    /// Utterances per case.
    #[arg(long, default_value_t = 16)]
    utterances_per_case: usize,

    /// Share of filler/noise utterances (0.0-1.0), randomly placed;
    /// 0.0 means all utterances are device commands.
    #[arg(long, default_value_t = 0.5)]
    noise_ratio: f64,

    /// RNG seed for the noise placement; random when omitted.
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    let args = Args::parse();
    let mut rng: StdRng = match args.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let client = match OpenRouterClient::from_env() {
        Ok(client) => client,
        Err(err) => {
            eprintln!("generate-data: {err}");
            std::process::exit(1);
        }
    };
    let generator = Generator::new(
        client,
        GeneratorConfig {
            utterances_per_case: args.utterances_per_case,
            noise_ratio: args.noise_ratio,
        },
    );

    let scenarios: Vec<&str> = args
        .scenarios
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if scenarios.is_empty() {
        eprintln!("generate-data: --scenarios must list at least one scenario");
        std::process::exit(1);
    }

    if let Some(dir) = args.output.parent()
        && !dir.as_os_str().is_empty()
        && let Err(err) = std::fs::create_dir_all(dir)
    {
        eprintln!("generate-data: cannot create {}: {err}", dir.display());
        std::process::exit(1);
    }
    let file = match File::create(&args.output) {
        Ok(file) => file,
        Err(err) => {
            eprintln!(
                "generate-data: cannot create {}: {err}",
                args.output.display()
            );
            std::process::exit(1);
        }
    };
    let mut writer = BufWriter::new(file);

    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut written = 0usize;
    let mut failed = 0usize;
    for i in 0..args.count {
        let scenario = scenarios[i % scenarios.len()];
        let id = format!("case_{i:03}");
        let case = runtime.block_on(generator.generate_case(&mut rng, &id, scenario));
        match case {
            Ok(case) => {
                let line = serde_json::to_string(&case).expect("case serializes");
                if let Err(err) = writeln!(writer, "{line}") {
                    eprintln!("generate-data: write failed: {err}");
                    std::process::exit(1);
                }
                written += 1;
                tracing::info!(id = %case.id, scenario = %case.scenario, "case generated");
            }
            Err(err) => {
                failed += 1;
                tracing::warn!(id = %id, error = %err, "case skipped");
            }
        }
        // Persist as we go: a crashed run keeps the completed cases.
        let _ = writer.flush();
    }

    println!(
        "generate-data: wrote {written} cases to {} ({failed} skipped)",
        args.output.display()
    );
    if written == 0 {
        std::process::exit(1);
    }
}
