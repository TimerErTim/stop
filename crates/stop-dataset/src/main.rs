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
//!
//! Output is append-only with resume: case ids already present in the output
//! file are skipped, so an interrupted run continues where it stopped. Cases
//! generate concurrently (`--concurrency`) with per-case retries.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand::rngs::StdRng;
use stop_dataset::error::DatasetError;
use stop_dataset::generator::{
    Generator, GeneratorConfig, UtteranceType, build_utterance_type_sequence,
};
use stop_dataset::openrouter::OpenRouterClient;
use stop_dataset::schema::DatasetCase;
use tokio::sync::{Semaphore, mpsc};
use tokio::time::sleep;

/// Attempts per case before it counts as skipped.
const MAX_ATTEMPTS: usize = 3;
/// Base backoff between attempts; doubles per attempt, capped at 30 s.
const BACKOFF_BASE: Duration = Duration::from_secs(2);

#[derive(Parser, Debug)]
#[command(
    name = "generate-data",
    about = "Generate the synthetic ground-truth dataset (incl. noise transcripts)"
)]
struct Args {
    /// Number of scenario cases (one JSONL line each).
    #[arg(long, default_value_t = 250)]
    count: usize,

    /// Output JSONL path (appended; existing case ids are skipped).
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

    /// Cases generated concurrently (bounded by provider rate limits).
    #[arg(long, default_value_t = 8)]
    concurrency: usize,

    /// RNG seed for the noise placement; random when omitted.
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    let args = Args::parse();
    let base_seed = args.seed.unwrap_or_else(rand::random);

    let client = match OpenRouterClient::from_env() {
        Ok(client) => client,
        Err(err) => {
            eprintln!("generate-data: {err}");
            std::process::exit(1);
        }
    };
    let generator = Arc::new(Generator::new(
        client,
        GeneratorConfig {
            utterances_per_case: args.utterances_per_case,
            noise_ratio: args.noise_ratio,
        },
    ));

    let scenarios: Vec<String> = args
        .scenarios
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
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
    // Append + resume: cases already present in the output are skipped, so an
    // interrupted run continues where it stopped instead of starting over.
    let existing = load_existing_ids(&args.output);
    let file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&args.output)
    {
        Ok(file) => file,
        Err(err) => {
            eprintln!(
                "generate-data: cannot open {}: {err}",
                args.output.display()
            );
            std::process::exit(1);
        }
    };
    let mut writer = BufWriter::new(file);

    let progress = ProgressBar::new(args.count as u64).with_style(progress_style());
    progress.set_message("starting");

    let (tx, mut rx) = mpsc::unbounded_channel::<(String, Result<DatasetCase, DatasetError>)>();
    let semaphore = Arc::new(Semaphore::new(args.concurrency.max(1)));
    let mut written = 0usize;
    let mut resumed = 0usize;
    let mut failed = 0usize;

    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    {
        let _guard = runtime.enter();
        for i in 0..args.count {
            let scenario = scenarios[i % scenarios.len()].clone();
            let id = format!("case_{i:03}");
            if existing.contains(&id) {
                resumed += 1;
                progress.inc(1);
                progress.set_message(format!(
                    "written {written}, resumed {resumed}, skipped {failed}"
                ));
                continue;
            }
            // Deterministic per-case seed: same noise placement regardless of
            // scheduling order across concurrent tasks.
            let mut rng = StdRng::seed_from_u64(base_seed ^ (i as u64).wrapping_mul(0x9E37_79B9));
            let types = build_utterance_type_sequence(generator.config(), &mut rng);

            let generator = Arc::clone(&generator);
            let semaphore = Arc::clone(&semaphore);
            let tx = tx.clone();
            tokio::spawn(async move {
                let _permit = semaphore.acquire_owned().await.expect("semaphore open");
                let result = generate_with_retries(&generator, &types, &id, &scenario).await;
                let _ = tx.send((id, result));
            });
        }
    }
    drop(tx);

    runtime.block_on(async {
        while let Some((id, result)) = rx.recv().await {
            match result {
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
            progress.inc(1);
            progress.set_message(format!(
                "written {written}, resumed {resumed}, skipped {failed}"
            ));
        }
    });
    progress.finish_with_message("done");

    println!(
        "generate-data: wrote {written} cases to {} ({resumed} resumed, {failed} skipped)",
        args.output.display()
    );
    if written == 0 && resumed == 0 {
        std::process::exit(1);
    }
}

/// Generates one case with up to [`MAX_ATTEMPTS`] attempts per provider
/// error (backoff doubling, honoring `Retry-After` hints).
async fn generate_with_retries(
    generator: &Generator,
    types: &[UtteranceType],
    id: &str,
    scenario: &str,
) -> Result<DatasetCase, DatasetError> {
    let mut attempt = 0;
    loop {
        attempt += 1;
        match generator
            .generate_case_with_types(types, id, scenario)
            .await
        {
            Ok(case) => return Ok(case),
            Err(err) if attempt < MAX_ATTEMPTS => {
                let backoff = err
                    .retry_after()
                    .unwrap_or_else(|| BACKOFF_BASE * 2u32.pow(attempt as u32 - 1));
                let backoff = backoff.min(Duration::from_secs(30));
                tracing::warn!(id, attempt, error = %err, retry_in = ?backoff, "retrying case");
                sleep(backoff).await;
            }
            Err(err) => return Err(err),
        }
    }
}

/// Case ids already present in the output file (append + resume support).
fn load_existing_ids(path: &PathBuf) -> HashSet<String> {
    let Ok(content) = std::fs::read_to_string(path) else {
        return HashSet::new();
    };
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter_map(|value| {
            value
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string)
        })
        .collect()
}

fn progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
    )
    .expect("valid progress template")
    .progress_chars("=>-")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_existing_ids_reads_case_ids_and_ignores_garbage() {
        let path =
            std::env::temp_dir().join(format!("stop_dataset_ids_{}.jsonl", std::process::id()));
        std::fs::write(
            &path,
            "{\"id\":\"case_000\",\"scenario\":\"x\"}\nnot json\n\n{\"id\":\"case_007\"}\n",
        )
        .expect("write fixture");

        let ids = load_existing_ids(&path);
        let _ = std::fs::remove_file(&path);

        assert_eq!(ids.len(), 2);
        assert!(ids.contains("case_000"));
        assert!(ids.contains("case_007"));
    }

    #[test]
    fn load_existing_ids_of_missing_file_is_empty() {
        let path = std::env::temp_dir().join("stop_dataset_ids_missing_file.jsonl");
        assert!(load_existing_ids(&path).is_empty());
    }
}
