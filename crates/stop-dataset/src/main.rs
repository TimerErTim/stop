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

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
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

    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let progress = ProgressBar::new(args.count as u64).with_style(progress_style());
    progress.set_message("starting");
    let mut written = 0usize;
    let mut resumed = 0usize;
    let mut failed = 0usize;
    for i in 0..args.count {
        let scenario = scenarios[i % scenarios.len()];
        let id = format!("case_{i:03}");
        if existing.contains(&id) {
            resumed += 1;
            progress.inc(1);
            progress.set_message(format!(
                "written {written}, resumed {resumed}, skipped {failed}"
            ));
            continue;
        }
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
        progress.inc(1);
        progress.set_message(format!(
            "written {written}, resumed {resumed}, skipped {failed}"
        ));
    }
    progress.finish_with_message("done");

    println!(
        "generate-data: wrote {written} cases to {} ({resumed} resumed, {failed} skipped)",
        args.output.display()
    );
    if written == 0 && resumed == 0 {
        std::process::exit(1);
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
