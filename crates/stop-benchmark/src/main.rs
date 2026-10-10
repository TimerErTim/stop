//! `run-benchmark` (package main binary): runs every dataset case against
//! the live System-One provider and persists the raw results, one JSONL line
//! per whole-case rollout.
//!
//! Each case is an independent predicted rollout: it starts at the case
//! `initial_state`, chains on its own predicted room states, and retries an
//! utterance up to three times on failure. The evaluation binaries
//! (`eval-accuracy`, `eval-latency`) work purely on this raw output.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use stop_benchmark::load_cases;
use stop_benchmark::run::run_case;
use stop_core::executor::SinglePassExecutor;
use stop_core::systemone::SystemOneClient;

#[derive(Parser, Debug)]
#[command(
    name = "run-benchmark",
    about = "Run all dataset cases against the local System-One / JevK5 instance"
)]
struct Args {
    /// Input dataset (JSONL, one case per line).
    #[arg(long, default_value = "data/test_suite.jsonl")]
    input: PathBuf,

    /// Raw output (JSONL, one case per line), overwritten.
    #[arg(long, default_value = "data/benchmark_results.jsonl")]
    output: PathBuf,
}

fn setup_tracing() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_tracing();
    let args = Args::parse();
    let cases = load_cases(&args.input)?;
    let client = SystemOneClient::from_env().map_err(|e| format!("system-one: {e}"))?;
    let executor = SinglePassExecutor::new(client);

    let file = File::create(&args.output)?;
    let mut writer = BufWriter::new(file);
    let runtime = tokio::runtime::Runtime::new()?;

    let total_entries: usize = cases.iter().map(|case| case.history.len()).sum();
    let progress = ProgressBar::new(total_entries as u64).with_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .expect("valid progress template")
        .progress_chars("=>-"),
    );
    progress.set_message("starting");

    let mut processed = 0usize;
    let mut failed = 0usize;
    for case in &cases {
        let raw = runtime.block_on(run_case(case, |state, utterance| {
            let executor = &executor;
            async move {
                match executor.process_utterance(&state, &utterance).await {
                    Ok(result) => Ok((result.new_room, result.report.latency)),
                    Err(err) => Err(err.to_string()),
                }
            }
        }));

        for entry in &raw.entries {
            if entry.predicted_output_state.is_err() {
                failed += 1;
                tracing::error!(
                    case = %raw.case_id,
                    entry_index = entry.entry_index,
                    error = %entry.predicted_output_state.as_ref().expect_err("checked"),
                    "utterance failed after retries"
                );
            }
        }

        // Persist as we go: a crashed run keeps the completed cases.
        let line = serde_json::to_string(&raw)?;
        writeln!(writer, "{line}")?;
        writer.flush()?;

        processed += raw.entries.len();
        progress.inc(raw.entries.len() as u64);
        progress.set_message(format!("{} ({failed} failed)", case.id));
    }
    progress.finish_with_message("done");

    println!(
        "run-benchmark: wrote {} cases / {processed} entries to {} ({failed} failed)",
        cases.len(),
        args.output.display()
    );
    Ok(())
}
