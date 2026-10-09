//! `run-benchmark`: runs every dataset case against the live System-One
//! provider and persists the raw per-utterance results.
//!
//! One pass over the dataset, no re-queries: the evaluation binaries work
//! purely on the raw output. Utterances chain state within a case (each
//! utterance sees the previous predicted state), matching the dataset
//! rollout semantics.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use stop_benchmark::{RawEntry, load_cases};
use stop_core::executor::MultiPassExecutor;
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

    /// Raw output (JSONL, one entry per line), overwritten.
    #[arg(long, default_value = "data/benchmark_results.jsonl")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let cases = load_cases(&args.input)?;
    let client = SystemOneClient::from_env().map_err(|e| format!("system-one: {e}"))?;
    let executor = MultiPassExecutor::new(client);

    let file = File::create(&args.output)?;
    let mut writer = BufWriter::new(file);
    let runtime = tokio::runtime::Runtime::new()?;

    let mut processed = 0usize;
    let mut failed = 0usize;
    for case in &cases {
        let mut state = case.initial_state.clone();
        for (entry_index, entry) in case.history.iter().enumerate() {
            let started = Instant::now();
            let result = runtime.block_on(executor.process_utterance(&state, &entry.raw_utterance));
            let wall_latency_ms = started.elapsed().as_secs_f64() * 1000.0;

            let raw = match result {
                Ok(result) => {
                    state = result.new_room.clone();
                    RawEntry {
                        case_id: case.id.clone(),
                        scenario: case.scenario.clone(),
                        entry_index,
                        raw_utterance: entry.raw_utterance.clone(),
                        initial_state: case.initial_state.clone(),
                        expected_output_state: entry.expected_output_state.clone(),
                        predicted_output_state: Some(result.new_room),
                        wall_latency_ms,
                        pass_latencies_ms: result
                            .passes
                            .iter()
                            .map(|pass| pass.latency.as_secs_f64() * 1000.0)
                            .collect(),
                        error: None,
                    }
                }
                Err(err) => {
                    failed += 1;
                    tracing::warn!(case = %case.id, entry_index, error = %err, "utterance failed");
                    RawEntry {
                        case_id: case.id.clone(),
                        scenario: case.scenario.clone(),
                        entry_index,
                        raw_utterance: entry.raw_utterance.clone(),
                        initial_state: case.initial_state.clone(),
                        expected_output_state: entry.expected_output_state.clone(),
                        predicted_output_state: None,
                        wall_latency_ms,
                        pass_latencies_ms: Vec::new(),
                        error: Some(err.to_string()),
                    }
                }
            };

            let line = serde_json::to_string(&raw)?;
            writeln!(writer, "{line}")?;
            // Persist as we go: a crashed run keeps the completed entries.
            writer.flush()?;
            processed += 1;
        }
    }

    println!(
        "run-benchmark: wrote {processed} entries to {} ({failed} failed)",
        args.output.display()
    );
    Ok(())
}
