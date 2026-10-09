//! `eval-latency`: latency distributions (P50/P95/P99) from raw benchmark
//! results, computed per pass and per utterance (no live model calls).

use std::path::PathBuf;

use clap::Parser;
use stop_benchmark::load_raw_entries;
use stop_benchmark::metrics::{LatencyReport, Stats};
use stop_benchmark::report::render_table;

#[derive(Parser, Debug)]
#[command(
    name = "eval-latency",
    about = "Latency distribution evaluation on raw benchmark output"
)]
struct Args {
    /// Raw benchmark results (JSONL, one entry per line).
    #[arg(long, default_value = "data/benchmark_results.jsonl")]
    input: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let entries = load_raw_entries(&args.input)?;
    let report = LatencyReport::compute(&entries);

    let rows = vec![
        stats_row("Pass", &report.pass),
        stats_row("Utterance", &report.utterance),
    ];
    print!(
        "{}",
        render_table(
            &[
                "Unit", "P50 ms", "P95 ms", "P99 ms", "Mean ms", "Max ms", "Samples"
            ],
            &rows
        )
    );
    println!(
        "Mean latency per pass: {:.1} ms (local inference)",
        report.pass.mean_ms
    );
    Ok(())
}

fn stats_row(unit: &str, stats: &Stats) -> Vec<String> {
    vec![
        unit.to_string(),
        format!("{:.1}", stats.p50_ms),
        format!("{:.1}", stats.p95_ms),
        format!("{:.1}", stats.p99_ms),
        format!("{:.1}", stats.mean_ms),
        format!("{:.1}", stats.max_ms),
        stats.count.to_string(),
    ]
}
