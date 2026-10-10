//! `eval-latency`: latency distributions (P50/P95/P99) from raw benchmark
//! results, computed per inference pass and per utterance wall-clock, for
//! both prediction variants (fresh, rolling). No live model calls.

use std::path::PathBuf;

use clap::Parser;
use stop_benchmark::load_raw_cases_multi;
use stop_benchmark::metrics::{LatencyBreakdown, LatencyBreakdownSides, Stats};
use stop_benchmark::report::render_section;

#[derive(Parser, Debug)]
#[command(
    name = "eval-latency",
    about = "Latency distribution evaluation on raw benchmark output (fresh vs rolling)"
)]
struct Args {
    /// Raw benchmark results (JSONL, one case per line). Repeatable; each
    /// value may be a glob pattern.
    #[arg(long, default_value = "data/benchmark*.jsonl")]
    input: Vec<String>,

    /// Write the full latency breakdown as pretty JSON to this path.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let cases = load_raw_cases_multi(&args.input)?;
    let breakdown = LatencyBreakdown::compute(&cases);

    for model in &breakdown.per_model {
        print!(
            "{}",
            section_for(&format!("Model: {}", model.model_name), &model.report)
        );
    }
    print!("{}", section_for("Overall", &breakdown.overall));

    println!(
        "Mean latency per pass: fresh {:.1} ms, rolling {:.1} ms (local inference)",
        breakdown.overall.fresh.pass.mean_ms, breakdown.overall.rolling.pass.mean_ms
    );

    if let Some(out) = &args.out {
        let json = serde_json::to_string_pretty(&breakdown)?;
        std::fs::write(out, format!("{json}\n"))?;
        println!("Wrote latency breakdown to {}", out.display());
    }
    Ok(())
}

fn section_for(title: &str, report: &LatencyBreakdownSides) -> String {
    let rows = vec![
        stats_row("Pass fresh", &report.fresh.pass),
        stats_row("Utterance fresh", &report.fresh.utterance),
        stats_row("Pass rolling", &report.rolling.pass),
        stats_row("Utterance rolling", &report.rolling.utterance),
    ];
    let section = render_section(
        title,
        &[
            "Unit", "P50 ms", "P95 ms", "P99 ms", "Mean ms", "Max ms", "Samples",
        ],
        &rows,
    );
    format!("{section}\n")
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
