//! `eval-accuracy`: state-match accuracy, action/no-change split, and
//! Sequence Exact Match from raw benchmark results (no live model calls).

use std::path::PathBuf;

use clap::Parser;
use stop_benchmark::load_raw_cases;
use stop_benchmark::metrics::AccuracyReport;
use stop_benchmark::report::render_table;

#[derive(Parser, Debug)]
#[command(
    name = "eval-accuracy",
    about = "State-match accuracy evaluation on raw benchmark output"
)]
struct Args {
    /// Raw benchmark results (JSONL, one case per line).
    #[arg(long, default_value = "data/benchmark_results.jsonl")]
    input: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let cases = load_raw_cases(&args.input)?;
    let report = AccuracyReport::compute(&cases);

    let rows = vec![
        row(
            "Overall accuracy",
            report.accuracy(),
            report.matched_entries,
            report.total_entries,
        ),
        row(
            "Action-entry accuracy",
            report.action_accuracy(),
            report.action_matched,
            report.action_entries,
        ),
        row(
            "No-change accuracy",
            report.no_change_accuracy(),
            report.no_change_matched,
            report.no_change_entries,
        ),
    ];
    print!(
        "{}",
        render_table(&["Metric / Slot", "Accuracy", "Matched", "Total"], &rows)
    );
    println!(
        "Sequence Exact Match (SEM): {:.1}% ({}/{} cases)",
        report.sequence_exact_match() * 100.0,
        report.exact_cases,
        report.total_cases
    );
    println!(
        "No-change false positives (predicted state change on unchanged expected state): {}",
        report.no_change_false_positives()
    );
    Ok(())
}

fn row(metric: &str, accuracy: f64, matched: usize, total: usize) -> Vec<String> {
    vec![
        metric.to_string(),
        format!("{accuracy:.3}"),
        matched.to_string(),
        total.to_string(),
    ]
}
