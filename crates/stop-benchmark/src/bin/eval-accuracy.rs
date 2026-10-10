//! `eval-accuracy`: state-match accuracy, action/no-change split, and
//! Sequence Exact Match from raw benchmark results (no live model calls).

use std::path::PathBuf;

use clap::Parser;
use stop_benchmark::load_raw_cases;
use stop_benchmark::metrics::AccuracyBreakdown;
use stop_benchmark::report::render_section;

#[derive(Parser, Debug)]
#[command(
    name = "eval-accuracy",
    about = "State-match accuracy evaluation on raw benchmark output"
)]
struct Args {
    /// Raw benchmark results (JSONL, one case per line).
    #[arg(long, default_value = "data/benchmark_results.jsonl")]
    input: PathBuf,

    /// Write the full accuracy breakdown as pretty JSON to this path.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let cases = load_raw_cases(&args.input)?;
    let breakdown = AccuracyBreakdown::compute(&cases);
    let report = &breakdown.total;

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
        render_section(
            "Overall",
            &["Metric / Slot", "Accuracy", "Matched", "Total"],
            &rows
        )
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
    println!();

    print!(
        "{}",
        render_section(
            "Per field",
            &["Field", "Accuracy", "Matched", "Total"],
            &breakdown
                .per_field
                .iter()
                .map(|field| row(&field.field, field.accuracy(), field.matched, field.total))
                .collect::<Vec<_>>(),
        )
    );
    println!();

    print!(
        "{}",
        render_section(
            "Per scenario",
            &["Scenario", "Accuracy", "Matched", "Total", "SEM"],
            &breakdown
                .per_scenario
                .iter()
                .map(|group| group_row(&group.key, group))
                .collect::<Vec<_>>(),
        )
    );
    println!();

    print!(
        "{}",
        render_section(
            "Per model",
            &["Model", "Accuracy", "Matched", "Total", "SEM"],
            &breakdown
                .per_model
                .iter()
                .map(|group| group_row(&group.key, group))
                .collect::<Vec<_>>(),
        )
    );

    if let Some(out) = &args.out {
        let json = serde_json::to_string_pretty(&breakdown)?;
        std::fs::write(out, format!("{json}\n"))?;
        println!("Wrote accuracy breakdown to {}", out.display());
    }
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

fn group_row(key: &str, group: &stop_benchmark::metrics::GroupAccuracy) -> Vec<String> {
    vec![
        key.to_string(),
        format!("{:.3}", group.accuracy()),
        group.matched_entries.to_string(),
        group.total_entries.to_string(),
        format!("{:.1}%", group.sequence_exact_match() * 100.0),
    ]
}
