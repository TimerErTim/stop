//! `eval-accuracy`: state-match accuracy, action/no-change split, and
//! Sequence Exact Match from raw benchmark results (no live model calls).
//! Every metric is reported per prediction variant: fresh (ground-truth
//! chaining) vs rolling (self-chaining rollout).

use std::path::PathBuf;

use clap::Parser;
use stop_benchmark::load_raw_cases_multi;
use stop_benchmark::metrics::{AccuracyBreakdown, VariantGroupAccuracy};
use stop_benchmark::report::render_section;

#[derive(Parser, Debug)]
#[command(
    name = "eval-accuracy",
    about = "State-match accuracy evaluation on raw benchmark output (fresh vs rolling)"
)]
struct Args {
    /// Raw benchmark results (JSONL, one case per line). Repeatable; each
    /// value may be a glob pattern.
    #[arg(long, default_value = "data/benchmark*.jsonl")]
    input: Vec<String>,

    /// Write the full accuracy breakdown as pretty JSON to this path.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let cases = load_raw_cases_multi(&args.input)?;
    let breakdown = AccuracyBreakdown::compute(&cases);
    let fresh = &breakdown.total.fresh;
    let rolling = &breakdown.total.rolling;

    let overall_rows = vec![
        split_row(
            "Overall accuracy",
            "fresh",
            fresh.accuracy(),
            &fresh.matched_entries,
            &fresh.total_entries,
        ),
        split_row(
            "Overall accuracy",
            "rolling",
            rolling.accuracy(),
            &rolling.matched_entries,
            &rolling.total_entries,
        ),
        split_row(
            "Action-entry accuracy",
            "fresh",
            fresh.action_accuracy(),
            &fresh.action_matched,
            &fresh.action_entries,
        ),
        split_row(
            "Action-entry accuracy",
            "rolling",
            rolling.action_accuracy(),
            &rolling.action_matched,
            &rolling.action_entries,
        ),
        split_row(
            "No-change accuracy",
            "fresh",
            fresh.no_change_accuracy(),
            &fresh.no_change_matched,
            &fresh.no_change_entries,
        ),
        split_row(
            "No-change accuracy",
            "rolling",
            rolling.no_change_accuracy(),
            &rolling.no_change_matched,
            &rolling.no_change_entries,
        ),
    ];
    print!(
        "{}",
        render_section(
            "Overall",
            &["Metric", "Variant", "Accuracy", "Matched", "Total"],
            &overall_rows,
        )
    );
    println!(
        "Sequence Exact Match (SEM): fresh {:.1}% ({}/{}), rolling {:.1}% ({}/{}) cases",
        fresh.sequence_exact_match() * 100.0,
        fresh.exact_cases,
        fresh.total_cases,
        rolling.sequence_exact_match() * 100.0,
        rolling.exact_cases,
        rolling.total_cases,
    );
    println!(
        "No-change false positives (predicted change on unchanged expected state): fresh {}, rolling {}",
        fresh.no_change_false_positives(),
        rolling.no_change_false_positives(),
    );
    println!();

    let field_rows: Vec<Vec<String>> = breakdown
        .per_field
        .iter()
        .flat_map(|field| {
            vec![
                value_row(
                    &field.field,
                    "fresh",
                    field.fresh.accuracy(),
                    &field.fresh.matched,
                    &field.fresh.total,
                ),
                value_row(
                    &field.field,
                    "rolling",
                    field.rolling.accuracy(),
                    &field.rolling.matched,
                    &field.rolling.total,
                ),
            ]
        })
        .collect();
    print!(
        "{}",
        render_section(
            "Per field",
            &["Field", "Variant", "Accuracy", "Matched", "Total"],
            &field_rows,
        )
    );
    println!();

    print!(
        "{}",
        render_section(
            "Per scenario",
            &["Scenario", "Variant", "Accuracy", "Matched", "Total", "SEM"],
            &breakdown
                .per_scenario
                .iter()
                .flat_map(|group| group_rows(&group.key, group))
                .collect::<Vec<_>>(),
        )
    );
    println!();

    print!(
        "{}",
        render_section(
            "Per model",
            &["Model", "Variant", "Accuracy", "Matched", "Total", "SEM"],
            &breakdown
                .per_model
                .iter()
                .flat_map(|group| group_rows(&group.key, group))
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

fn split_row(
    metric: &str,
    variant: &str,
    accuracy: f64,
    matched: &usize,
    total: &usize,
) -> Vec<String> {
    value_row(metric, variant, accuracy, matched, total)
}

fn value_row(
    label: &str,
    variant: &str,
    accuracy: f64,
    matched: &usize,
    total: &usize,
) -> Vec<String> {
    vec![
        label.to_string(),
        variant.to_string(),
        format!("{accuracy:.3}"),
        matched.to_string(),
        total.to_string(),
    ]
}

fn group_rows(key: &str, group: &VariantGroupAccuracy) -> Vec<Vec<String>> {
    vec![
        value_row(
            key,
            "fresh",
            group.fresh.accuracy(),
            &group.fresh.matched_entries,
            &group.fresh.total_entries,
        )
        .tap_sem(group.fresh.sequence_exact_match()),
        value_row(
            key,
            "rolling",
            group.rolling.accuracy(),
            &group.rolling.matched_entries,
            &group.rolling.total_entries,
        )
        .tap_sem(group.rolling.sequence_exact_match()),
    ]
}

trait TapSem {
    fn tap_sem(self, sem: f64) -> Vec<String>;
}

impl TapSem for Vec<String> {
    fn tap_sem(mut self, sem: f64) -> Vec<String> {
        self.push(format!("{:.1}%", sem * 100.0));
        self
    }
}
