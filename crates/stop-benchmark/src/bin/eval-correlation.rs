//! `eval-correlation`: error-rate correlation with case length (entry count)
//! and entry index from raw benchmark results (no live model calls).
//!
//! An "error entry" is one whose prediction failed (`Err`) or mismatched the
//! expected state; a case error means the case was not exact. Pearson `r` is
//! reported as `null` in JSON when undefined (empty input or no variance).

use std::path::PathBuf;

use clap::Parser;
use stop_benchmark::load_raw_cases;
use stop_benchmark::metrics::CorrelationBreakdown;
use stop_benchmark::report::render_section;

#[derive(Parser, Debug)]
#[command(
    name = "eval-correlation",
    about = "Case-length and entry-index error correlation on raw benchmark output"
)]
struct Args {
    /// Raw benchmark results (JSONL, one case per line).
    #[arg(long, default_value = "data/benchmark_results.jsonl")]
    input: PathBuf,

    /// Write the full correlation breakdown as pretty JSON to this path.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let cases = load_raw_cases(&args.input)?;
    let breakdown = CorrelationBreakdown::compute(&cases);

    let index_rows: Vec<Vec<String>> = breakdown
        .by_entry_index
        .iter()
        .map(|stat| {
            vec![
                stat.entry_index.to_string(),
                stat.entries.to_string(),
                stat.errors.to_string(),
                format!("{:.3}", stat.error_rate()),
            ]
        })
        .collect();
    print!(
        "{}",
        render_section(
            "Error rate by entry index",
            &["Entry index", "Entries", "Errors", "Error rate"],
            &index_rows,
        )
    );
    println!();

    let length_rows: Vec<Vec<String>> = breakdown
        .by_case_length
        .iter()
        .map(|stat| {
            vec![
                stat.case_length.to_string(),
                stat.cases.to_string(),
                stat.case_errors.to_string(),
                format!("{:.3}", stat.case_error_rate()),
                format!("{:.3}", stat.mean_entry_error_rate()),
            ]
        })
        .collect();
    print!(
        "{}",
        render_section(
            "Error rate by case length",
            &[
                "Case length",
                "Cases",
                "Case errors",
                "Case error rate",
                "Mean entry error rate"
            ],
            &length_rows,
        )
    );
    println!();

    println!(
        "Pearson r (entry index vs entry error): {}",
        format_r(breakdown.pearson_entry_index_vs_error)
    );
    println!(
        "Pearson r (case length vs case error): {}",
        format_r(breakdown.pearson_case_length_vs_case_error)
    );

    if let Some(out) = &args.out {
        let json = serde_json::to_string_pretty(&breakdown)?;
        std::fs::write(out, format!("{json}\n"))?;
        println!("Wrote correlation breakdown to {}", out.display());
    }
    Ok(())
}

fn format_r(value: Option<f64>) -> String {
    match value {
        Some(value) => format!("{value:.4}"),
        None => "undefined (no variance)".to_string(),
    }
}