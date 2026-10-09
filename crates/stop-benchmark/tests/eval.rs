//! Offline evaluation tests over a hand-written raw-results fixture:
//! metric goldens, noise false-positive counting, and analyzer binary smoke
//! runs (no live model calls).

use std::path::PathBuf;
use std::process::Command;

use stop_benchmark::load_raw_entries;
use stop_benchmark::metrics::{AccuracyReport, LatencyReport};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/benchmark_results.jsonl")
}

fn fixture_entries() -> Vec<stop_benchmark::RawEntry> {
    load_raw_entries(&fixture_path()).expect("fixture parses")
}

#[test]
fn fixture_parses_all_entries() {
    let entries = fixture_entries();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].case_id, "case_000");
    assert_eq!(entries[2].predicted_output_state, None);
}

#[test]
fn accuracy_metrics_match_golden_values() {
    let report = AccuracyReport::compute(&fixture_entries());

    assert_eq!(report.total_entries, 3);
    assert_eq!(report.matched_entries, 1);
    assert_eq!(report.action_entries, 2);
    assert_eq!(report.action_matched, 1);
    assert_eq!(report.no_change_entries, 1);
    assert_eq!(report.no_change_matched, 0);
    assert_eq!(report.no_change_false_positives(), 1);
    assert_eq!(report.total_cases, 2);
    assert_eq!(report.exact_cases, 0);
    assert!((report.accuracy() - 1.0 / 3.0).abs() < 1e-9);
    assert_eq!(report.sequence_exact_match(), 0.0);
}

#[test]
fn latency_metrics_match_golden_values() {
    let report = LatencyReport::compute(&fixture_entries());

    assert_eq!(report.pass.count, 2);
    assert!((report.pass.mean_ms - 19.5).abs() < 1e-9);
    assert_eq!(report.pass.p50_ms, 19.0);
    assert_eq!(report.pass.p95_ms, 20.0);
    assert_eq!(report.pass.max_ms, 20.0);

    assert_eq!(report.utterance.count, 3);
    assert_eq!(report.utterance.p50_ms, 21.0);
    assert_eq!(report.utterance.p95_ms, 30.0);
    assert_eq!(report.utterance.max_ms, 30.0);
}

#[test]
fn eval_accuracy_binary_reports_from_raw_results() {
    let output = Command::new(env!("CARGO_BIN_EXE_eval-accuracy"))
        .arg("--input")
        .arg(fixture_path())
        .output()
        .expect("run eval-accuracy");
    assert!(output.status.success(), "stderr: {:?}", output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Overall accuracy"), "{stdout}");
    assert!(stdout.contains("0.333"), "{stdout}");
    assert!(
        stdout.contains("Sequence Exact Match (SEM): 0.0% (0/2 cases)"),
        "{stdout}"
    );
    assert!(stdout.contains("unchanged expected state): 1"), "{stdout}");
}

#[test]
fn eval_latency_binary_reports_from_raw_results() {
    let output = Command::new(env!("CARGO_BIN_EXE_eval-latency"))
        .arg("--input")
        .arg(fixture_path())
        .output()
        .expect("run eval-latency");
    assert!(output.status.success(), "stderr: {:?}", output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Pass"), "{stdout}");
    assert!(stdout.contains("Utterance"), "{stdout}");
    assert!(
        stdout.contains("Mean latency per pass: 19.5 ms"),
        "{stdout}"
    );
}
