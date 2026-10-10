//! Offline evaluation tests over a hand-written raw-results fixture:
//! metric goldens, noise false-positive counting, and analyzer binary smoke
//! runs (no live model calls).

use std::path::PathBuf;
use std::process::Command;

use stop_benchmark::load_raw_cases;
use stop_benchmark::metrics::{
    AccuracyBreakdown, AccuracyReport, CorrelationBreakdown, LatencyBreakdown, LatencyReport,
};

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/benchmark_results.jsonl")
}

fn fixture_cases() -> Vec<stop_benchmark::RawCase> {
    load_raw_cases(&fixture_path()).expect("fixture parses")
}

#[test]
fn fixture_parses_one_line_per_case() {
    let cases = fixture_cases();
    assert_eq!(cases.len(), 2);
    assert_eq!(cases[0].case_id, "case_000");
    assert_eq!(cases[0].entries.len(), 2);
    // Entries are aligned with the case history, zero-based and in order.
    assert_eq!(cases[0].entries[0].entry_index, 0);
    assert_eq!(cases[0].entries[1].entry_index, 1);
    assert_eq!(cases[1].case_id, "case_001");
    assert_eq!(
        cases[1].entries[0].predicted_output_state,
        Err("provider timeout".to_string())
    );
}

#[test]
fn accuracy_metrics_match_golden_values() {
    let report = AccuracyReport::compute(&fixture_cases());

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
    let report = LatencyReport::compute(&fixture_cases());

    // Pass samples include the three failed attempts of case_001's entry.
    assert_eq!(report.pass.count, 5);
    assert!((report.pass.mean_ms - 13.8).abs() < 1e-9);
    assert_eq!(report.pass.p50_ms, 10.0);
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
        stdout.contains("Mean latency per pass: 13.8 ms"),
        "{stdout}"
    );
}

fn temp_json_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "stop-eval-test-{}-{name}.json",
        std::process::id()
    ));
    path
}

#[test]
fn accuracy_breakdown_matches_golden_splits() {
    let breakdown = AccuracyBreakdown::compute(&fixture_cases());

    // Per field: 3 entries; brightness differs once (case_000 entry 1
    // predicted 60 vs expected 78), the Err entry mismatches all fields.
    assert_eq!(breakdown.per_field.len(), 11);
    let brightness = breakdown
        .per_field
        .iter()
        .find(|field| field.field == "light_brightness")
        .expect("brightness");
    assert_eq!(brightness.matched, 1);
    assert_eq!(brightness.total, 3);
    let light_mode = breakdown
        .per_field
        .iter()
        .find(|field| field.field == "light_mode")
        .expect("light_mode");
    assert_eq!(light_mode.matched, 2);
    assert_eq!(light_mode.total, 3);
    let tilt = breakdown
        .per_field
        .iter()
        .find(|field| field.field == "table_tilt_degrees")
        .expect("tilt");
    assert_eq!(tilt.matched, 2);
    assert_eq!(tilt.total, 3);

    // Per scenario: cholecystectomy (1/2), hernia_repair (0/1).
    assert_eq!(breakdown.per_scenario.len(), 2);
    assert_eq!(breakdown.per_scenario[0].key, "cholecystectomy");
    assert_eq!(breakdown.per_scenario[0].matched_entries, 1);
    assert_eq!(breakdown.per_scenario[0].total_entries, 2);
    assert_eq!(breakdown.per_scenario[1].key, "hernia_repair");
    assert_eq!(breakdown.per_scenario[1].matched_entries, 0);

    // Per model: single model aggregates everything.
    assert_eq!(breakdown.per_model.len(), 1);
    assert_eq!(breakdown.per_model[0].key, "jev-latest");
    assert_eq!(breakdown.per_model[0].matched_entries, 1);
    assert_eq!(breakdown.per_model[0].total_entries, 3);

    // Per case: failed indices per case.
    assert_eq!(breakdown.per_case.len(), 2);
    assert_eq!(breakdown.per_case[0].failed_entry_indices, vec![1]);
    assert_eq!(breakdown.per_case[1].failed_entry_indices, vec![0]);
}

#[test]
fn eval_accuracy_out_writes_parseable_json() {
    let out = temp_json_path("accuracy");
    let output = Command::new(env!("CARGO_BIN_EXE_eval-accuracy"))
        .arg("--input")
        .arg(fixture_path())
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run eval-accuracy");
    assert!(output.status.success(), "stderr: {:?}", output.stderr);

    let text = std::fs::read_to_string(&out).expect("read out");
    let breakdown: AccuracyBreakdown = serde_json::from_str(&text).expect("parse json");
    assert_eq!(breakdown.total.matched_entries, 1);
    assert_eq!(breakdown.total.total_entries, 3);
    assert_eq!(breakdown.per_scenario.len(), 2);
    assert_eq!(breakdown.per_model.len(), 1);
    assert_eq!(breakdown.per_case.len(), 2);
    std::fs::remove_file(&out).ok();
}

#[test]
fn eval_latency_out_writes_parseable_json() {
    let out = temp_json_path("latency");
    let output = Command::new(env!("CARGO_BIN_EXE_eval-latency"))
        .arg("--input")
        .arg(fixture_path())
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run eval-latency");
    assert!(output.status.success(), "stderr: {:?}", output.stderr);

    let text = std::fs::read_to_string(&out).expect("read out");
    let breakdown: LatencyBreakdown = serde_json::from_str(&text).expect("parse json");
    assert_eq!(breakdown.overall.pass.count, 5);
    assert_eq!(breakdown.per_model.len(), 1);
    assert_eq!(breakdown.per_model[0].model_name, "jev-latest");
    assert_eq!(breakdown.per_model[0].report.pass.count, 5);
    assert_eq!(breakdown.per_case.len(), 2);
    assert_eq!(breakdown.per_case[0].total_passes, 2);
    assert_eq!(breakdown.per_case[1].total_passes, 3);
    std::fs::remove_file(&out).ok();
}

#[test]
fn eval_correlation_binary_reports_from_raw_results() {
    let output = Command::new(env!("CARGO_BIN_EXE_eval-correlation"))
        .arg("--input")
        .arg(fixture_path())
        .output()
        .expect("run eval-correlation");
    assert!(output.status.success(), "stderr: {:?}", output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Error rate by entry index"), "{stdout}");
    assert!(stdout.contains("Error rate by case length"), "{stdout}");
    assert!(
        stdout.contains("Pearson r (entry index vs entry error): 0.5000"),
        "{stdout}"
    );
    assert!(
        stdout.contains("Pearson r (case length vs case error): undefined"),
        "{stdout}"
    );
}

#[test]
fn eval_correlation_out_writes_parseable_json() {
    let out = temp_json_path("correlation");
    let output = Command::new(env!("CARGO_BIN_EXE_eval-correlation"))
        .arg("--input")
        .arg(fixture_path())
        .arg("--out")
        .arg(&out)
        .output()
        .expect("run eval-correlation");
    assert!(output.status.success(), "stderr: {:?}", output.stderr);

    let text = std::fs::read_to_string(&out).expect("read out");
    let breakdown: CorrelationBreakdown = serde_json::from_str(&text).expect("parse json");
    // Entry index 0: two entries, one Err; entry index 1: one mismatch.
    assert_eq!(breakdown.by_entry_index.len(), 2);
    assert_eq!(breakdown.by_entry_index[0].entry_index, 0);
    assert_eq!(breakdown.by_entry_index[0].entries, 2);
    assert_eq!(breakdown.by_entry_index[0].errors, 1);
    assert_eq!(breakdown.by_entry_index[1].entry_index, 1);
    assert_eq!(breakdown.by_entry_index[1].errors, 1);
    // Case lengths: 2 (case_000) and 1 (case_001), both inexact.
    assert_eq!(breakdown.by_case_length.len(), 2);
    assert_eq!(breakdown.by_case_length[0].case_length, 1);
    assert_eq!(breakdown.by_case_length[1].case_length, 2);
    assert_eq!(breakdown.by_case_length[1].case_errors, 1);
    // Entry-index correlation is defined (0.5); case-length correlation is
    // not (both cases errored, no variance).
    assert!((breakdown.pearson_entry_index_vs_error.unwrap() - 0.5).abs() < 1e-9);
    assert!(breakdown.pearson_case_length_vs_case_error.is_none());
    assert_eq!(breakdown.per_case.len(), 2);
    assert!(!breakdown.per_case[0].exact);
    std::fs::remove_file(&out).ok();
}
