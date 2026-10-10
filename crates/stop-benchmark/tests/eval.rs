//! Offline evaluation tests over a hand-written raw-results fixture:
//! metric goldens, noise false-positive counting, and analyzer binary smoke
//! runs (no live model calls).

use std::path::PathBuf;
use std::process::Command;

use stop_benchmark::metrics::{
    AccuracyBreakdown, AccuracyReport, CorrelationBreakdown, LatencyBreakdown, LatencyReport,
};
use stop_benchmark::{load_raw_cases, load_raw_cases_multi};

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
        cases[1].entries[0].rolling_prediction.state,
        Err("provider timeout".to_string())
    );
    assert_eq!(
        cases[1].entries[0].fresh_prediction.state,
        Err("provider timeout".to_string())
    );
}

#[test]
fn accuracy_metrics_match_golden_values() {
    for report in [
        AccuracyReport::compute_fresh(&fixture_cases()),
        AccuracyReport::compute_rolling(&fixture_cases()),
    ] {
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
}

#[test]
fn latency_metrics_match_golden_values() {
    for report in [
        LatencyReport::compute_fresh(&fixture_cases()),
        LatencyReport::compute_rolling(&fixture_cases()),
    ] {
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
    assert!(stdout.contains("fresh"), "{stdout}");
    assert!(stdout.contains("rolling"), "{stdout}");
    assert!(
        stdout.contains("Sequence Exact Match (SEM): fresh 0.0% (0/2), rolling 0.0% (0/2) cases"),
        "{stdout}"
    );
    assert!(
        stdout.contains("unchanged expected state): fresh 1, rolling 1"),
        "{stdout}"
    );
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
    assert!(stdout.contains("Pass fresh"), "{stdout}");
    assert!(stdout.contains("Utterance fresh"), "{stdout}");
    assert!(stdout.contains("Pass rolling"), "{stdout}");
    assert!(stdout.contains("Utterance rolling"), "{stdout}");
    assert!(
        stdout.contains("Mean latency per pass: fresh 13.8 ms, rolling 13.8 ms"),
        "{stdout}"
    );
}

fn temp_json_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("stop-eval-test-{}-{name}.json", std::process::id()));
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
    assert_eq!(brightness.fresh.matched, 1);
    assert_eq!(brightness.fresh.total, 3);
    assert_eq!(brightness.rolling.matched, 1);
    let light_mode = breakdown
        .per_field
        .iter()
        .find(|field| field.field == "light_mode")
        .expect("light_mode");
    assert_eq!(light_mode.fresh.matched, 2);
    assert_eq!(light_mode.fresh.total, 3);
    assert_eq!(light_mode.rolling.matched, 2);
    let tilt = breakdown
        .per_field
        .iter()
        .find(|field| field.field == "table_tilt_degrees")
        .expect("tilt");
    assert_eq!(tilt.fresh.matched, 2);
    assert_eq!(tilt.fresh.total, 3);

    // Per scenario: cholecystectomy (1/2), hernia_repair (0/1).
    assert_eq!(breakdown.per_scenario.len(), 2);
    assert_eq!(breakdown.per_scenario[0].key, "cholecystectomy");
    assert_eq!(breakdown.per_scenario[0].fresh.matched_entries, 1);
    assert_eq!(breakdown.per_scenario[0].fresh.total_entries, 2);
    assert_eq!(breakdown.per_scenario[1].key, "hernia_repair");
    assert_eq!(breakdown.per_scenario[1].fresh.matched_entries, 0);

    // Per model: single model aggregates everything.
    assert_eq!(breakdown.per_model.len(), 1);
    assert_eq!(breakdown.per_model[0].key, "jev-latest");
    assert_eq!(breakdown.per_model[0].fresh.matched_entries, 1);
    assert_eq!(breakdown.per_model[0].fresh.total_entries, 3);

    // Per case: failed indices per case and variant.
    assert_eq!(breakdown.per_case.len(), 2);
    assert_eq!(breakdown.per_case[0].fresh.failed_entry_indices, vec![1]);
    assert_eq!(breakdown.per_case[0].rolling.failed_entry_indices, vec![1]);
    assert_eq!(breakdown.per_case[1].fresh.failed_entry_indices, vec![0]);
    assert_eq!(breakdown.per_case[1].rolling.failed_entry_indices, vec![0]);
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
    assert_eq!(breakdown.total.fresh.matched_entries, 1);
    assert_eq!(breakdown.total.rolling.matched_entries, 1);
    assert_eq!(breakdown.total.fresh.total_entries, 3);
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
    assert_eq!(breakdown.overall.fresh.pass.count, 5);
    assert_eq!(breakdown.overall.rolling.pass.count, 5);
    assert_eq!(breakdown.per_model.len(), 1);
    assert_eq!(breakdown.per_model[0].model_name, "jev-latest");
    assert_eq!(breakdown.per_model[0].report.fresh.pass.count, 5);
    assert_eq!(breakdown.per_case.len(), 2);
    assert_eq!(breakdown.per_case[0].fresh.total_passes, 2);
    assert_eq!(breakdown.per_case[0].rolling.total_passes, 2);
    assert_eq!(breakdown.per_case[1].fresh.total_passes, 3);
    assert_eq!(breakdown.per_case[1].rolling.total_passes, 3);
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
    assert_eq!(breakdown.by_entry_index[0].fresh.entries, 2);
    assert_eq!(breakdown.by_entry_index[0].fresh.errors, 1);
    assert_eq!(breakdown.by_entry_index[0].rolling.errors, 1);
    assert_eq!(breakdown.by_entry_index[1].entry_index, 1);
    assert_eq!(breakdown.by_entry_index[1].fresh.errors, 1);
    // Case lengths: 2 (case_000) and 1 (case_001), both inexact.
    assert_eq!(breakdown.by_case_length.len(), 2);
    assert_eq!(breakdown.by_case_length[0].case_length, 1);
    assert_eq!(breakdown.by_case_length[1].case_length, 2);
    assert_eq!(breakdown.by_case_length[1].fresh.errors, 1);
    // Entry-index correlation is defined (0.5); case-length correlation is
    // not (both cases errored, no variance).
    assert!((breakdown.pearson_entry_index_vs_error.unwrap() - 0.5).abs() < 1e-9);
    assert!(breakdown.pearson_case_length_vs_case_error.is_none());
    assert_eq!(breakdown.per_case.len(), 2);
    assert!(breakdown.per_case[0].fresh.errors > 0);
    std::fs::remove_file(&out).ok();
}

#[test]
fn expand_inputs_expands_globs_and_dedupes() {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let first = fixture_dir.join("benchmark_results.jsonl");
    let second = fixture_dir.join("benchmark_results_extra.jsonl");

    let paths = stop_benchmark::expand_inputs(&[format!("{}/*.jsonl", fixture_dir.display())])
        .expect("glob expands");
    assert_eq!(paths, vec![first.clone(), second.clone()]);

    // Literal path plus overlapping glob: paths dedupe.
    let paths = stop_benchmark::expand_inputs(&[
        first.display().to_string(),
        format!("{}/*.jsonl", fixture_dir.display()),
    ])
    .expect("literal + glob dedupe");
    assert_eq!(paths, vec![first, second]);
}

#[test]
fn expand_inputs_fails_when_nothing_matches() {
    let err = stop_benchmark::expand_inputs(&["data/does_not_exist_*.jsonl".to_string()])
        .expect_err("no matches");
    assert!(err.to_string().contains("no files matched"), "{err}");
}

#[test]
fn multi_loader_merges_and_dedupes_cases() {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let glob = format!("{}/benchmark_results*.jsonl", fixture_dir.display());
    let ids = |cases: &[stop_benchmark::RawCase]| -> Vec<String> {
        cases.iter().map(|case| case.case_id.clone()).collect()
    };

    // Glob matches both fixture files; duplicate case_000 is deduped.
    let cases = load_raw_cases_multi(std::slice::from_ref(&glob)).expect("multi load");
    assert_eq!(
        ids(&cases),
        vec![
            "case_000".to_string(),
            "case_001".to_string(),
            "case_002".to_string()
        ]
    );

    // Same pattern twice: still deduped to the same view.
    let cases = load_raw_cases_multi(&[glob.clone(), glob]).expect("multi load dedupe");
    assert_eq!(
        ids(&cases),
        vec![
            "case_000".to_string(),
            "case_001".to_string(),
            "case_002".to_string()
        ]
    );
}

#[test]
fn multi_loader_merges_two_files_with_duplicate_case() {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let first = fixture_dir.join("benchmark_results.jsonl");

    // Second fixture file repeats case_000 with fresh metrics, then adds
    // case_002; the duplicate must be skipped, the new case appended.
    let second = fixture_dir.join("benchmark_results_extra.jsonl");
    let cases = load_raw_cases_multi(&[first.display().to_string(), second.display().to_string()])
        .expect("multi load");
    let ids: Vec<&str> = cases.iter().map(|case| case.case_id.as_str()).collect();
    assert_eq!(ids, vec!["case_000", "case_001", "case_002"]);
}

#[test]
fn eval_accuracy_binary_merges_repeated_inputs() {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let output = Command::new(env!("CARGO_BIN_EXE_eval-accuracy"))
        .arg("--input")
        .arg(fixture_dir.join("benchmark_results.jsonl"))
        .arg("--input")
        .arg(fixture_dir.join("benchmark_results_extra.jsonl"))
        .output()
        .expect("run eval-accuracy");
    assert!(output.status.success(), "stderr: {:?}", output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    // Merged three-case view: per scenario grows by case_002's scenario.
    assert!(stdout.contains("Overall accuracy"), "{stdout}");
    assert!(stdout.contains("appendectomy"), "{stdout}");
}

#[test]
fn existing_case_ids_reads_case_ids_and_ignores_missing_file() {
    // Missing file: empty set, no error.
    let missing =
        std::env::temp_dir().join(format!("stop-bench-missing-{}.jsonl", std::process::id()));
    let _ = std::fs::remove_file(&missing);
    assert!(
        stop_benchmark::existing_case_ids(&missing)
            .expect("missing file")
            .is_empty()
    );

    // Fixture ids parse.
    let done = stop_benchmark::existing_case_ids(&fixture_path()).expect("fixture ids");
    assert_eq!(
        done,
        ["case_000".to_string(), "case_001".to_string()]
            .into_iter()
            .collect()
    );
}

#[test]
fn run_benchmark_skips_cases_already_in_output() {
    let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let temp = std::env::temp_dir().join(format!("stop-bench-resume-{}.jsonl", std::process::id()));

    // Seed the output with the full fixture: run-benchmark must skip both
    // cases and append nothing (no live model calls happen).
    std::fs::copy(fixture_dir.join("benchmark_results.jsonl"), &temp).expect("seed output");

    let dataset = fixture_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/dataset_resume.jsonl");
    std::fs::write(
        &dataset,
        r#"{"id":"case_000","scenario":"cholecystectomy","model":"jev-latest","initial_state":{"lighting":{"primary_intensity_pct":80,"field_mode":"Normal"},"endoscope":{"zoom_level":2,"white_balance_locked":true,"irrigation_active":false},"insufflator":{"target_pressure_mmhg":12,"gas_flow_l_min":10,"is_active":true},"table":{"tilt_degrees":0,"height_cm":100},"safety_interlock_active":false},"history":[]}
{"id":"case_999","scenario":"appendectomy","model":"jev-latest","initial_state":{"lighting":{"primary_intensity_pct":80,"field_mode":"Normal"},"endoscope":{"zoom_level":2,"white_balance_locked":true,"irrigation_active":false},"insufflator":{"target_pressure_mmhg":12,"gas_flow_l_min":10,"is_active":true},"table":{"tilt_degrees":0,"height_cm":100},"safety_interlock_active":false},"history":[]}
"#,
    );

    // Both cases would need live model calls if not skipped; case_999 has
    // an empty history so it runs without any provider request.
    let output = Command::new(env!("CARGO_BIN_EXE_run-benchmark"))
        .arg("--input")
        .arg(&dataset)
        .arg("--output")
        .arg(&temp)
        // Fixed model skips the endpoint probe; dead base URL would fail
        // it, but no inference call happens for the skipped/empty cases.
        .env("SYSTEMONE_MODEL", "test-model")
        .env("SYSTEMONE_API_BASE_URL", "http://127.0.0.1:1")
        .output()
        .expect("run run-benchmark");
    assert!(output.status.success(), "stderr: {:?}", output.stderr);

    let done = stop_benchmark::existing_case_ids(&temp).expect("ids after run");
    // Seeded case_000 + case_001 kept as-is, case_999 appended.
    assert_eq!(
        done,
        [
            "case_000".to_string(),
            "case_001".to_string(),
            "case_999".to_string()
        ]
        .into_iter()
        .collect(),
        "seeded ids kept, case_999 appended"
    );
    let text = std::fs::read_to_string(&temp).expect("read output");
    assert_eq!(
        text.lines().filter(|line| !line.trim().is_empty()).count(),
        3,
        "no duplicate case lines appended"
    );

    std::fs::remove_file(&temp).ok();
    std::fs::remove_file(&dataset).ok();
}
