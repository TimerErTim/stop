//! Raw benchmark output: one JSONL line per whole-case rollout.
//!
//! `run-benchmark` writes a [`RawCase`] once all utterances of a case have
//! been processed against the live System-One provider (never re-queries the
//! model). The evaluation binaries read only this file.
//!
//! Each utterance is processed in two variants:
//!
//! - `fresh_prediction`: the utterance is applied to the previous
//!   **expected** state (ground truth chaining). Isolates per-utterance
//!   errors from rollout drift.
//! - `rolling_prediction`: the utterance is applied to the previous
//!   **predicted** state (self-chaining rollout). Measures error
//!   accumulation. When a rolling prediction failed, the next one fails
//!   too — there is no recovered state to chain on.
//!
//! The rolling rollout starts at the case `initial_state`; the fresh
//! variant's input for the first entry is the same `initial_state`.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use stop_core::RoomState;
use stop_dataset::DatasetCase;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("malformed JSON on line {line}: {source}")]
    Json {
        line: usize,
        source: serde_json::Error,
    },

    #[error("glob error: {0}")]
    Glob(String),

    #[error("no files matched input: {0}")]
    NoMatches(String),
}

/// Raw result of one whole-case rollout.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawCase {
    pub case_id: String,
    pub scenario: String,
    pub model_name: String,
    /// Case start state; the predicted rollout chains from here.
    pub initial_state: RoomState,
    /// One per dataset history utterance, in order.
    pub entries: Vec<RawUtterance>,
}

/// Result of one processed utterance: fresh and rolling prediction
/// variants over the same utterance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawUtterance {
    /// 0-based position of the utterance within the case history.
    pub entry_index: usize,
    pub raw_utterance: String,
    /// Room state expected after the utterance (dataset ground truth).
    pub expected_state: RoomState,
    /// Utterance applied to the previous expected state.
    pub fresh_prediction: RawPrediction,
    /// Utterance applied to the previous predicted state (rollout).
    pub rolling_prediction: RawPrediction,
}

/// One prediction variant of an utterance: the resulting room state plus
/// every inference attempt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawPrediction {
    /// `Ok(room state after the utterance)`, or `Err(final error)` when all
    /// retry attempts failed.
    pub state: Result<RoomState, String>,
    /// One element per inference attempt, in order (failed attempts
    /// included, with empty `answers`).
    #[serde(default)]
    pub inference_passes: Vec<RawPass>,
    /// Wall-clock latency covering all attempts and backoff of this variant.
    pub wall_latency_ms: f64,
}

/// One inference pass: latency plus the raw model response snapshot
/// (decision answers exactly as returned by the provider; empty when the
/// attempt failed).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct RawPass {
    /// Latency of this single pass in milliseconds.
    pub latency_ms: f64,
    /// Raw decision answers keyed by question name.
    pub answers: BTreeMap<String, Value>,
}

/// Parses dataset cases (one [`DatasetCase`] per line).
pub fn load_cases(path: &Path) -> Result<Vec<DatasetCase>, BenchmarkError> {
    parse_jsonl(path)
}

/// Parses raw benchmark results (one [`RawCase`] per line).
pub fn load_raw_cases(path: &Path) -> Result<Vec<RawCase>, BenchmarkError> {
    parse_jsonl(path)
}

/// Reads the case ids already present in a raw benchmark output file
/// (empty when the file does not exist yet). Malformed lines are an error.
pub fn existing_case_ids(path: &Path) -> Result<BTreeSet<String>, BenchmarkError> {
    if !path.exists() {
        return Ok(BTreeSet::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut ids = BTreeSet::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|source| BenchmarkError::Json {
            line: index + 1,
            source,
        })?;
        if let Some(id) = value.get("case_id").and_then(Value::as_str) {
            ids.insert(id.to_string());
        }
    }
    Ok(ids)
}

/// Expands a list of input patterns into concrete file paths.
///
/// Values containing glob metacharacters (`*?[]{}`) are expanded via the
/// `glob` crate; anything else is treated as a literal path. Paths are
/// sorted and deduplicated; a pattern matching nothing is an error.
pub fn expand_inputs(patterns: &[String]) -> Result<Vec<PathBuf>, BenchmarkError> {
    const METACHARS: &str = "*?[]{}";
    let mut paths = Vec::new();
    for pattern in patterns {
        let mut matched = 0usize;
        if pattern.chars().any(|c| METACHARS.contains(c)) {
            for entry in glob::glob(pattern).map_err(|e| BenchmarkError::Glob(e.to_string()))? {
                let path = entry.map_err(|e| BenchmarkError::Glob(e.to_string()))?;
                if path.is_file() {
                    matched += 1;
                    paths.push(path);
                }
            }
        } else {
            let path = PathBuf::from(pattern);
            if path.is_file() {
                matched += 1;
                paths.push(path);
            }
        }
        if matched == 0 {
            return Err(BenchmarkError::NoMatches(pattern.clone()));
        }
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}

/// Parses raw benchmark results from several files, each value possibly a
/// glob pattern. Cases are deduplicated by `(model_name, case_id)` keeping
/// the first occurrence so overlapping runs are not double counted.
pub fn load_raw_cases_multi(patterns: &[String]) -> Result<Vec<RawCase>, BenchmarkError> {
    let paths = expand_inputs(patterns)?;
    let mut cases = Vec::new();
    let mut seen = BTreeSet::new();
    for path in &paths {
        for case in load_raw_cases(path)? {
            let key = (case.model_name.clone(), case.case_id.clone());
            if seen.insert(key.clone()) {
                cases.push(case);
            } else {
                tracing::warn!(
                    case_id = %case.case_id,
                    model_name = %case.model_name,
                    path = %path.display(),
                    "skipping duplicate case (already loaded)"
                );
            }
        }
    }
    Ok(cases)
}

fn parse_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Result<Vec<T>, BenchmarkError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let item = serde_json::from_str(&line).map_err(|source| BenchmarkError::Json {
            line: index + 1,
            source,
        })?;
        items.push(item);
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn case() -> RawCase {
        RawCase {
            case_id: "case_000".to_string(),
            scenario: "cholecystectomy".to_string(),
            model_name: "jev".to_string(),
            initial_state: RoomState::default(),
            entries: vec![
                RawUtterance {
                    entry_index: 0,
                    raw_utterance: "dim the lights".to_string(),
                    expected_state: RoomState::default(),
                    fresh_prediction: RawPrediction {
                        state: Ok(RoomState::default()),
                        inference_passes: vec![RawPass {
                            latency_ms: 10.0,
                            answers: serde_json::from_str(
                                r#"{"light_brightness":{"choice":"null","confidence":0.9}}"#,
                            )
                            .expect("answers"),
                        }],
                        wall_latency_ms: 11.0,
                    },
                    rolling_prediction: RawPrediction {
                        state: Ok(RoomState::default()),
                        inference_passes: vec![RawPass {
                            latency_ms: 12.0,
                            answers: BTreeMap::new(),
                        }],
                        wall_latency_ms: 12.0,
                    },
                },
                RawUtterance {
                    entry_index: 1,
                    raw_utterance: "what time is it".to_string(),
                    expected_state: RoomState::default(),
                    fresh_prediction: RawPrediction {
                        state: Ok(RoomState::default()),
                        inference_passes: vec![RawPass::default()],
                        wall_latency_ms: 13.0,
                    },
                    rolling_prediction: RawPrediction {
                        state: Err("provider timeout".to_string()),
                        inference_passes: vec![RawPass::default(); 3],
                        wall_latency_ms: 30.0,
                    },
                },
            ],
        }
    }

    #[test]
    fn raw_case_round_trips_through_json_line() {
        let line = serde_json::to_string(&case()).expect("serialize");
        let back: RawCase = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(back, case());
    }

    #[test]
    fn failed_entry_deserializes_as_err_prediction() {
        let line = serde_json::to_string(&case()).expect("serialize");
        let back: RawCase = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(
            back.entries[1].rolling_prediction.state,
            Err("provider timeout".to_string())
        );
        assert!(back.entries[1].fresh_prediction.state.is_ok());
    }

    #[test]
    fn inference_passes_round_trip_and_default_when_absent() {
        let line = serde_json::to_string(&case()).expect("serialize");
        let back: RawCase = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(back.entries[0].fresh_prediction.inference_passes.len(), 1);
        assert_eq!(back.entries[1].rolling_prediction.inference_passes.len(), 3);
        assert!(
            back.entries[1]
                .rolling_prediction
                .inference_passes
                .iter()
                .all(|pass| pass.answers.is_empty())
        );

        // Legacy lines without the field still parse (serde default).
        let legacy = r#"{"case_id":"c","scenario":"s","model_name":"m","initial_state":{"lighting":{"primary_intensity_pct":80,"field_mode":"Normal"},"endoscope":{"zoom_level":2,"white_balance_locked":true,"irrigation_active":false},"insufflator":{"target_pressure_mmhg":12,"gas_flow_l_min":10,"is_active":true},"table":{"tilt_degrees":0,"height_cm":100},"safety_interlock_active":false},"entries":[]}"#;
        let back: RawCase = serde_json::from_str(legacy).expect("deserialize legacy");
        assert!(back.entries.is_empty());
    }
}
