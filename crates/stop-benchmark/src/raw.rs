//! Raw benchmark output: one JSONL line per whole-case rollout.
//!
//! `run-benchmark` writes a [`RawCase`] once all utterances of a case have
//! been processed against the live System-One provider (never re-queries the
//! model). The evaluation binaries read only this file.
//!
//! Predicted run semantics: the predicted rollout starts at the case
//! `initial_state` and chains on its own predicted room state, completely
//! separate from the dataset's expected states. The input state of an
//! utterance is the previous entry's `Ok` predicted state; when an entry
//! failed (even after retries) the next utterance uses the latest `Ok()`
//! room state, falling back to `initial_state` when no entry succeeded yet.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

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

/// Result of one processed utterance inside a case rollout.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawUtterance {
    /// 0-based position of the utterance within the case history.
    pub entry_index: usize,
    pub raw_utterance: String,
    pub expected_output_state: RoomState,
    /// `Ok(room state after the utterance)`, or `Err(final error)` when all
    /// retry attempts failed. `Err` never feeds the rollout chain.
    pub predicted_output_state: Result<RoomState, String>,
    /// One element per inference attempt, in order (failed attempts
    /// included, with empty `answers`).
    #[serde(default)]
    pub inference_passes: Vec<RawPass>,
    /// Wall-clock latency covering all attempts and backoff of this utterance.
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
                    expected_output_state: RoomState::default(),
                    predicted_output_state: Ok(RoomState::default()),
                    inference_passes: vec![
                        RawPass {
                            latency_ms: 10.0,
                            answers: BTreeMap::new(),
                        },
                        RawPass {
                            latency_ms: 12.0,
                            answers: serde_json::from_str(
                                r#"{"light_brightness":{"choice":"null","confidence":0.9}}"#,
                            )
                            .expect("answers"),
                        },
                    ],
                    wall_latency_ms: 12.0,
                },
                RawUtterance {
                    entry_index: 1,
                    raw_utterance: "what time is it".to_string(),
                    expected_output_state: RoomState::default(),
                    predicted_output_state: Err("provider timeout".to_string()),
                    inference_passes: vec![RawPass::default(); 3],
                    wall_latency_ms: 30.0,
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
            back.entries[1].predicted_output_state,
            Err("provider timeout".to_string())
        );
    }

    #[test]
    fn inference_passes_round_trip_and_default_when_absent() {
        let line = serde_json::to_string(&case()).expect("serialize");
        let back: RawCase = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(back.entries[0].inference_passes.len(), 2);
        assert_eq!(back.entries[1].inference_passes.len(), 3);
        assert!(back.entries[1]
            .inference_passes
            .iter()
            .all(|pass| pass.answers.is_empty()));

        // Legacy lines without the field still parse (serde default).
        let legacy = r#"{"case_id":"c","scenario":"s","model_name":"m","initial_state":{"lighting":{"primary_intensity_pct":80,"field_mode":"Normal"},"endoscope":{"zoom_level":2,"white_balance_locked":true,"irrigation_active":false},"insufflator":{"target_pressure_mmhg":12,"gas_flow_l_min":10,"is_active":true},"table":{"tilt_degrees":0,"height_cm":100},"safety_interlock_active":false},"entries":[]}"#;
        let back: RawCase = serde_json::from_str(legacy).expect("deserialize legacy");
        assert!(back.entries.is_empty());
    }
}
