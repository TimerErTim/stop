//! Raw benchmark output: one JSONL line per processed utterance.
//!
//! `run-benchmark` appends a [`RawEntry`] per utterance as soon as it
//! completes (crash-safe, never re-queries the model). The evaluation
//! binaries read only this file.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use serde::{Deserialize, Serialize};
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

/// Raw result of one processed utterance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawEntry {
    pub case_id: String,
    pub scenario: String,
    /// 0-based position of the utterance within the case history.
    pub entry_index: usize,
    pub raw_utterance: String,
    /// Case initial state (repeated per entry so lines stay self-contained).
    pub initial_state: RoomState,
    pub expected_output_state: RoomState,
    /// `None` when the run failed for this utterance (`error` is set then).
    pub predicted_output_state: Option<RoomState>,
    pub wall_latency_ms: f64,
    /// Latency of each multi-pass loop pass, in execution order.
    pub pass_latencies_ms: Vec<f64>,
    #[serde(default)]
    pub error: Option<String>,
}

/// Parses dataset cases (one [`DatasetCase`] per line).
pub fn load_cases(path: &Path) -> Result<Vec<DatasetCase>, BenchmarkError> {
    parse_jsonl(path)
}

/// Parses raw benchmark results (one [`RawEntry`] per line).
pub fn load_raw_entries(path: &Path) -> Result<Vec<RawEntry>, BenchmarkError> {
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

    fn entry(index: usize) -> RawEntry {
        RawEntry {
            case_id: "case_000".to_string(),
            scenario: "cholecystectomy".to_string(),
            entry_index: index,
            raw_utterance: format!("utterance {index}"),
            initial_state: RoomState::default(),
            expected_output_state: RoomState::default(),
            predicted_output_state: Some(RoomState::default()),
            wall_latency_ms: 12.0,
            pass_latencies_ms: vec![10.0, 12.0],
            error: None,
        }
    }

    #[test]
    fn raw_entry_round_trips_through_json_line() {
        let line = serde_json::to_string(&entry(0)).expect("serialize");
        let back: RawEntry = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(back, entry(0));
    }

    #[test]
    fn error_entries_deserialize_without_predicted_state() {
        let mut broken = entry(1);
        broken.predicted_output_state = None;
        broken.error = Some("provider timeout".to_string());
        let line = serde_json::to_string(&broken).expect("serialize");
        let back: RawEntry = serde_json::from_str(&line).expect("deserialize");
        assert_eq!(back.predicted_output_state, None);
        assert_eq!(back.error.as_deref(), Some("provider timeout"));
    }
}
