//! Dataset schema: one full operation scenario per JSONL line.
//!
//! The schema is intentionally minimal: each history entry carries only the
//! raw utterance and the expected room state after the utterance has been
//! fully processed. Entry kinds and per-action decisions stay generator
//! internals; evaluation compares states only.

use serde::{Deserialize, Serialize};
use stop_core::RoomState;

/// One synthetic operation scenario (`data/test_suite.jsonl` line).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetCase {
    pub id: String,
    pub scenario: String,
    /// Model that generated the case (provenance for the ground truth).
    pub model: String,
    pub initial_state: RoomState,
    pub history: Vec<HistoryEntry>,
}

/// One spoken utterance of the scenario plus the room state expected after
/// the utterance has been fully processed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoryEntry {
    pub raw_utterance: String,
    #[serde(alias = "expected_output_state")]
    pub expected_state: RoomState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_round_trips_through_json() {
        let case = DatasetCase {
            id: "case_000".to_string(),
            scenario: "cholecystectomy".to_string(),
            model: "inclusionai/ling-3.0-flash-vl:floor".to_string(),
            initial_state: RoomState::default(),
            history: vec![HistoryEntry {
                raw_utterance: "dim the lights".to_string(),
                expected_state: RoomState::default(),
            }],
        };
        let json = serde_json::to_string(&case).expect("serialize");
        let back: DatasetCase = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(case, back);
    }

    #[test]
    fn entry_carries_exactly_two_fields() {
        let entry = HistoryEntry {
            raw_utterance: "hello".to_string(),
            expected_state: RoomState::default(),
        };
        let value = serde_json::to_value(&entry).expect("to value");
        let object = value.as_object().expect("object");
        assert_eq!(object.len(), 2);
        assert!(object.contains_key("raw_utterance"));
        assert!(object.contains_key("expected_state"));

        // Legacy lines with the old name still parse.
        let legacy = r#"{"raw_utterance":"hi","expected_output_state":{"lighting":{"primary_intensity_pct":80,"field_mode":"Normal"},"endoscope":{"zoom_level":2,"white_balance_locked":true,"irrigation_active":false},"insufflator":{"target_pressure_mmhg":12,"gas_flow_l_min":10,"is_active":true},"table":{"tilt_degrees":0,"height_cm":100},"safety_interlock_active":false}}"#;
        let back: HistoryEntry = serde_json::from_str(legacy).expect("legacy entry");
        assert_eq!(back.expected_state, entry.expected_state);
    }
}
