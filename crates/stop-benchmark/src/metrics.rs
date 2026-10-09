//! Pure metric computation over raw benchmark entries.
//!
//! Scoring model: per-utterance exact match of the predicted final room
//! state against the expected state. Entry classes are derived from the
//! expected states: an entry is an "action" entry when its expected state
//! differs from the previous expected state (case start: initial state),
//! otherwise a "no-change" entry whose only correct prediction is the
//! unchanged state (noise semantics, spec 4.3).

use std::collections::BTreeMap;

use crate::raw::RawEntry;

/// Per-utterance state-match accuracy, split by derived entry class.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AccuracyReport {
    pub total_entries: usize,
    pub matched_entries: usize,
    pub action_entries: usize,
    pub action_matched: usize,
    pub no_change_entries: usize,
    pub no_change_matched: usize,
    pub total_cases: usize,
    /// Cases where every entry matched (Sequence Exact Match).
    pub exact_cases: usize,
}

impl AccuracyReport {
    pub fn compute(entries: &[RawEntry]) -> Self {
        let mut report = Self::default();
        for (_case_id, case_entries) in group_by_case(entries) {
            report.total_cases += 1;
            let mut case_exact = true;
            let mut previous_expected = case_entries
                .first()
                .map(|entry| entry.initial_state.clone());
            for entry in &case_entries {
                let matched =
                    entry.predicted_output_state.as_ref() == Some(&entry.expected_output_state);
                let is_action = previous_expected
                    .as_ref()
                    .is_some_and(|prev| *prev != entry.expected_output_state);
                previous_expected = Some(entry.expected_output_state.clone());

                report.total_entries += 1;
                report.matched_entries += usize::from(matched);
                case_exact &= matched;
                if is_action {
                    report.action_entries += 1;
                    report.action_matched += usize::from(matched);
                } else {
                    report.no_change_entries += 1;
                    report.no_change_matched += usize::from(matched);
                }
            }
            report.exact_cases += usize::from(case_exact);
        }
        report
    }

    /// Matches / total, `0.0` when the class is empty.
    pub fn accuracy(&self) -> f64 {
        ratio(self.matched_entries, self.total_entries)
    }

    pub fn action_accuracy(&self) -> f64 {
        ratio(self.action_matched, self.action_entries)
    }

    pub fn no_change_accuracy(&self) -> f64 {
        ratio(self.no_change_matched, self.no_change_entries)
    }

    /// No-change entries where the model predicted a state change.
    pub fn no_change_false_positives(&self) -> usize {
        self.no_change_entries - self.no_change_matched
    }

    pub fn sequence_exact_match(&self) -> f64 {
        ratio(self.exact_cases, self.total_cases)
    }
}

/// Latency summary statistics in milliseconds.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Stats {
    pub count: usize,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

impl Stats {
    pub fn from_samples(mut samples: Vec<f64>) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("finite latencies"));
        let count = samples.len();
        let mean = samples.iter().sum::<f64>() / count as f64;
        Self {
            count,
            mean_ms: mean,
            p50_ms: percentile(&samples, 50.0),
            p95_ms: percentile(&samples, 95.0),
            p99_ms: percentile(&samples, 99.0),
            max_ms: samples[count - 1],
        }
    }
}

/// Per-pass and per-utterance latency distributions.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LatencyReport {
    pub pass: Stats,
    pub utterance: Stats,
}

impl LatencyReport {
    pub fn compute(entries: &[RawEntry]) -> Self {
        let pass_samples: Vec<f64> = entries
            .iter()
            .flat_map(|entry| entry.pass_latencies_ms.iter().copied())
            .collect();
        let utterance_samples: Vec<f64> =
            entries.iter().map(|entry| entry.wall_latency_ms).collect();
        Self {
            pass: Stats::from_samples(pass_samples),
            utterance: Stats::from_samples(utterance_samples),
        }
    }
}

/// Nearest-rank percentile over a sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let n = sorted.len();
    let rank = ((p / 100.0) * n as f64).ceil() as usize;
    sorted[rank.clamp(1, n) - 1]
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

/// Groups entries by case id; each group is sorted by `entry_index`.
fn group_by_case(entries: &[RawEntry]) -> Vec<(String, Vec<&RawEntry>)> {
    let mut grouped: BTreeMap<String, Vec<&RawEntry>> = BTreeMap::new();
    for entry in entries {
        grouped
            .entry(entry.case_id.clone())
            .or_default()
            .push(entry);
    }
    let mut cases: Vec<(String, Vec<&RawEntry>)> = grouped.into_iter().collect();
    for (_, case_entries) in &mut cases {
        case_entries.sort_by_key(|entry| entry.entry_index);
    }
    cases
}

#[cfg(test)]
mod tests {
    use super::*;
    use stop_core::RoomState;

    fn entry(
        case_id: &str,
        index: usize,
        initial: &RoomState,
        expected: &RoomState,
        predicted: Option<RoomState>,
    ) -> RawEntry {
        RawEntry {
            case_id: case_id.to_string(),
            scenario: "test".to_string(),
            entry_index: index,
            raw_utterance: format!("u{index}"),
            initial_state: initial.clone(),
            expected_output_state: expected.clone(),
            predicted_output_state: predicted,
            wall_latency_ms: 10.0,
            pass_latencies_ms: vec![5.0, 5.0],
            error: None,
        }
    }

    fn changed_light(base: &RoomState, pct: u8) -> RoomState {
        let mut state = base.clone();
        state.lighting.primary_intensity_pct = pct;
        state
    }

    #[test]
    fn classifies_action_and_no_change_entries() {
        let initial = RoomState::default();
        let after = changed_light(&initial, 70);
        let entries = vec![
            // Action entry (state changes), matched.
            entry("c1", 0, &initial, &after, Some(after.clone())),
            // No-change entry (expected == previous), predicted change = FP.
            entry("c1", 1, &initial, &after, Some(changed_light(&after, 50))),
        ];

        let report = AccuracyReport::compute(&entries);

        assert_eq!(report.total_entries, 2);
        assert_eq!(report.action_entries, 1);
        assert_eq!(report.action_matched, 1);
        assert_eq!(report.no_change_entries, 1);
        assert_eq!(report.no_change_matched, 0);
        assert_eq!(report.no_change_false_positives(), 1);
        assert_eq!(report.accuracy(), 0.5);
        assert_eq!(report.sequence_exact_match(), 0.0);
    }

    #[test]
    fn exact_match_counts_only_fully_correct_cases() {
        let initial = RoomState::default();
        let after = changed_light(&initial, 70);
        let entries = vec![
            entry("c1", 0, &initial, &after, Some(after.clone())),
            entry("c1", 1, &initial, &after, Some(after.clone())),
            entry("c2", 0, &initial, &after, Some(initial.clone())),
        ];

        let report = AccuracyReport::compute(&entries);

        assert_eq!(report.total_cases, 2);
        assert_eq!(report.exact_cases, 1);
        assert_eq!(report.sequence_exact_match(), 0.5);
    }

    #[test]
    fn failed_entries_never_match() {
        let initial = RoomState::default();
        let entries = vec![entry("c1", 0, &initial, &initial, None)];
        let report = AccuracyReport::compute(&entries);
        assert_eq!(report.matched_entries, 0);
        assert_eq!(report.no_change_entries, 1);
        assert_eq!(report.no_change_false_positives(), 1);
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(percentile(&samples, 50.0), 5.0);
        assert_eq!(percentile(&samples, 95.0), 10.0);
        assert_eq!(percentile(&samples, 99.0), 10.0);
        assert_eq!(percentile(&samples, 10.0), 1.0);
    }

    #[test]
    fn stats_aggregate_mean_and_max() {
        let stats = Stats::from_samples(vec![10.0, 20.0, 30.0]);
        assert_eq!(stats.count, 3);
        assert_eq!(stats.mean_ms, 20.0);
        assert_eq!(stats.p50_ms, 20.0);
        assert_eq!(stats.max_ms, 30.0);
        assert_eq!(Stats::from_samples(vec![]), Stats::default());
    }

    #[test]
    fn latency_report_collects_pass_and_utterance_samples() {
        let initial = RoomState::default();
        let entries = vec![
            entry("c1", 0, &initial, &initial, Some(initial.clone())),
            entry("c1", 1, &initial, &initial, Some(initial.clone())),
        ];
        let report = LatencyReport::compute(&entries);
        assert_eq!(report.pass.count, 4);
        assert_eq!(report.pass.mean_ms, 5.0);
        assert_eq!(report.utterance.count, 2);
        assert_eq!(report.utterance.mean_ms, 10.0);
    }
}
