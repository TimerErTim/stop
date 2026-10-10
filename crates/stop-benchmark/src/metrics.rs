//! Pure metric computation over raw benchmark cases.
//!
//! Scoring model: per-utterance exact match of the predicted room state
//! against the expected state. Entry classes are derived from the expected
//! states: an entry is an "action" entry when its expected state differs from
//! the previous expected state (case start: initial state), otherwise a
//! "no-change" entry whose only correct prediction is the unchanged state
//! (noise semantics, spec 4.3).

use serde::{Deserialize, Serialize};

use crate::raw::RawCase;

/// Per-utterance state-match accuracy, split by derived entry class.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
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
    pub fn compute(cases: &[RawCase]) -> Self {
        let mut report = Self::default();
        for case in cases {
            report.total_cases += 1;
            let mut case_exact = true;
            let mut previous_expected = Some(case.initial_state.clone());
            for entry in &case.entries {
                let matched =
                    entry.predicted_output_state == Ok(entry.expected_output_state.clone());
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

/// Accuracy of one `RoomState` field (value match between expected and
/// predicted state; a failed prediction mismatches every field).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldAccuracy {
    pub field: String,
    pub matched: usize,
    pub total: usize,
}

impl FieldAccuracy {
    /// Matches / total, `0.0` when the field has no entries.
    pub fn accuracy(&self) -> f64 {
        ratio(self.matched, self.total)
    }
}

/// Accuracy grouped by one key (scenario, model).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupAccuracy {
    pub key: String,
    pub total_entries: usize,
    pub matched_entries: usize,
    pub total_cases: usize,
    /// Cases where every entry matched.
    pub exact_cases: usize,
}

impl GroupAccuracy {
    /// Matches / total, `0.0` when the group is empty.
    pub fn accuracy(&self) -> f64 {
        ratio(self.matched_entries, self.total_entries)
    }

    /// Cases where every entry matched / total cases, `0.0` when empty.
    pub fn sequence_exact_match(&self) -> f64 {
        ratio(self.exact_cases, self.total_cases)
    }
}

/// One case's own score plus the entry indices that failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseAccuracy {
    pub case_id: String,
    pub scenario: String,
    pub model_name: String,
    pub total_entries: usize,
    pub matched_entries: usize,
    /// Every entry matched.
    pub exact: bool,
    /// Entry indices whose prediction failed (`Err`) or mismatched.
    pub failed_entry_indices: Vec<usize>,
}

/// Full accuracy evaluation output: totals plus every split, JSON-serializable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccuracyBreakdown {
    /// Whole-input totals (entry classes, sequence exact match).
    pub total: AccuracyReport,
    pub per_field: Vec<FieldAccuracy>,
    pub per_scenario: Vec<GroupAccuracy>,
    pub per_model: Vec<GroupAccuracy>,
    pub per_case: Vec<CaseAccuracy>,
}

/// Field names in stable `RoomState` order (see `stop_core::RoomState`).
pub const FIELDS: [&str; 11] = [
    "light_brightness",
    "light_mode",
    "zoom_level",
    "white_balance_locked",
    "irrigation_active",
    "target_pressure_mmhg",
    "gas_flow_l_min",
    "insufflator_active",
    "table_tilt_degrees",
    "table_height_cm",
    "safety_interlock_active",
];

/// Latency summary statistics in milliseconds.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LatencyReport {
    pub pass: Stats,
    pub utterance: Stats,
}

impl LatencyReport {
    pub fn compute(cases: &[RawCase]) -> Self {
        Self::compute_owned(cases.iter())
    }

    /// Same as [`LatencyReport::compute`] over an arbitrary case iterator
    /// (used for per-model groups without copying the raw cases).
    pub fn compute_owned<'a>(cases: impl IntoIterator<Item = &'a RawCase>) -> Self {
        let mut pass_samples = Vec::new();
        let mut utterance_samples = Vec::new();
        for case in cases {
            for entry in &case.entries {
                pass_samples.extend(entry.inference_passes.iter().map(|pass| pass.latency_ms));
                utterance_samples.push(entry.wall_latency_ms);
            }
        }
        Self {
            pass: Stats::from_samples(pass_samples),
            utterance: Stats::from_samples(utterance_samples),
        }
    }
}

/// Per-case latency summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseLatency {
    pub case_id: String,
    pub model_name: String,
    pub entries: usize,
    /// Mean wall-clock latency across this case's utterances, milliseconds.
    pub mean_wall_ms: f64,
    /// Total inference passes (failed attempts included).
    pub total_passes: usize,
}

/// Latency evaluation output: overall plus per-model and per-case splits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    pub overall: LatencyReport,
    pub per_model: Vec<LatencyByModel>,
    pub per_case: Vec<CaseLatency>,
}

/// One model's latency distribution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyByModel {
    pub model_name: String,
    pub report: LatencyReport,
}

/// Error-rate statistics for one `entry_index` position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryIndexStat {
    pub entry_index: usize,
    pub entries: usize,
    pub errors: usize,
}

impl EntryIndexStat {
    /// Errors / entries, `0.0` when empty.
    pub fn error_rate(&self) -> f64 {
        ratio(self.errors, self.entries)
    }
}

/// Error-rate statistics for one case-length bucket (case length = number of
/// history entries).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseLengthStat {
    pub case_length: usize,
    pub cases: usize,
    pub case_errors: usize,
    /// Sum of per-case entry error counts (for the mean below).
    pub total_entry_errors: usize,
    pub total_entries: usize,
}

impl CaseLengthStat {
    /// Cases that were not exact / cases, `0.0` when empty.
    pub fn case_error_rate(&self) -> f64 {
        ratio(self.case_errors, self.cases)
    }

    /// Entry errors / entries within this bucket, `0.0` when empty.
    pub fn mean_entry_error_rate(&self) -> f64 {
        ratio(self.total_entry_errors, self.total_entries)
    }
}

/// One case's length, error count, and exactness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseCorrelation {
    pub case_id: String,
    pub scenario: String,
    pub model_name: String,
    pub case_length: usize,
    pub error_count: usize,
    /// Every entry matched.
    pub exact: bool,
}

/// Correlation evaluation output: error rate by entry index and by case
/// length, plus Pearson correlations (null when undefined, i.e. no variance
/// or empty input).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrelationBreakdown {
    pub by_entry_index: Vec<EntryIndexStat>,
    pub by_case_length: Vec<CaseLengthStat>,
    /// Pearson r over (entry index, entry error 0/1) for all entries.
    pub pearson_entry_index_vs_error: Option<f64>,
    /// Pearson r over (case length, case error 0/1) for all cases.
    pub pearson_case_length_vs_case_error: Option<f64>,
    pub per_case: Vec<CaseCorrelation>,
}

impl AccuracyBreakdown {
    pub fn compute(cases: &[RawCase]) -> Self {
        let total = AccuracyReport::compute(cases);
        let per_field = field_accuracy(cases);
        let per_scenario = group_accuracy(cases, |case| case.scenario.clone());
        let per_model = group_accuracy(cases, |case| case.model_name.clone());
        let per_case = cases.iter().map(case_accuracy).collect();
        Self {
            total,
            per_field,
            per_scenario,
            per_model,
            per_case,
        }
    }
}

/// Per-field accuracy: each field counts once per entry.
fn field_accuracy(cases: &[RawCase]) -> Vec<FieldAccuracy> {
    let mut reports: Vec<FieldAccuracy> = FIELDS
        .iter()
        .map(|field| FieldAccuracy {
            field: (*field).to_string(),
            matched: 0,
            total: 0,
        })
        .collect();
    for case in cases {
        for entry in &case.entries {
            let predicted = entry.predicted_output_state.as_ref().ok();
            for (report, matched) in reports.iter_mut().zip(field_matches(&entry.expected_output_state, predicted)) {
                report.total += 1;
                report.matched += usize::from(matched);
            }
        }
    }
    reports
}

/// Per-field match of one entry: each field is `true` when the predicted
/// value equals the expected value. A failed prediction (`None`) mismatches
/// every field.
fn field_matches(expected: &stop_core::RoomState, predicted: Option<&stop_core::RoomState>) -> [bool; 11] {
    let Some(predicted) = predicted else {
        return [false; 11];
    };
    [
        expected.lighting.primary_intensity_pct == predicted.lighting.primary_intensity_pct,
        expected.lighting.field_mode == predicted.lighting.field_mode,
        expected.endoscope.zoom_level == predicted.endoscope.zoom_level,
        expected.endoscope.white_balance_locked == predicted.endoscope.white_balance_locked,
        expected.endoscope.irrigation_active == predicted.endoscope.irrigation_active,
        expected.insufflator.target_pressure_mmhg == predicted.insufflator.target_pressure_mmhg,
        expected.insufflator.gas_flow_l_min == predicted.insufflator.gas_flow_l_min,
        expected.insufflator.is_active == predicted.insufflator.is_active,
        expected.table.tilt_degrees == predicted.table.tilt_degrees,
        expected.table.height_cm == predicted.table.height_cm,
        expected.safety_interlock_active == predicted.safety_interlock_active,
    ]
}

/// Groups entries and cases by `key`, preserving first-seen order.
fn group_accuracy<F>(cases: &[RawCase], key: F) -> Vec<GroupAccuracy>
where
    F: Fn(&RawCase) -> String,
{
    let mut groups: Vec<GroupAccuracy> = Vec::new();
    for case in cases {
        let key = key(case);
        let case_exact = case.entries.iter().all(|entry| {
            entry.predicted_output_state == Ok(entry.expected_output_state.clone())
        });
        let matched = case
            .entries
            .iter()
            .filter(|entry| entry.predicted_output_state == Ok(entry.expected_output_state.clone()))
            .count();
        match groups.iter_mut().find(|group| group.key == key) {
            Some(group) => {
                group.total_cases += 1;
                group.exact_cases += usize::from(case_exact);
                group.total_entries += case.entries.len();
                group.matched_entries += matched;
            }
            None => {
                groups.push(GroupAccuracy {
                    key,
                    total_entries: case.entries.len(),
                    matched_entries: matched,
                    total_cases: 1,
                    exact_cases: usize::from(case_exact),
                });
            }
        };
    }
    groups
}

fn case_accuracy(case: &RawCase) -> CaseAccuracy {
    let mut failed_entry_indices = Vec::new();
    for entry in &case.entries {
        if entry.predicted_output_state != Ok(entry.expected_output_state.clone()) {
            failed_entry_indices.push(entry.entry_index);
        }
    }
    CaseAccuracy {
        case_id: case.case_id.clone(),
        scenario: case.scenario.clone(),
        model_name: case.model_name.clone(),
        total_entries: case.entries.len(),
        matched_entries: case.entries.len() - failed_entry_indices.len(),
        exact: failed_entry_indices.is_empty(),
        failed_entry_indices,
    }
}

impl LatencyBreakdown {
    pub fn compute(cases: &[RawCase]) -> Self {
        let overall = LatencyReport::compute(cases);

        // Group cases per model first, then compute each model's report in
        // one pass so percentiles cover all of that model's samples.
        let mut model_names: Vec<&str> = Vec::new();
        for case in cases {
            if !model_names.contains(&case.model_name.as_str()) {
                model_names.push(&case.model_name);
            }
        }
        let per_model = model_names
            .into_iter()
            .map(|model_name| {
                let own: Vec<&RawCase> = cases
                    .iter()
                    .filter(|case| case.model_name == model_name)
                    .collect();
                LatencyByModel {
                    model_name: model_name.to_string(),
                    report: LatencyReport::compute_owned(own),
                }
            })
            .collect();

        let per_case = cases
            .iter()
            .map(|case| {
                let wall: Vec<f64> = case
                    .entries
                    .iter()
                    .map(|entry| entry.wall_latency_ms)
                    .collect();
                let mean_wall_ms = if wall.is_empty() {
                    0.0
                } else {
                    wall.iter().sum::<f64>() / wall.len() as f64
                };
                CaseLatency {
                    case_id: case.case_id.clone(),
                    model_name: case.model_name.clone(),
                    entries: case.entries.len(),
                    mean_wall_ms,
                    total_passes: case
                        .entries
                        .iter()
                        .map(|entry| entry.inference_passes.len())
                        .sum(),
                }
            })
            .collect();

        Self {
            overall,
            per_model,
            per_case,
        }
    }
}

impl CorrelationBreakdown {
    pub fn compute(cases: &[RawCase]) -> Self {
        let mut by_index: Vec<EntryIndexStat> = Vec::new();
        let mut index_pairs: Vec<(f64, f64)> = Vec::new();
        for case in cases {
            for entry in &case.entries {
                let error =
                    usize::from(entry.predicted_output_state != Ok(entry.expected_output_state.clone()));
                index_pairs.push((entry.entry_index as f64, error as f64));
                match by_index
                    .iter_mut()
                    .find(|stat| stat.entry_index == entry.entry_index)
                {
                    Some(stat) => {
                        stat.entries += 1;
                        stat.errors += error;
                    }
                    None => by_index.push(EntryIndexStat {
                        entry_index: entry.entry_index,
                        entries: 1,
                        errors: error,
                    }),
                }
            }
        }
        by_index.sort_by_key(|stat| stat.entry_index);

        let mut by_length: Vec<CaseLengthStat> = Vec::new();
        let mut length_pairs: Vec<(f64, f64)> = Vec::new();
        let mut per_case = Vec::new();
        for case in cases {
            let error_count = case
                .entries
                .iter()
                .filter(|entry| entry.predicted_output_state != Ok(entry.expected_output_state.clone()))
                .count();
            let exact = error_count == 0;
            let case_length = case.entries.len();
            length_pairs.push((case_length as f64, usize::from(!exact) as f64));
            match by_length
                .iter_mut()
                .find(|stat| stat.case_length == case_length)
            {
                Some(stat) => {
                    stat.cases += 1;
                    stat.case_errors += usize::from(!exact);
                    stat.total_entry_errors += error_count;
                    stat.total_entries += case_length;
                }
                None => by_length.push(CaseLengthStat {
                    case_length,
                    cases: 1,
                    case_errors: usize::from(!exact),
                    total_entry_errors: error_count,
                    total_entries: case_length,
                }),
            }
            per_case.push(CaseCorrelation {
                case_id: case.case_id.clone(),
                scenario: case.scenario.clone(),
                model_name: case.model_name.clone(),
                case_length,
                error_count,
                exact,
            });
        }
        by_length.sort_by_key(|stat| stat.case_length);

        Self {
            by_entry_index: by_index,
            by_case_length: by_length,
            pearson_entry_index_vs_error: pearson(&index_pairs),
            pearson_case_length_vs_case_error: pearson(&length_pairs),
            per_case,
        }
    }
}

/// Pearson correlation coefficient over `(x, y)` pairs; `None` when fewer
/// than two pairs or either variable has zero variance.
fn pearson(pairs: &[(f64, f64)]) -> Option<f64> {
    let n = pairs.len();
    if n < 2 {
        return None;
    }
    let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / n as f64;
    let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (x, y) in pairs {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    if var_x == 0.0 || var_y == 0.0 {
        return None;
    }
    Some(cov / (var_x * var_y).sqrt())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raw::{RawPass, RawUtterance};
    use std::collections::BTreeMap;
    use stop_core::RoomState;

    fn utterance(
        index: usize,
        expected: &RoomState,
        predicted: Result<RoomState, String>,
    ) -> RawUtterance {
        RawUtterance {
            entry_index: index,
            raw_utterance: format!("u{index}"),
            expected_output_state: expected.clone(),
            predicted_output_state: predicted,
            inference_passes: vec![
                RawPass {
                    latency_ms: 5.0,
                    answers: BTreeMap::new(),
                },
                RawPass {
                    latency_ms: 5.0,
                    answers: BTreeMap::new(),
                },
            ],
            wall_latency_ms: 10.0,
        }
    }

    fn case(id: &str, initial: &RoomState, entries: Vec<RawUtterance>) -> RawCase {
        RawCase {
            case_id: id.to_string(),
            scenario: "test".to_string(),
            model_name: "jev".to_string(),
            initial_state: initial.clone(),
            entries,
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
        let cases = vec![case(
            "c1",
            &initial,
            vec![
                // Action entry (state changes), matched.
                utterance(0, &after, Ok(after.clone())),
                // No-change entry (expected == previous), predicted change = FP.
                utterance(1, &after, Ok(changed_light(&after, 50))),
            ],
        )];

        let report = AccuracyReport::compute(&cases);

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
        let cases = vec![
            case(
                "c1",
                &initial,
                vec![
                    utterance(0, &after, Ok(after.clone())),
                    utterance(1, &after, Ok(after.clone())),
                ],
            ),
            case(
                "c2",
                &initial,
                vec![utterance(0, &after, Ok(initial.clone()))],
            ),
        ];

        let report = AccuracyReport::compute(&cases);

        assert_eq!(report.total_cases, 2);
        assert_eq!(report.exact_cases, 1);
        assert_eq!(report.sequence_exact_match(), 0.5);
    }

    #[test]
    fn failed_entries_never_match() {
        let initial = RoomState::default();
        let cases = vec![case(
            "c1",
            &initial,
            vec![utterance(0, &initial, Err("provider timeout".to_string()))],
        )];
        let report = AccuracyReport::compute(&cases);
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
        let cases = vec![case(
            "c1",
            &initial,
            vec![
                utterance(0, &initial, Ok(initial.clone())),
                utterance(1, &initial, Ok(initial.clone())),
            ],
        )];
        let report = LatencyReport::compute(&cases);
        assert_eq!(report.pass.count, 4);
        assert_eq!(report.pass.mean_ms, 5.0);
        assert_eq!(report.utterance.count, 2);
        assert_eq!(report.utterance.mean_ms, 10.0);
    }

    #[test]
    fn field_accuracy_counts_each_field_per_entry() {
        let initial = RoomState::default();
        let mut wrong = initial.clone();
        wrong.lighting.primary_intensity_pct = 50;
        wrong.table.tilt_degrees = 5;
        let cases = vec![case(
            "c1",
            &initial,
            vec![utterance(0, &initial, Ok(wrong))],
        )];

        let breakdown = AccuracyBreakdown::compute(&cases);

        assert_eq!(breakdown.per_field.len(), FIELDS.len());
        for field in &breakdown.per_field {
            assert_eq!(field.total, 1, "{}", field.field);
        }
        // Only brightness and tilt differ; every other field matches.
        let matched: usize = breakdown.per_field.iter().map(|f| f.matched).sum();
        assert_eq!(matched, FIELDS.len() - 2);
        let brightness = breakdown
            .per_field
            .iter()
            .find(|f| f.field == "light_brightness")
            .expect("brightness field");
        assert_eq!(brightness.matched, 0);
        assert_eq!(brightness.accuracy(), 0.0);
    }

    #[test]
    fn failed_prediction_mismatches_every_field() {
        let initial = RoomState::default();
        let cases = vec![case(
            "c1",
            &initial,
            vec![utterance(0, &initial, Err("timeout".to_string()))],
        )];

        let breakdown = AccuracyBreakdown::compute(&cases);

        for field in &breakdown.per_field {
            assert_eq!(field.matched, 0, "{}", field.field);
            assert_eq!(field.total, 1, "{}", field.field);
        }
    }

    #[test]
    fn groups_by_scenario_and_model() {
        let initial = RoomState::default();
        let after = changed_light(&initial, 70);
        let mut first = case("c1", &initial, vec![utterance(0, &after, Ok(after.clone()))]);
        first.scenario = "cholecystectomy".to_string();
        first.model_name = "model-a".to_string();
        let mut second = case(
            "c2",
            &initial,
            vec![utterance(0, &after, Ok(initial.clone()))],
        );
        second.scenario = "hernia_repair".to_string();
        second.model_name = "model-a".to_string();

        let breakdown = AccuracyBreakdown::compute(&[first, second]);

        assert_eq!(breakdown.per_scenario.len(), 2);
        let scenario = &breakdown.per_scenario[0];
        assert_eq!(scenario.key, "cholecystectomy");
        assert_eq!(scenario.accuracy(), 1.0);
        assert_eq!(scenario.sequence_exact_match(), 1.0);

        assert_eq!(breakdown.per_model.len(), 1);
        let model = &breakdown.per_model[0];
        assert_eq!(model.key, "model-a");
        assert_eq!(model.total_cases, 2);
        assert_eq!(model.total_entries, 2);
        assert_eq!(model.matched_entries, 1);
        assert_eq!(model.accuracy(), 0.5);
    }

    #[test]
    fn per_case_detail_lists_failed_indices() {
        let initial = RoomState::default();
        let cases = vec![case(
            "c1",
            &initial,
            vec![
                utterance(0, &initial, Ok(initial.clone())),
                utterance(1, &initial, Err("boom".to_string())),
            ],
        )];

        let breakdown = AccuracyBreakdown::compute(&cases);

        assert_eq!(breakdown.per_case.len(), 1);
        let detail = &breakdown.per_case[0];
        assert_eq!(detail.case_id, "c1");
        assert_eq!(detail.total_entries, 2);
        assert_eq!(detail.matched_entries, 1);
        assert!(!detail.exact);
        assert_eq!(detail.failed_entry_indices, vec![1]);
    }

    #[test]
    fn latency_breakdown_groups_per_model_and_case() {
        let initial = RoomState::default();
        let mut first = case(
            "c1",
            &initial,
            vec![utterance(0, &initial, Ok(initial.clone()))],
        );
        first.model_name = "model-a".to_string();
        let mut second = case(
            "c2",
            &initial,
            vec![utterance(1, &initial, Ok(initial.clone()))],
        );
        second.model_name = "model-b".to_string();
        second.entries[0].wall_latency_ms = 20.0;

        let breakdown = LatencyBreakdown::compute(&[first, second]);

        assert_eq!(breakdown.overall.utterance.count, 2);
        assert_eq!(breakdown.per_model.len(), 2);
        assert_eq!(breakdown.per_model[0].model_name, "model-a");
        assert_eq!(breakdown.per_model[0].report.utterance.count, 1);
        assert_eq!(breakdown.per_model[1].model_name, "model-b");
        assert_eq!(breakdown.per_model[1].report.utterance.max_ms, 20.0);
        assert_eq!(breakdown.per_case.len(), 2);
        assert_eq!(breakdown.per_case[0].total_passes, 2);
        assert!((breakdown.per_case[1].mean_wall_ms - 20.0).abs() < 1e-9);
    }

    #[test]
    fn correlation_buckets_entry_index_and_case_length() {
        let initial = RoomState::default();
        let after = changed_light(&initial, 70);
        // c1: 3 entries, entry 2 wrong -> length 3, 1 error.
        let long = case(
            "c1",
            &initial,
            vec![
                utterance(0, &after, Ok(after.clone())),
                utterance(1, &after, Ok(after.clone())),
                utterance(2, &after, Ok(initial.clone())),
            ],
        );
        // c2: 1 entry, wrong -> length 1, 1 error.
        let short = case("c2", &initial, vec![utterance(0, &after, Ok(initial.clone()))]);

        let breakdown = CorrelationBreakdown::compute(&[long, short]);

        assert_eq!(
            breakdown.by_entry_index.iter().map(|s| s.entry_index).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(breakdown.by_entry_index[0].entries, 2);
        assert_eq!(breakdown.by_entry_index[2].errors, 1);
        assert_eq!(
            breakdown.by_case_length.iter().map(|s| s.case_length).collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(breakdown.by_case_length[0].cases, 1);
        assert_eq!(breakdown.by_case_length[1].case_errors, 1);
        assert!((breakdown.by_case_length[1].mean_entry_error_rate() - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(breakdown.per_case.len(), 2);
        assert_eq!(breakdown.per_case[0].case_length, 3);
        assert_eq!(breakdown.per_case[0].error_count, 1);
        assert!(!breakdown.per_case[0].exact);
    }

    #[test]
    fn pearson_is_none_without_variance() {
        assert_eq!(pearson(&[]), None);
        assert_eq!(pearson(&[(1.0, 0.0)]), None);
        // y constant: no variance.
        assert_eq!(pearson(&[(1.0, 0.0), (2.0, 0.0)]), None);
        // x constant: no variance.
        assert_eq!(pearson(&[(1.0, 0.0), (1.0, 1.0)]), None);
    }

    #[test]
    fn pearson_matches_known_value() {
        // Perfect negative correlation.
        let value = pearson(&[(1.0, 2.0), (2.0, 1.0), (3.0, 0.0)]).expect("defined");
        assert!((value - (-1.0)).abs() < 1e-9);
        // Perfect positive correlation.
        let value = pearson(&[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]).expect("defined");
        assert!((value - 1.0).abs() < 1e-9);
    }

    #[test]
    fn breakdowns_serialize_to_json() {
        let initial = RoomState::default();
        let cases = vec![case("c1", &initial, vec![utterance(0, &initial, Ok(initial.clone()))])];

        let accuracy = serde_json::to_string(&AccuracyBreakdown::compute(&cases)).expect("serialize");
        assert!(accuracy.contains("\"per_field\""), "{accuracy}");
        assert!(accuracy.contains("light_brightness"), "{accuracy}");

        let latency = serde_json::to_string(&LatencyBreakdown::compute(&cases)).expect("serialize");
        assert!(latency.contains("\"per_model\""), "{latency}");

        let correlation =
            serde_json::to_string(&CorrelationBreakdown::compute(&cases)).expect("serialize");
        assert!(correlation.contains("pearson_entry_index_vs_error"), "{correlation}");
    }
}
