//! Pure metric computation over raw benchmark cases.
//!
//! Scoring model: per-utterance exact match of each prediction variant
//! (fresh, rolling) against the expected state. Entry classes are derived
//! from the expected states: an entry is an "action" entry when its
//! expected state differs from the previous expected state (case start:
//! initial state), otherwise a "no-change" entry whose only correct
//! prediction is the unchanged state (noise semantics, spec 4.3).

use serde::{Deserialize, Serialize};
use stop_core::RoomState;

use crate::raw::{RawCase, RawPrediction, RawUtterance};

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
    /// Scores every entry against its expected state, per variant.
    pub fn compute_variants(
        cases: &[RawCase],
        state: impl Fn(&RawUtterance) -> &Result<stop_core::RoomState, String>,
    ) -> Self {
        let mut report = Self::default();
        for case in cases {
            report.total_cases += 1;
            let mut case_exact = true;
            let mut previous_expected = Some(case.initial_state.clone());
            for entry in &case.entries {
                let predicted = state(entry);
                let matched = *predicted == Ok(entry.expected_state.clone());
                let is_action = previous_expected
                    .as_ref()
                    .is_some_and(|prev| *prev != entry.expected_state);
                previous_expected = Some(entry.expected_state.clone());

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

    /// Fresh variant (ground-truth chaining) accuracy.
    pub fn compute_fresh(cases: &[RawCase]) -> Self {
        Self::compute_variants(cases, |entry| &entry.fresh_prediction.state)
    }

    /// Rolling variant (self-chaining) accuracy.
    pub fn compute_rolling(cases: &[RawCase]) -> Self {
        Self::compute_variants(cases, |entry| &entry.rolling_prediction.state)
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

/// One case's own score per variant plus the entry indices that failed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseAccuracy {
    pub case_id: String,
    pub scenario: String,
    pub model_name: String,
    pub total_entries: usize,
    pub fresh: CaseAccuracySide,
    pub rolling: CaseAccuracySide,
}

/// One variant's per-case match summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseAccuracySide {
    pub matched_entries: usize,
    /// Every entry matched.
    pub exact: bool,
    /// Entry indices whose prediction failed (`Err`) or mismatched.
    pub failed_entry_indices: Vec<usize>,
}

/// Full accuracy evaluation output: totals plus every split, JSON-serializable.
/// All metrics are computed per prediction variant (fresh, rolling).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccuracyBreakdown {
    /// Whole-input totals (entry classes, sequence exact match), per variant.
    pub total: VariantAccuracy,
    pub per_field: Vec<VariantFieldAccuracy>,
    pub per_scenario: Vec<VariantGroupAccuracy>,
    pub per_model: Vec<VariantGroupAccuracy>,
    pub per_case: Vec<CaseAccuracy>,
}

/// One accuracy metric set for both prediction variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantAccuracy {
    /// Fresh variant (ground-truth chaining).
    pub fresh: AccuracyReport,
    /// Rolling variant (self-chaining).
    pub rolling: AccuracyReport,
}

/// One field's accuracy for both prediction variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantFieldAccuracy {
    pub field: String,
    pub fresh: FieldAccuracy,
    pub rolling: FieldAccuracy,
}

/// One group's (scenario, model) accuracy for both prediction variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariantGroupAccuracy {
    pub key: String,
    pub fresh: GroupAccuracy,
    pub rolling: GroupAccuracy,
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

/// Per-pass and per-utterance latency distributions, per prediction variant.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct LatencyReport {
    pub pass: Stats,
    pub utterance: Stats,
}

impl LatencyReport {
    /// Aggregates latencies of one prediction variant of every entry.
    pub fn compute_variants<'a>(
        cases: impl IntoIterator<Item = &'a RawCase>,
        prediction: impl Fn(&'a RawUtterance) -> &'a RawPrediction,
    ) -> Self {
        let mut pass_samples = Vec::new();
        let mut utterance_samples = Vec::new();
        for case in cases {
            for entry in &case.entries {
                let variant = prediction(entry);
                pass_samples.extend(variant.inference_passes.iter().map(|pass| pass.latency_ms));
                utterance_samples.push(variant.wall_latency_ms);
            }
        }
        Self {
            pass: Stats::from_samples(pass_samples),
            utterance: Stats::from_samples(utterance_samples),
        }
    }

    /// Fresh variant (ground-truth chaining) latencies.
    pub fn compute_fresh<'a>(cases: impl IntoIterator<Item = &'a RawCase>) -> Self {
        Self::compute_variants(cases, |entry: &'a RawUtterance| &entry.fresh_prediction)
    }

    /// Rolling variant (self-chaining) latencies.
    pub fn compute_rolling<'a>(cases: impl IntoIterator<Item = &'a RawCase>) -> Self {
        Self::compute_variants(cases, |entry: &'a RawUtterance| &entry.rolling_prediction)
    }
}

/// Per-case latency summary, per variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseLatency {
    pub case_id: String,
    pub model_name: String,
    pub entries: usize,
    pub fresh: CaseLatencySide,
    pub rolling: CaseLatencySide,
}

/// One variant's per-case latency summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseLatencySide {
    /// Mean wall-clock latency across this case's utterances, milliseconds.
    pub mean_wall_ms: f64,
    /// Total inference passes (failed attempts included).
    pub total_passes: usize,
}

/// Latency evaluation output: overall plus per-model and per-case splits,
/// per prediction variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyBreakdown {
    pub overall: LatencyBreakdownSides,
    pub per_model: Vec<LatencyByModel>,
    pub per_case: Vec<CaseLatency>,
}

/// One latency metric set for both prediction variants.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyBreakdownSides {
    pub fresh: LatencyReport,
    pub rolling: LatencyReport,
}

/// One model's latency distribution, per variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatencyByModel {
    pub model_name: String,
    pub report: LatencyBreakdownSides,
}

/// Error-rate statistics for one `entry_index` position, per variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntryIndexStat {
    pub entry_index: usize,
    pub fresh: ErrorCount,
    pub rolling: ErrorCount,
}

/// One variant's error tally.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ErrorCount {
    pub entries: usize,
    pub errors: usize,
}

impl ErrorCount {
    /// Errors / entries, `0.0` when empty.
    pub fn error_rate(&self) -> f64 {
        ratio(self.errors, self.entries)
    }
}

impl EntryIndexStat {
    /// Errors / entries, `0.0` when empty.
    pub fn error_rate(&self) -> f64 {
        ratio(
            self.fresh.errors + self.rolling.errors,
            self.fresh.entries + self.rolling.entries,
        )
    }
}

/// Error-rate statistics for one case-length bucket (case length = number of
/// history entries), per variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseLengthStat {
    pub case_length: usize,
    pub fresh: ErrorCount,
    pub rolling: ErrorCount,
}

impl CaseLengthStat {
    /// Cases that were not exact / cases, `0.0` when empty. Case errors are
    /// entry-error-based: a case errs when any entry mismatched.
    pub fn case_error_rate(&self) -> f64 {
        ratio(
            self.fresh.errors + self.rolling.errors,
            self.fresh.entries + self.rolling.entries,
        )
    }

    /// Entry errors / entries within this bucket, `0.0` when empty.
    pub fn mean_entry_error_rate(&self) -> f64 {
        ratio(
            self.fresh.errors + self.rolling.errors,
            self.fresh.entries + self.rolling.entries,
        )
    }
}

/// One case's length, error counts, and exactness, per variant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseCorrelation {
    pub case_id: String,
    pub scenario: String,
    pub model_name: String,
    pub case_length: usize,
    pub fresh: ErrorCount,
    pub rolling: ErrorCount,
}

/// Correlation evaluation output: error rate by entry index and by case
/// length, plus Pearson correlations (null when undefined, i.e. no variance
/// or empty input), per variant.
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
        let total = VariantAccuracy {
            fresh: AccuracyReport::compute_fresh(cases),
            rolling: AccuracyReport::compute_rolling(cases),
        };
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

/// Per-field accuracy: each field counts once per entry, per variant.
fn field_accuracy(cases: &[RawCase]) -> Vec<VariantFieldAccuracy> {
    let mut fresh: Vec<FieldAccuracy> = FIELDS
        .iter()
        .map(|field| FieldAccuracy {
            field: (*field).to_string(),
            matched: 0,
            total: 0,
        })
        .collect();
    let mut rolling: Vec<FieldAccuracy> = fresh.clone();
    for case in cases {
        for entry in &case.entries {
            for (reports, predicted) in [
                (&mut fresh, entry.fresh_prediction.state.as_ref().ok()),
                (&mut rolling, entry.rolling_prediction.state.as_ref().ok()),
            ] {
                for (report, matched) in reports
                    .iter_mut()
                    .zip(field_matches(&entry.expected_state, predicted))
                {
                    report.total += 1;
                    report.matched += usize::from(matched);
                }
            }
        }
    }
    fresh
        .into_iter()
        .zip(rolling)
        .map(|(fresh, rolling)| VariantFieldAccuracy {
            field: fresh.field.clone(),
            fresh,
            rolling,
        })
        .collect()
}

/// Per-field match of one entry: each field is `true` when the predicted
/// value equals the expected value. A failed prediction (`None`) mismatches
/// every field.
fn field_matches(
    expected: &stop_core::RoomState,
    predicted: Option<&stop_core::RoomState>,
) -> [bool; 11] {
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

/// Groups entries and cases by `key`, preserving first-seen order; per variant.
fn group_accuracy<F>(cases: &[RawCase], key: F) -> Vec<VariantGroupAccuracy>
where
    F: Fn(&RawCase) -> String,
{
    let mut groups: Vec<VariantGroupAccuracy> = Vec::new();
    for case in cases {
        let key = key(case);
        let fresh = GroupAccuracy {
            key: key.clone(),
            total_cases: 1,
            total_entries: case.entries.len(),
            matched_entries: case
                .entries
                .iter()
                .filter(|entry| entry.fresh_prediction.state == Ok(entry.expected_state.clone()))
                .count(),
            exact_cases: usize::from(
                case.entries
                    .iter()
                    .all(|entry| entry.fresh_prediction.state == Ok(entry.expected_state.clone())),
            ),
        };
        let rolling = GroupAccuracy {
            key: key.clone(),
            total_cases: 1,
            total_entries: case.entries.len(),
            matched_entries: case
                .entries
                .iter()
                .filter(|entry| entry.rolling_prediction.state == Ok(entry.expected_state.clone()))
                .count(),
            exact_cases: usize::from(
                case.entries.iter().all(|entry| {
                    entry.rolling_prediction.state == Ok(entry.expected_state.clone())
                }),
            ),
        };
        match groups.iter_mut().find(|group| group.key == key) {
            Some(group) => {
                group.fresh.total_cases += 1;
                group.fresh.exact_cases += fresh.exact_cases;
                group.fresh.total_entries += fresh.total_entries;
                group.fresh.matched_entries += fresh.matched_entries;
                group.rolling.total_cases += 1;
                group.rolling.exact_cases += rolling.exact_cases;
                group.rolling.total_entries += rolling.total_entries;
                group.rolling.matched_entries += rolling.matched_entries;
            }
            None => {
                groups.push(VariantGroupAccuracy {
                    key,
                    fresh,
                    rolling,
                });
            }
        };
    }
    groups
}

fn case_accuracy(case: &RawCase) -> CaseAccuracy {
    fn side(
        case: &RawCase,
        state: fn(&RawUtterance) -> &Result<RoomState, String>,
    ) -> CaseAccuracySide {
        let mut failed_entry_indices = Vec::new();
        for entry in &case.entries {
            if state(entry) != &Ok(entry.expected_state.clone()) {
                failed_entry_indices.push(entry.entry_index);
            }
        }
        CaseAccuracySide {
            matched_entries: case.entries.len() - failed_entry_indices.len(),
            exact: failed_entry_indices.is_empty(),
            failed_entry_indices,
        }
    }
    CaseAccuracy {
        case_id: case.case_id.clone(),
        scenario: case.scenario.clone(),
        model_name: case.model_name.clone(),
        total_entries: case.entries.len(),
        fresh: side(case, |entry| &entry.fresh_prediction.state),
        rolling: side(case, |entry| &entry.rolling_prediction.state),
    }
}

impl LatencyBreakdown {
    pub fn compute(cases: &[RawCase]) -> Self {
        let overall = LatencyBreakdownSides {
            fresh: LatencyReport::compute_fresh(cases),
            rolling: LatencyReport::compute_rolling(cases),
        };

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
                    report: LatencyBreakdownSides {
                        fresh: LatencyReport::compute_fresh(own.clone()),
                        rolling: LatencyReport::compute_rolling(own),
                    },
                }
            })
            .collect();

        let per_case = cases
            .iter()
            .map(|case| {
                fn side(
                    case: &RawCase,
                    prediction: fn(&RawUtterance) -> &RawPrediction,
                ) -> (f64, usize) {
                    let wall: Vec<f64> = case
                        .entries
                        .iter()
                        .map(|entry| prediction(entry).wall_latency_ms)
                        .collect();
                    let mean_wall_ms = if wall.is_empty() {
                        0.0
                    } else {
                        wall.iter().sum::<f64>() / wall.len() as f64
                    };
                    let total_passes = case
                        .entries
                        .iter()
                        .map(|entry| prediction(entry).inference_passes.len())
                        .sum();
                    (mean_wall_ms, total_passes)
                }
                let (fresh_mean, fresh_passes) = side(case, |entry| &entry.fresh_prediction);
                let (rolling_mean, rolling_passes) = side(case, |entry| &entry.rolling_prediction);
                CaseLatency {
                    case_id: case.case_id.clone(),
                    model_name: case.model_name.clone(),
                    entries: case.entries.len(),
                    fresh: CaseLatencySide {
                        mean_wall_ms: fresh_mean,
                        total_passes: fresh_passes,
                    },
                    rolling: CaseLatencySide {
                        mean_wall_ms: rolling_mean,
                        total_passes: rolling_passes,
                    },
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
        let is_error = |state: &Result<RoomState, String>, expected: &RoomState| {
            usize::from(*state != Ok(expected.clone()))
        };

        let mut by_index: Vec<EntryIndexStat> = Vec::new();
        let mut index_pairs: Vec<(f64, f64)> = Vec::new();
        for case in cases {
            for entry in &case.entries {
                let fresh_error = is_error(&entry.fresh_prediction.state, &entry.expected_state);
                let rolling_error =
                    is_error(&entry.rolling_prediction.state, &entry.expected_state);
                index_pairs.push((entry.entry_index as f64, fresh_error as f64));
                index_pairs.push((entry.entry_index as f64, rolling_error as f64));
                match by_index
                    .iter_mut()
                    .find(|stat| stat.entry_index == entry.entry_index)
                {
                    Some(stat) => {
                        stat.fresh.entries += 1;
                        stat.fresh.errors += fresh_error;
                        stat.rolling.entries += 1;
                        stat.rolling.errors += rolling_error;
                    }
                    None => by_index.push(EntryIndexStat {
                        entry_index: entry.entry_index,
                        fresh: ErrorCount {
                            entries: 1,
                            errors: fresh_error,
                        },
                        rolling: ErrorCount {
                            entries: 1,
                            errors: rolling_error,
                        },
                    }),
                }
            }
        }
        by_index.sort_by_key(|stat| stat.entry_index);

        let mut by_length: Vec<CaseLengthStat> = Vec::new();
        let mut length_pairs: Vec<(f64, f64)> = Vec::new();
        let mut per_case = Vec::new();
        for case in cases {
            let fresh_errors = case
                .entries
                .iter()
                .filter(|entry| entry.fresh_prediction.state != Ok(entry.expected_state.clone()))
                .count();
            let rolling_errors = case
                .entries
                .iter()
                .filter(|entry| entry.rolling_prediction.state != Ok(entry.expected_state.clone()))
                .count();
            let case_length = case.entries.len();
            length_pairs.push((case_length as f64, usize::from(fresh_errors > 0) as f64));
            length_pairs.push((case_length as f64, usize::from(rolling_errors > 0) as f64));
            match by_length
                .iter_mut()
                .find(|stat| stat.case_length == case_length)
            {
                Some(stat) => {
                    stat.fresh.entries += case_length;
                    stat.fresh.errors += fresh_errors;
                    stat.rolling.entries += case_length;
                    stat.rolling.errors += rolling_errors;
                }
                None => by_length.push(CaseLengthStat {
                    case_length,
                    fresh: ErrorCount {
                        entries: case_length,
                        errors: fresh_errors,
                    },
                    rolling: ErrorCount {
                        entries: case_length,
                        errors: rolling_errors,
                    },
                }),
            }
            per_case.push(CaseCorrelation {
                case_id: case.case_id.clone(),
                scenario: case.scenario.clone(),
                model_name: case.model_name.clone(),
                case_length,
                fresh: ErrorCount {
                    entries: case_length,
                    errors: fresh_errors,
                },
                rolling: ErrorCount {
                    entries: case_length,
                    errors: rolling_errors,
                },
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

    fn prediction(
        state: Result<RoomState, String>,
        passes: Vec<RawPass>,
        wall: f64,
    ) -> crate::raw::RawPrediction {
        crate::raw::RawPrediction {
            state,
            inference_passes: passes,
            wall_latency_ms: wall,
        }
    }

    fn utterance(
        index: usize,
        expected: &RoomState,
        predicted: Result<RoomState, String>,
    ) -> RawUtterance {
        let passes = vec![
            RawPass {
                latency_ms: 5.0,
                answers: BTreeMap::new(),
            },
            RawPass {
                latency_ms: 5.0,
                answers: BTreeMap::new(),
            },
        ];
        RawUtterance {
            entry_index: index,
            raw_utterance: format!("u{index}"),
            expected_state: expected.clone(),
            fresh_prediction: prediction(predicted.clone(), passes.clone(), 10.0),
            rolling_prediction: prediction(predicted, passes, 10.0),
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

        for report in [
            AccuracyReport::compute_fresh(&cases),
            AccuracyReport::compute_rolling(&cases),
        ] {
            assert_eq!(report.total_entries, 2);
            assert_eq!(report.action_entries, 1);
            assert_eq!(report.action_matched, 1);
            assert_eq!(report.no_change_entries, 1);
            assert_eq!(report.no_change_matched, 0);
            assert_eq!(report.no_change_false_positives(), 1);
            assert_eq!(report.accuracy(), 0.5);
            assert_eq!(report.sequence_exact_match(), 0.0);
        }
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

        for report in [
            AccuracyReport::compute_fresh(&cases),
            AccuracyReport::compute_rolling(&cases),
        ] {
            assert_eq!(report.total_cases, 2);
            assert_eq!(report.exact_cases, 1);
            assert_eq!(report.sequence_exact_match(), 0.5);
        }
    }

    #[test]
    fn failed_entries_never_match() {
        let initial = RoomState::default();
        let cases = vec![case(
            "c1",
            &initial,
            vec![utterance(0, &initial, Err("provider timeout".to_string()))],
        )];
        for report in [
            AccuracyReport::compute_fresh(&cases),
            AccuracyReport::compute_rolling(&cases),
        ] {
            assert_eq!(report.matched_entries, 0);
            assert_eq!(report.no_change_entries, 1);
            assert_eq!(report.no_change_false_positives(), 1);
        }
    }

    #[test]
    fn variants_score_independently() {
        let initial = RoomState::default();
        let after = changed_light(&initial, 70);
        let mut cases = vec![case(
            "c1",
            &initial,
            vec![utterance(0, &after, Ok(after.clone()))],
        )];
        // Fresh matched, rolling mismatched.
        cases[0].entries[0].rolling_prediction.state = Ok(initial.clone());

        let breakdown = AccuracyBreakdown::compute(&cases);
        assert_eq!(breakdown.total.fresh.matched_entries, 1);
        assert_eq!(breakdown.total.rolling.matched_entries, 0);
        assert_eq!(breakdown.total.fresh.accuracy(), 1.0);
        assert_eq!(breakdown.total.rolling.accuracy(), 0.0);
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
        for report in [
            LatencyReport::compute_fresh(&cases),
            LatencyReport::compute_rolling(&cases),
        ] {
            assert_eq!(report.pass.count, 4);
            assert_eq!(report.pass.mean_ms, 5.0);
            assert_eq!(report.utterance.count, 2);
            assert_eq!(report.utterance.mean_ms, 10.0);
        }
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
            assert_eq!(field.fresh.total, 1, "{}", field.field);
            assert_eq!(field.rolling.total, 1, "{}", field.field);
        }
        // Only brightness and tilt differ; every other field matches.
        let matched: usize = breakdown
            .per_field
            .iter()
            .map(|f| f.fresh.matched + f.rolling.matched)
            .sum();
        assert_eq!(matched, 2 * (FIELDS.len() - 2));
        let brightness = breakdown
            .per_field
            .iter()
            .find(|f| f.field == "light_brightness")
            .expect("brightness field");
        assert_eq!(brightness.fresh.matched, 0);
        assert_eq!(brightness.rolling.matched, 0);
        assert_eq!(brightness.fresh.accuracy(), 0.0);
        assert_eq!(brightness.rolling.accuracy(), 0.0);
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
            assert_eq!(field.fresh.matched, 0, "{}", field.field);
            assert_eq!(field.fresh.total, 1, "{}", field.field);
            assert_eq!(field.rolling.matched, 0, "{}", field.field);
            assert_eq!(field.rolling.total, 1, "{}", field.field);
        }
    }

    #[test]
    fn groups_by_scenario_and_model() {
        let initial = RoomState::default();
        let after = changed_light(&initial, 70);
        let mut first = case(
            "c1",
            &initial,
            vec![utterance(0, &after, Ok(after.clone()))],
        );
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
        assert_eq!(scenario.fresh.accuracy(), 1.0);
        assert_eq!(scenario.rolling.accuracy(), 1.0);
        assert_eq!(scenario.fresh.sequence_exact_match(), 1.0);
        assert_eq!(scenario.rolling.sequence_exact_match(), 1.0);

        assert_eq!(breakdown.per_model.len(), 1);
        let model = &breakdown.per_model[0];
        assert_eq!(model.key, "model-a");
        assert_eq!(model.fresh.total_cases, 2);
        assert_eq!(model.fresh.total_entries, 2);
        assert_eq!(model.fresh.matched_entries, 1);
        assert_eq!(model.fresh.accuracy(), 0.5);
        assert_eq!(model.rolling.accuracy(), 0.5);
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
        assert_eq!(detail.fresh.matched_entries, 1);
        assert!(!detail.fresh.exact);
        assert_eq!(detail.fresh.failed_entry_indices, vec![1]);
        assert_eq!(detail.rolling.failed_entry_indices, vec![1]);
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
        second.entries[0].fresh_prediction.wall_latency_ms = 20.0;
        second.entries[0].rolling_prediction.wall_latency_ms = 25.0;

        let breakdown = LatencyBreakdown::compute(&[first, second]);

        assert_eq!(breakdown.overall.fresh.utterance.count, 2);
        assert_eq!(breakdown.overall.rolling.utterance.count, 2);
        assert_eq!(breakdown.per_model.len(), 2);
        assert_eq!(breakdown.per_model[0].model_name, "model-a");
        assert_eq!(breakdown.per_model[0].report.fresh.utterance.count, 1);
        assert_eq!(breakdown.per_model[1].model_name, "model-b");
        assert_eq!(breakdown.per_model[1].report.fresh.utterance.max_ms, 20.0);
        assert_eq!(breakdown.per_model[1].report.rolling.utterance.max_ms, 25.0);
        assert_eq!(breakdown.per_case.len(), 2);
        assert_eq!(breakdown.per_case[0].fresh.total_passes, 2);
        assert!((breakdown.per_case[1].fresh.mean_wall_ms - 20.0).abs() < 1e-9);
        assert!((breakdown.per_case[1].rolling.mean_wall_ms - 25.0).abs() < 1e-9);
    }

    #[test]
    fn correlation_buckets_entry_index_and_case_length() {
        let initial = RoomState::default();
        let after = changed_light(&initial, 70);
        // c1: 3 entries, entry 2 wrong -> length 3, 1 error per variant.
        let long = case(
            "c1",
            &initial,
            vec![
                utterance(0, &after, Ok(after.clone())),
                utterance(1, &after, Ok(after.clone())),
                utterance(2, &after, Ok(initial.clone())),
            ],
        );
        // c2: 1 entry, wrong -> length 1, 1 error per variant.
        let short = case(
            "c2",
            &initial,
            vec![utterance(0, &after, Ok(initial.clone()))],
        );

        let breakdown = CorrelationBreakdown::compute(&[long, short]);

        assert_eq!(
            breakdown
                .by_entry_index
                .iter()
                .map(|s| s.entry_index)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(breakdown.by_entry_index[0].fresh.entries, 2);
        assert_eq!(breakdown.by_entry_index[0].rolling.entries, 2);
        assert_eq!(breakdown.by_entry_index[2].fresh.errors, 1);
        assert_eq!(breakdown.by_entry_index[2].rolling.errors, 1);
        assert_eq!(
            breakdown
                .by_case_length
                .iter()
                .map(|s| s.case_length)
                .collect::<Vec<_>>(),
            vec![1, 3]
        );
        assert_eq!(breakdown.by_case_length[0].fresh.entries, 1);
        assert_eq!(breakdown.by_case_length[1].fresh.errors, 1);
        assert_eq!(breakdown.by_case_length[1].rolling.errors, 1);
        assert!((breakdown.by_case_length[1].mean_entry_error_rate() - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(breakdown.per_case.len(), 2);
        assert_eq!(breakdown.per_case[0].case_length, 3);
        assert_eq!(breakdown.per_case[0].fresh.errors, 1);
        assert_eq!(breakdown.per_case[0].rolling.errors, 1);
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
        let cases = vec![case(
            "c1",
            &initial,
            vec![utterance(0, &initial, Ok(initial.clone()))],
        )];

        let accuracy =
            serde_json::to_string(&AccuracyBreakdown::compute(&cases)).expect("serialize");
        assert!(accuracy.contains("\"per_field\""), "{accuracy}");
        assert!(accuracy.contains("light_brightness"), "{accuracy}");
        assert!(accuracy.contains("\"fresh\""), "{accuracy}");
        assert!(accuracy.contains("\"rolling\""), "{accuracy}");

        let latency = serde_json::to_string(&LatencyBreakdown::compute(&cases)).expect("serialize");
        assert!(latency.contains("\"per_model\""), "{latency}");

        let correlation =
            serde_json::to_string(&CorrelationBreakdown::compute(&cases)).expect("serialize");
        assert!(
            correlation.contains("pearson_entry_index_vs_error"),
            "{correlation}"
        );
    }
}
