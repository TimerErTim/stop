//! `stop-benchmark`: benchmark runner and offline evaluation binaries.
//!
//! `run-benchmark` executes every dataset case against the live System-One
//! provider once and persists the raw per-utterance results; `eval-accuracy`
//! and `eval-latency` then work purely on that raw output. Scoring compares
//! the predicted final room state per utterance with the expected state.
//! See `docs/INSTRUCTIONS.md` section 5.

pub mod metrics;
pub mod raw;
pub mod report;

pub use metrics::{AccuracyReport, LatencyReport, Stats};
pub use raw::{BenchmarkError, RawEntry, load_cases, load_raw_entries};
