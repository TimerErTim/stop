//! `stop-benchmark`: benchmark runner and offline evaluation binaries.
//!
//! `run-benchmark` (package main binary) executes every dataset case against
//! the live System-One provider once and persists the raw per-case results
//! (one JSONL line per case); `eval-accuracy` and `eval-latency` then work
//! purely on that raw output. Scoring compares the predicted room state per
//! utterance with the expected state. See `docs/INSTRUCTIONS.md` section 5.

pub mod metrics;
pub mod raw;
pub mod report;
pub mod run;

pub use metrics::{AccuracyReport, LatencyReport, Stats};
pub use raw::{
    BenchmarkError, RawCase, RawPass, RawUtterance, expand_inputs, load_cases, load_raw_cases,
    load_raw_cases_multi,
};
