//! `stop-dataset`: synthetic ground-truth dataset generation via OpenRouter.
//!
//! Provides the dataset schema shared with `stop-benchmark`, the OpenRouter
//! JSON completion client, and the two-step scenario generator behind the
//! `generate-data` entry point (package main binary). See
//! `docs/INSTRUCTIONS.md` section 4.

pub mod error;
pub mod generator;
pub mod openrouter;
pub mod schema;

pub use error::DatasetError;
pub use generator::{
    Generator, GeneratorConfig, UtteranceType, build_utterance_type_sequence, normalize_state,
};
pub use schema::{DatasetCase, HistoryEntry};
