//! Errors of the dataset generator: OpenRouter transport/protocol failures
//! and inconsistent model extractions.

use std::time::Duration;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("missing environment variable {0}")]
    MissingEnv(String),

    #[error("transport error: {0}")]
    Transport(String),

    #[error("HTTP status {status}: {body}")]
    Status {
        status: u16,
        body: String,
        /// Server-provided `Retry-After` hint (rate limiting).
        retry_after: Option<Duration>,
    },

    #[error("malformed model response: {0}")]
    Malformed(String),
}

impl DatasetError {
    /// Server-provided retry hint, when the error carries one.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            DatasetError::Status { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}
