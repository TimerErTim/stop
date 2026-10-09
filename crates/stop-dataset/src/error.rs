//! Errors of the dataset generator: OpenRouter transport/protocol failures
//! and inconsistent model extractions.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("missing environment variable {0}")]
    MissingEnv(String),

    #[error("transport error: {0}")]
    Transport(String),

    #[error("HTTP status {status}: {body}")]
    Status { status: u16, body: String },

    #[error("malformed model response: {0}")]
    Malformed(String),

    #[error("inconsistent extraction: {0}")]
    Inconsistent(String),
}
