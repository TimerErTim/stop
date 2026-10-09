//! OpenRouter chat-completions client (JSON-only replies).
//!
//! The generator drives two structured-prompting steps (transcript draft and
//! command extraction) through [`OpenRouterClient::complete_json`]. High model
//! latency is expected, so the client is pure async reqwest and never blocks.

use reqwest::Client;
use serde_json::Value;

use crate::error::DatasetError;

pub const DEFAULT_OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub const DEFAULT_OPENROUTER_MODEL: &str = "inclusionai/ling-3.0-flash-vl:floor";

/// One JSON-producing chat completion against OpenRouter.
pub struct OpenRouterClient {
    http: Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl OpenRouterClient {
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            http: Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    /// Reads `OPENROUTER_API_KEY` (required) and the optional
    /// `OPENROUTER_MODEL` and `OPENROUTER_BASE_URL` overrides from the
    /// environment.
    pub fn from_env() -> Result<Self, DatasetError> {
        let api_key = std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| DatasetError::MissingEnv("OPENROUTER_API_KEY".into()))?;
        if api_key.is_empty() {
            return Err(DatasetError::MissingEnv("OPENROUTER_API_KEY".into()));
        }
        let model = std::env::var("OPENROUTER_MODEL")
            .unwrap_or_else(|_| DEFAULT_OPENROUTER_MODEL.to_string());
        let base_url = std::env::var("OPENROUTER_BASE_URL")
            .unwrap_or_else(|_| DEFAULT_OPENROUTER_BASE_URL.to_string());
        Ok(Self::new(base_url, api_key, model))
    }

    /// Model id used for generation (stored in each dataset case).
    pub fn model(&self) -> &str {
        &self.model
    }

    /// One chat completion constrained to a JSON object reply; returns the
    /// parsed JSON payload.
    pub async fn complete_json(&self, system: &str, user: &str) -> Result<Value, DatasetError> {
        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                { "role": "system", "content": system },
                { "role": "user", "content": user },
            ],
            "response_format": { "type": "json_object" },
        });

        let response = self
            .http
            .post(format!("{}/chat/completions", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| DatasetError::Transport(e.to_string()))?;

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| DatasetError::Transport(e.to_string()))?;
        if !status.is_success() {
            return Err(DatasetError::Status {
                status: status.as_u16(),
                body: text.chars().take(512).collect(),
            });
        }

        let value: Value =
            serde_json::from_str(&text).map_err(|e| DatasetError::Malformed(e.to_string()))?;
        let content = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_str)
            .ok_or_else(|| DatasetError::Malformed("missing choices[0].message.content".into()))?;

        parse_json_content(content)
    }
}

/// Strips optional Markdown code fences and parses the JSON payload.
fn parse_json_content(content: &str) -> Result<Value, DatasetError> {
    let trimmed = content.trim();
    let stripped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|rest| rest.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(trimmed);
    serde_json::from_str(stripped).map_err(|e| DatasetError::Malformed(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_json() {
        let value = parse_json_content(r#"{"a": 1}"#).expect("parse");
        assert_eq!(value["a"], 1);
    }

    #[test]
    fn parses_fenced_json() {
        let value = parse_json_content("```json\n{\"a\": 2}\n```").expect("parse");
        assert_eq!(value["a"], 2);
    }

    #[test]
    fn rejects_non_json() {
        assert!(matches!(
            parse_json_content("sorry, no json here"),
            Err(DatasetError::Malformed(_))
        ));
    }
}
