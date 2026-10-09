//! Typed System-One / JevK5 HTTP client (`POST {base}/v1/systemone`).
//!
//! One pass maps to one request and returns ALL decisions at once: one
//! question group per room object (choice with a `null` = no-change option,
//! plus conditional absolute-target questions) and two global flags
//! (`emergency_stop`, `requires_sterile_confirm`). Request and answer shapes
//! follow the documented System-One contract; the self-hosted JevK5 server
//! (`jevk5-serve`) speaks the same shape.
//!
//! Latency: high model latency is expected, so this client is pure async
//! reqwest, never blocks, and enforces a hard request timeout.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::decision::{ActionKind, DeviceDecision, TargetDevice, UtteranceDecision};
use crate::engine::{InferenceInput, InferenceOutcome, InferencePort, ProviderError};

/// Hard request timeout (1 minute): a hung JevK5 must not stall the pipeline.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Brightness rubric (10 levels, 0-100 %).
const BRIGHTNESS_LEVELS: [i16; 10] = [0, 10, 20, 30, 40, 50, 60, 75, 90, 100];
/// Zoom rubric levels 1-5.
const ZOOM_LEVELS: [i16; 5] = [1, 2, 3, 4, 5];
/// Pressure rubric (10 levels, 0-25 mmHg, typical 12-14 explicit).
const PRESSURE_LEVELS: [i16; 10] = [0, 5, 8, 10, 12, 14, 16, 19, 22, 25];
/// Table tilt rubric (9 levels, -15..+15 deg).
const TILT_LEVELS: [i16; 9] = [-15, -12, -8, -4, 0, 4, 8, 12, 15];
/// Table height rubric (7 levels, 70-130 cm).
const HEIGHT_LEVELS: [i16; 7] = [70, 80, 90, 100, 110, 120, 130];
/// Light mode rubric order, mapped to `LightMode` codes 0/1/2.
const LIGHT_MODES: [(&str, i16); 3] = [("Normal", 0), ("CavityFocus", 1), ("AmbientRed", 2)];

/// Choice keys that mean "no change" for an object group.
const NO_CHANGE_KEYS: [&str; 4] = ["null", "None", "NoChange", "Idle"];

/// Which absolute-target question applies to a value-setting action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbsoluteTarget {
    Brightness,
    Zoom,
    Pressure,
    Tilt,
    Height,
    LightMode,
}

/// Absolute targets are the only value source on the wire (the action
/// direction is intent metadata); toggles carry no target.
fn absolute_target_for(action: ActionKind) -> Option<AbsoluteTarget> {
    match action {
        ActionKind::IncreaseBrightness | ActionKind::DecreaseBrightness => {
            Some(AbsoluteTarget::Brightness)
        }
        ActionKind::SetLightMode => Some(AbsoluteTarget::LightMode),
        ActionKind::ZoomIn | ActionKind::ZoomOut => Some(AbsoluteTarget::Zoom),
        ActionKind::AdjustPressure => Some(AbsoluteTarget::Pressure),
        ActionKind::TiltTable => Some(AbsoluteTarget::Tilt),
        ActionKind::SetTableHeight => Some(AbsoluteTarget::Height),
        _ => None,
    }
}

/// `SystemOneClient`: async provider for the JevK5 / System-One endpoint.
pub struct SystemOneClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl SystemOneClient {
    /// `base_url` without a trailing path, e.g. `http://localhost:8080`.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            http: timeout_client(REQUEST_TIMEOUT),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key: None,
        }
    }

    /// Overrides the request timeout (tests, benchmarks).
    pub fn with_request_timeout(self, timeout: Duration) -> Self {
        Self {
            http: timeout_client(timeout),
            ..self
        }
    }

    /// Reads `SYSTEMONE_API_BASE_URL`, `SYSTEMONE_MODEL` (default `jev-latest`) and the
    /// optional `SYSTEMONE_API_KEY` bearer from the environment.
    pub fn from_env() -> Result<Self, ProviderError> {
        let base_url = std::env::var("SYSTEMONE_API_BASE_URL")
            .map_err(|_| ProviderError::Transport("SYSTEMONE_API_BASE_URL is not set".into()))?;
        let model = std::env::var("SYSTEMONE_MODEL").unwrap_or_else(|_| "jev-latest".to_string());
        let mut client = Self::new(base_url, model);
        if let Ok(key) = std::env::var("SYSTEMONE_API_KEY")
            && !key.is_empty()
        {
            client.api_key = Some(key);
        }
        Ok(client)
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    fn endpoint(&self) -> String {
        format!("{}/v1/systemone", self.base_url)
    }

    /// Builds the System-One request body: `{ model, state, questions }`.
    /// `state` carries only the room snapshot and the utterance — the whole
    /// input budget of the single pass (token minimization).
    fn build_request(&self, input: &InferenceInput<'_>) -> Value {
        let state = json!({
            "room_state": input.room_state,
            "utterance": input.utterance,
        });

        json!({
            "model": self.model,
            "state": state,
            "questions": questions(),
        })
    }

    async fn post(&self, body: Value) -> Result<(Value, Duration), ProviderError> {
        let started = Instant::now();
        let mut request = self
            .http
            .post(self.endpoint())
            .header("Content-Type", "application/json")
            .json(&body);
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }

        let response = request.send().await.map_err(map_reqwest_error)?;
        let status = response.status();
        let text = response.text().await.map_err(map_reqwest_error)?;
        if !status.is_success() {
            return Err(ProviderError::Status {
                status: status.as_u16(),
                body: text.chars().take(512).collect(),
            });
        }

        let elapsed = started.elapsed();
        let value: Value =
            serde_json::from_str(&text).map_err(|e| ProviderError::Malformed(e.to_string()))?;
        let latency = server_latency(&value).unwrap_or(elapsed);
        Ok((value, latency))
    }
}

impl InferencePort for SystemOneClient {
    async fn single_pass(
        &self,
        input: &InferenceInput<'_>,
    ) -> Result<InferenceOutcome, ProviderError> {
        let (response, latency) = self.post(self.build_request(input)).await?;
        decode_outcome(&response, latency)
    }
}

fn timeout_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(timeout)
        .timeout(timeout)
        .build()
        .expect("valid reqwest client")
}

fn map_reqwest_error(err: reqwest::Error) -> ProviderError {
    if err.is_timeout() {
        ProviderError::Timeout(err.to_string())
    } else {
        ProviderError::Transport(err.to_string())
    }
}

/// Prefers a server-reported latency (`elapsed_ms` / `latency_ms` fields)
/// over wall-clock time.
fn server_latency(response: &Value) -> Option<Duration> {
    for key in ["elapsed_ms", "latency_ms", "server_latency_ms"] {
        if let Some(ms) = response.get(key).and_then(Value::as_f64)
            && ms.is_finite()
            && ms >= 0.0
        {
            return Some(Duration::from_secs_f64(ms / 1000.0));
        }
    }
    None
}

// --- Question catalog -------------------------------------------------------

/// The 12 typed questions of one pass (limit: 16): one group per room
/// object — action choice with a `null` = no-change option plus conditional
/// absolute targets — and two global flags.
fn questions() -> Value {
    json!({
        "emergency_stop": {
            "type": "noul",
            "instructions": "Trigger safety interlock, shut down insufflation/irrigation?",
        },
        "requires_sterile_confirm": {
            "type": "noul",
            "instructions": "Needs sterile confirmation (pressure, tilt, emergency)?",
        },
        "light_action": {
            "type": "choice",
            "instructions": "Action for the surgical light",
            "criteria": {
                "null": "No change",
                "IncreaseBrightness": "Set brightness to the brightness target",
                "DecreaseBrightness": "Set brightness to the brightness target",
                "SetLightMode": "Switch field mode to the mode target",
            },
        },
        "brightness_target": {
            "type": "score",
            "instructions": "Target light brightness in percent",
            "criteria": ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "75%", "90%", "100%"],
        },
        "light_mode_target": {
            "type": "choice",
            "instructions": "Target light field mode",
            "criteria": {
                "Normal": "Standard white illumination",
                "CavityFocus": "Focused high-intensity cavity light",
                "AmbientRed": "Red laparoscopy ambient background light",
            },
        },
        "camera_action": {
            "type": "choice",
            "instructions": "Action for the endoscope camera",
            "criteria": {
                "null": "No change",
                "ZoomIn": "Zoom in to the zoom target",
                "ZoomOut": "Zoom out to the zoom target",
                "ToggleIrrigation": "Toggle endoscope irrigation",
            },
        },
        "zoom_target": {
            "type": "score",
            "instructions": "Target endoscope zoom level",
            "criteria": ["level 1", "level 2", "level 3", "level 4", "level 5"],
        },
        "insufflator_action": {
            "type": "choice",
            "instructions": "Action for the CO2 insufflator",
            "criteria": {
                "null": "No change",
                "AdjustPressure": "Set target pressure to the pressure target",
                "ToggleInsufflation": "Toggle CO2 insufflation on or off",
            },
        },
        "pressure_target": {
            "type": "score",
            "instructions": "Target insufflator pressure in mmHg (typical 12-15, hard cap 25)",
            "criteria": [
                "0 mmHg", "5 mmHg", "8 mmHg", "10 mmHg", "12 mmHg",
                "14 mmHg", "16 mmHg", "19 mmHg", "22 mmHg", "25 mmHg",
            ],
        },
        "table_action": {
            "type": "choice",
            "instructions": "Action for the operating table",
            "criteria": {
                "null": "No change",
                "TiltTable": "Set tilt to the tilt target",
                "SetTableHeight": "Set height to the height target",
            },
        },
        "tilt_target": {
            "type": "score",
            "instructions": "Target table tilt in degrees (-15 Trendelenburg, 0 flat, +15 anti-Trendelenburg)",
            "criteria": [
                "-15 deg", "-12 deg", "-8 deg", "-4 deg", "0 deg",
                "+4 deg", "+8 deg", "+12 deg", "+15 deg",
            ],
        },
        "height_target": {
            "type": "score",
            "instructions": "Target table height in cm",
            "criteria": ["70 cm", "80 cm", "90 cm", "100 cm", "110 cm", "120 cm", "130 cm"],
        },
    })
}

// --- Answer decoding --------------------------------------------------------

#[derive(Deserialize)]
struct SystemOneResponse {
    answers: BTreeMap<String, Value>,
}

/// Decodes the System-One answers into one [`UtteranceDecision`] carrying
/// all device decisions of the pass.
fn decode_outcome(response: &Value, latency: Duration) -> Result<InferenceOutcome, ProviderError> {
    let parsed: SystemOneResponse = serde_json::from_value(response.clone())
        .map_err(|e| ProviderError::Malformed(format!("missing answers: {e}")))?;
    let answers = &parsed.answers;

    let (emergency_stop, _) = decode_noul(answers, "emergency_stop")?;
    let (requires_sterile_confirm, _) = decode_noul(answers, "requires_sterile_confirm")?;

    let devices = vec![
        decode_device(answers, TargetDevice::SurgicalLight, "light_action")?,
        decode_device(answers, TargetDevice::EndoscopeCamera, "camera_action")?,
        decode_device(answers, TargetDevice::Insufflator, "insufflator_action")?,
        decode_device(answers, TargetDevice::OperatingTable, "table_action")?,
    ];

    Ok(InferenceOutcome {
        decision: UtteranceDecision {
            devices,
            emergency_stop,
            requires_sterile_confirm,
        },
        latency,
    })
}

/// Decodes one object group: the action choice (`null` = no change) plus
/// the conditional absolute target. An absent target answer degrades to
/// `absolute: None` (the executor then applies no change); malformed
/// content stays a hard error.
fn decode_device(
    answers: &BTreeMap<String, Value>,
    device: TargetDevice,
    question: &str,
) -> Result<DeviceDecision, ProviderError> {
    let (choice, confidence) = decode_choice(answers, question)?;
    if NO_CHANGE_KEYS.contains(&choice.as_str()) {
        return Ok(DeviceDecision {
            target_device: device,
            action: None,
            absolute: None,
            confidence,
            absolute_confidence: 0.0,
        });
    }
    let action: ActionKind = serde_json::from_value(Value::String(choice))
        .map_err(|e| ProviderError::InconsistentSlots(format!("{question}: {e}")))?;

    let (absolute, absolute_confidence) = match absolute_target_for(action) {
        Some(target) => decode_absolute_optional(answers, target)?,
        None => (None, 0.0),
    };
    Ok(DeviceDecision {
        target_device: device,
        action: Some(action),
        absolute,
        confidence,
        absolute_confidence,
    })
}

/// Noul decode: probability of `true` >= 0.5, confidence from the answer's
/// own `confidence` field or the decision margin.
fn decode_noul(
    answers: &BTreeMap<String, Value>,
    name: &str,
) -> Result<(bool, f32), ProviderError> {
    let answer = answer(answers, name)?;
    let p_true = answer
        .get("noul")
        .and_then(Value::as_f64)
        .ok_or_else(|| malformed(name, "missing noul probability"))?;
    if !(0.0..=1.0).contains(&p_true) {
        return Err(malformed(name, "noul probability out of range"));
    }
    let confidence = answer
        .get("confidence")
        .and_then(Value::as_f64)
        .map(|c| c as f32)
        .unwrap_or_else(|| p_true.max(1.0 - p_true) as f32);
    Ok((p_true >= 0.5, confidence))
}

/// Choice decode: the selected option key must be a known option. An
/// explicit JSON `null` choice decodes to the `"null"` key (no change).
fn decode_choice(
    answers: &BTreeMap<String, Value>,
    name: &str,
) -> Result<(String, f32), ProviderError> {
    let answer = answer(answers, name)?;
    if answer.get("choice").is_some_and(Value::is_null) {
        return Ok(("null".to_string(), 0.0));
    }
    let choice = answer
        .get("choice")
        .and_then(Value::as_str)
        .ok_or_else(|| malformed(name, "missing choice"))?
        .to_string();
    let confidence = answer
        .get("confidence")
        .and_then(Value::as_f64)
        .map(|c| c as f32)
        .or_else(|| {
            answer
                .get("probabilities")
                .and_then(|p| p.get(&choice))
                .and_then(Value::as_f64)
                .map(|p| p as f32)
        })
        .unwrap_or(0.0);
    Ok((choice, confidence))
}

/// Score decode: probability-weighted rubric index (rounded) + confidence.
fn decode_score(
    answers: &BTreeMap<String, Value>,
    name: &str,
) -> Result<(usize, f32), ProviderError> {
    let answer = answer(answers, name)?;
    let probabilities = answer
        .get("probabilities")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed(name, "missing probabilities"))?;

    let mut weighted = 0.0f64;
    let mut count = 0usize;
    for (key, value) in probabilities {
        let index: f64 = key
            .parse()
            .map_err(|_| malformed(name, format!("bad level key {key}")))?;
        weighted += index
            * value
                .as_f64()
                .ok_or_else(|| malformed(name, "non-numeric probability"))?;
        count += 1;
    }
    if count == 0 {
        return Err(malformed(name, "empty probabilities"));
    }
    let index = weighted.round() as usize;
    let confidence = answer
        .get("confidence")
        .and_then(Value::as_f64)
        .map(|c| c as f32)
        .unwrap_or(1.0);
    Ok((index, confidence))
}

/// Optional absolute-target decode: absent answer -> no target (executor
/// degrades the action to no change), present answer -> decoded target.
fn decode_absolute_optional(
    answers: &BTreeMap<String, Value>,
    target: AbsoluteTarget,
) -> Result<(Option<i16>, f32), ProviderError> {
    match target {
        AbsoluteTarget::Brightness => opt_score(answers, "brightness_target", &BRIGHTNESS_LEVELS),
        AbsoluteTarget::Zoom => opt_score(answers, "zoom_target", &ZOOM_LEVELS),
        AbsoluteTarget::Pressure => opt_score(answers, "pressure_target", &PRESSURE_LEVELS),
        AbsoluteTarget::Tilt => opt_score(answers, "tilt_target", &TILT_LEVELS),
        AbsoluteTarget::Height => opt_score(answers, "height_target", &HEIGHT_LEVELS),
        AbsoluteTarget::LightMode => {
            if !answers.contains_key("light_mode_target") {
                return Ok((None, 0.0));
            }
            let (mode, confidence) = decode_choice(answers, "light_mode_target")?;
            let code = LIGHT_MODES
                .iter()
                .find(|(name, _)| *name == mode)
                .map(|(_, code)| *code)
                .ok_or_else(|| {
                    ProviderError::InconsistentSlots(format!("light mode {mode:?} unknown"))
                })?;
            // `set_light_mode` in delta.rs decodes 0/1/2 mode codes.
            Ok((Some(code), confidence))
        }
    }
}

fn opt_score(
    answers: &BTreeMap<String, Value>,
    name: &str,
    levels: &[i16],
) -> Result<(Option<i16>, f32), ProviderError> {
    if !answers.contains_key(name) {
        return Ok((None, 0.0));
    }
    let (index, confidence) = decode_score(answers, name)?;
    Ok((Some(levels[clamp_index(index, levels.len())]), confidence))
}

fn clamp_index(index: usize, len: usize) -> usize {
    index.min(len - 1)
}

fn answer<'a>(
    answers: &'a BTreeMap<String, Value>,
    name: &str,
) -> Result<&'a Value, ProviderError> {
    answers
        .get(name)
        .ok_or_else(|| ProviderError::Malformed(format!("missing answer for question {name}")))
}

fn malformed(name: &str, detail: impl std::fmt::Display) -> ProviderError {
    ProviderError::Malformed(format!("{name}: {detail}"))
}

/// Serializes the question catalog for tests / tooling.
#[doc(hidden)]
pub fn question_catalog_json() -> Value {
    questions()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn question_catalog_is_twelve_questions_under_the_limit() {
        let catalog = questions();
        let object = catalog.as_object().expect("catalog object");
        assert_eq!(object.len(), 12);
        assert!(object.len() <= 16, "system-one question limit is 16");
    }

    #[test]
    fn every_action_group_offers_a_null_no_change_option() {
        let catalog = questions();
        for group in [
            "light_action",
            "camera_action",
            "insufflator_action",
            "table_action",
        ] {
            assert_eq!(catalog[group]["criteria"]["null"], json!("No change"));
        }
    }
}
