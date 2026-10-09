//! Typed System-One / JevK5 HTTP client (`POST {base}/v1/systemone`).
//!
//! One pass maps to one request: the five `ActionDecision` slots become
//! typed questions (`noul` / `choice` / `score`), plus one dedicated
//! absolute-target question per device. Request and answer shapes follow
//! the documented System-One contract; the self-hosted JevK5 server
//! (`jevk5-serve`) speaks the same shape.
//!
//! Latency: high model latency is expected, so this client is pure async
//! reqwest and never blocks.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

use crate::decision::{ActionDecision, ActionKind, StepValue, TargetDevice};
use crate::engine::{
    InferenceInput, InferenceOutcome, InferencePort, ProviderError, SlotConfidences,
};

/// Rubric levels for the relative `step_value` score (index -> delta).
const STEP_LEVELS: [i16; 5] = [-2, -1, 0, 1, 2];
/// Brightness rubric (10 levels, 0-100 %).
const BRIGHTNESS_LEVELS: [i16; 10] = [0, 10, 20, 30, 40, 50, 60, 75, 90, 100];
/// Zoom rubric levels 1-5.
const ZOOM_LEVELS: [i16; 5] = [1, 2, 3, 4, 5];
/// Pressure rubric (10 levels, 0-25 mmHg, typical 12-14 explicit).
const PRESSURE_LEVELS: [i16; 10] = [0, 5, 8, 10, 12, 14, 16, 19, 22, 25];
/// Table tilt rubric (9 levels, -15..+15 deg).
const TILT_LEVELS: [i16; 9] = [-15, -12, -8, -4, 0, 4, 8, 12, 15];
/// Light mode rubric order, mapped to `LightMode` codes 0/1/2.
const LIGHT_MODES: [(&str, i16); 3] = [("Normal", 0), ("CavityFocus", 1), ("AmbientRed", 2)];

/// Which absolute-target question applies to an action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AbsoluteTarget {
    Brightness,
    Zoom,
    Pressure,
    Tilt,
    LightMode,
}

fn absolute_target_for(device: TargetDevice, action: ActionKind) -> Option<AbsoluteTarget> {
    match (device, action) {
        (
            TargetDevice::SurgicalLight,
            ActionKind::IncreaseBrightness | ActionKind::DecreaseBrightness,
        ) => Some(AbsoluteTarget::Brightness),
        (TargetDevice::SurgicalLight, ActionKind::SetLightMode) => Some(AbsoluteTarget::LightMode),
        (TargetDevice::EndoscopeCamera, ActionKind::ZoomIn | ActionKind::ZoomOut) => {
            Some(AbsoluteTarget::Zoom)
        }
        (TargetDevice::Insufflator, ActionKind::AdjustPressure) => Some(AbsoluteTarget::Pressure),
        (TargetDevice::OperatingTable, ActionKind::TiltTable) => Some(AbsoluteTarget::Tilt),
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
            http: reqwest::Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key: None,
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
    /// `state` carries the room snapshot, the utterance, and the action
    /// history of earlier passes. (Retry without history is an accuracy
    /// experiment worth trying later.)
    fn build_request(&self, input: &InferenceInput<'_>) -> Value {
        let history: Vec<Value> = input
            .history
            .iter()
            .map(|report| {
                json!({
                    "target_device": report.target_device,
                    "action_kind": report.action_kind,
                    "detail": report.detail,
                })
            })
            .collect();

        let state = json!({
            "room_state": input.room_state,
            "utterance": input.utterance,
            "history": history,
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

/// The 10 typed questions per pass (limit: 16).
fn questions() -> Value {
    // TODO: evaluate behavior improvements choice instead of noul
    json!({
        "further_action": {
            "type": "noul",
            "instructions": "Is another device action needed after this one to fully satisfy the utterance?",
        },
        "requires_sterile_confirm": {
            "type": "noul",
            "instructions": "Does this action need sterile confirmation for safety (pressure, table tilt, emergency)?",
        },
        "target_device": {
            "type": "choice",
            "instructions": "Which device does the current sub-action target?",
            "criteria": {
                "None": "No device, nothing to do",
                "SurgicalLight": "Ceiling surgical light (brightness, light mode)",
                "EndoscopeCamera": "Endoscope camera (zoom, irrigation)",
                "Insufflator": "CO2 insufflator (target pressure, insufflation)",
                "OperatingTable": "Operating table (tilt)",
            },
        },
        "action_kind": {
            "type": "choice",
            "instructions": "Which operation applies to the target device?",
            "criteria": {
                "Idle": "No operation",
                "IncreaseBrightness": "Raise light brightness",
                "DecreaseBrightness": "Lower light brightness",
                "SetLightMode": "Switch the light field mode",
                "ZoomIn": "Move the endoscope closer / zoom in",
                "ZoomOut": "Pull the endoscope back / zoom out",
                "ToggleIrrigation": "Toggle endoscope irrigation",
                "AdjustPressure": "Set insufflator target pressure",
                "ToggleInsufflation": "Toggle CO2 insufflation on or off",
                "TiltTable": "Tilt the operating table",
                "EmergencyStop": "Trigger the safety interlock, shut down insufflation and irrigation",
            },
        },
        "step_value": {
            "type": "score",
            "instructions": "Relative step size for this action",
            "criteria": [
                "two steps down", "one step down", "no change",
                "one step up", "two steps up",
            ],
        },
        "brightness_target": {
            "type": "score",
            "instructions": "Target light brightness in percent",
            "criteria": ["0%", "10%", "20%", "30%", "40%", "50%", "60%", "75%", "90%", "100%"],
        },
        "zoom_target": {
            "type": "score",
            "instructions": "Target endoscope zoom level",
            "criteria": ["level 1", "level 2", "level 3", "level 4", "level 5"],
        },
        "pressure_target": {
            "type": "score",
            "instructions": "Target insufflator pressure in mmHg (typical 12-15, hard cap 25)",
            "criteria": [
                "0 mmHg", "5 mmHg", "8 mmHg", "10 mmHg", "12 mmHg",
                "14 mmHg", "16 mmHg", "19 mmHg", "22 mmHg", "25 mmHg",
            ],
        },
        "tilt_target": {
            "type": "score",
            "instructions": "Target table tilt in degrees (-15 Trendelenburg, 0 flat, +15 anti-Trendelenburg)",
            "criteria": [
                "-15 deg", "-12 deg", "-8 deg", "-4 deg", "0 deg",
                "+4 deg", "+8 deg", "+12 deg", "+15 deg",
            ],
        },
        "light_mode_target": {
            "type": "choice",
            "instructions": "Target light field mode when SetLightMode applies",
            "criteria": {
                "Normal": "Standard white illumination",
                "CavityFocus": "Focused high-intensity cavity light",
                "AmbientRed": "Red laparoscopy ambient background light",
            },
        },
    })
}

// --- Answer decoding --------------------------------------------------------

#[derive(Deserialize)]
struct SystemOneResponse {
    answers: BTreeMap<String, Value>,
}

/// Decodes the System-One answers into one `ActionDecision` + confidences.
fn decode_outcome(response: &Value, latency: Duration) -> Result<InferenceOutcome, ProviderError> {
    let parsed: SystemOneResponse = serde_json::from_value(response.clone())
        .map_err(|e| ProviderError::Malformed(format!("missing answers: {e}")))?;
    let answers = &parsed.answers;

    let (further_action, c_further) = decode_noul(answers, "further_action")?;
    let (requires_confirm, c_confirm) = decode_noul(answers, "requires_sterile_confirm")?;
    let (device, c_device) = decode_choice(answers, "target_device")?;
    let target_device: TargetDevice = serde_json::from_value(Value::String(device))
        .map_err(|e| ProviderError::InconsistentSlots(format!("target_device: {e}")))?;
    let (action, c_action) = decode_choice(answers, "action_kind")?;
    let action_kind: ActionKind = serde_json::from_value(Value::String(action))
        .map_err(|e| ProviderError::InconsistentSlots(format!("action_kind: {e}")))?;

    let (step_index, c_step) = decode_score(answers, "step_value")?;
    let (step_value, c_absolute) = match absolute_target_for(target_device, action_kind) {
        Some(target) => {
            let (value, confidence) = decode_absolute(answers, target)?;
            (StepValue::AbsoluteValue(value), confidence)
        }
        None => {
            let index = clamp_index(step_index, STEP_LEVELS.len());
            (relative_step(STEP_LEVELS[index]), c_step)
        }
    };

    Ok(InferenceOutcome {
        decision: ActionDecision {
            further_action_needed: further_action,
            target_device,
            action_kind,
            step_value,
            requires_sterile_confirm: requires_confirm,
        },
        slot_confidences: SlotConfidences {
            further_action_needed: c_further,
            requires_sterile_confirm: c_confirm,
            target_device: c_device,
            action_kind: c_action,
            step_value: c_step,
            absolute_target: c_absolute,
        },
        latency,
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

/// Choice decode: the selected option key must be a known option.
fn decode_choice(
    answers: &BTreeMap<String, Value>,
    name: &str,
) -> Result<(String, f32), ProviderError> {
    let answer = answer(answers, name)?;
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

fn decode_absolute(
    answers: &BTreeMap<String, Value>,
    target: AbsoluteTarget,
) -> Result<(i16, f32), ProviderError> {
    match target {
        AbsoluteTarget::Brightness => {
            let (index, confidence) = decode_score(answers, "brightness_target")?;
            Ok((
                BRIGHTNESS_LEVELS[clamp_index(index, BRIGHTNESS_LEVELS.len())],
                confidence,
            ))
        }
        AbsoluteTarget::Zoom => {
            let (index, confidence) = decode_score(answers, "zoom_target")?;
            Ok((
                ZOOM_LEVELS[clamp_index(index, ZOOM_LEVELS.len())],
                confidence,
            ))
        }
        AbsoluteTarget::Pressure => {
            let (index, confidence) = decode_score(answers, "pressure_target")?;
            Ok((
                PRESSURE_LEVELS[clamp_index(index, PRESSURE_LEVELS.len())],
                confidence,
            ))
        }
        AbsoluteTarget::Tilt => {
            let (index, confidence) = decode_score(answers, "tilt_target")?;
            Ok((
                TILT_LEVELS[clamp_index(index, TILT_LEVELS.len())],
                confidence,
            ))
        }
        AbsoluteTarget::LightMode => {
            let (mode, confidence) = decode_choice(answers, "light_mode_target")?;
            let code = LIGHT_MODES
                .iter()
                .find(|(name, _)| *name == mode)
                .map(|(_, code)| *code)
                .ok_or_else(|| {
                    ProviderError::InconsistentSlots(format!("light mode {mode:?} unknown"))
                })?;
            // `set_light_mode` in delta.rs decodes 0/1/2 mode codes.
            Ok((code, confidence))
        }
    }
}

fn relative_step(delta: i16) -> StepValue {
    match delta {
        -2 => StepValue::MinusTwo,
        -1 => StepValue::MinusOne,
        0 => StepValue::Zero,
        1 => StepValue::PlusOne,
        2 => StepValue::PlusTwo,
        _ => StepValue::AbsoluteValue(delta),
    }
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
