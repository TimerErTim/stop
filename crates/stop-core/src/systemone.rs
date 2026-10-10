//! Typed System-One / JevK5 HTTP client (`POST {base}/v1/systemone`).
//!
//! One pass maps to one request and returns ALL decisions at once: one
//! question per object setting (brightness, field mode, zoom, irrigation,
//! pressure, insufflation, tilt, height) plus two global flags
//! (`emergency_stop`, `requires_sterile_confirm`). Value settings are
//! `choice` questions offering the numbers uttered in the text as
//! increase/decrease/set options; toggles are `noul` questions. Request and
//! answer shapes follow the documented System-One contract; the self-hosted
//! JevK5 server (`jevk5-serve`) speaks the same shape.
//!
//! Latency: high model latency is expected, so this client is pure async
//! reqwest, never blocks, and enforces a hard request timeout.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Map, Value, json};
use text2num::{Language, Token, find_numbers};

use crate::decision::{
    ActionKind, CameraDecision, InsufflatorDecision, LightDecision, TableDecision,
    UtteranceDecision, ValueChange,
};
use crate::engine::{InferenceInput, InferenceOutcome, InferencePort, ProviderError};
use crate::executor::MIN_ACTION_CONFIDENCE;
use crate::state::LightMode;

/// Hard request timeout (1 minute): a hung JevK5 must not stall the pipeline.
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Choice keys that mean "no change" for a setting.
const NO_CHANGE_KEYS: [&str; 4] = ["null", "None", "NoChange", "Idle"];

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
    /// input budget of the single pass (token minimization). The uttered
    /// numbers are offered to the model as per-setting choice options.
    fn build_request(&self, input: &InferenceInput<'_>) -> Value {
        let state = json!({
            "room_state": input.room_state,
            "utterance": input.utterance,
        });

        json!({
            "model": self.model,
            "state": state,
            "questions": questions(&extract_numbers(input.utterance)),
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

// --- Uttered number extraction ---------------------------------------------

/// One token of the utterance, word or separator, for the `text2num` scanner.
struct UtteranceToken {
    text: String,
    lower: String,
    is_word: bool,
}

impl UtteranceToken {
    fn word(text: &str) -> Self {
        Self {
            text: text.to_string(),
            lower: text.to_lowercase(),
            is_word: true,
        }
    }

    fn separator(text: &str) -> Self {
        Self {
            text: text.to_string(),
            lower: text.to_lowercase(),
            is_word: false,
        }
    }
}

impl Token for &UtteranceToken {
    fn text(&self) -> Cow<'_, str> {
        self.text.as_str().into()
    }

    fn text_lowercase(&self) -> Cow<'_, str> {
        self.lower.as_str().into()
    }

    fn not_a_number_part(&self) -> bool {
        !self.is_word
    }
}

/// Splits the utterance into word tokens (alphanumeric, `-`, `'`) and
/// separator runs, so punctuation between two numbers keeps them apart
/// (`"one, two"` -> `1`, `2`, never `12`).
fn tokenize_utterance(utterance: &str) -> Vec<UtteranceToken> {
    let is_word_char = |c: char| c.is_alphanumeric() || c == '-' || c == '\'';
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut current_is_word = false;
    for c in utterance.chars() {
        let is_word = is_word_char(c);
        if current.is_empty() {
            current.push(c);
            current_is_word = is_word;
        } else if is_word == current_is_word {
            current.push(c);
        } else {
            tokens.push(if current_is_word {
                UtteranceToken::word(&current)
            } else {
                UtteranceToken::separator(&current)
            });
            current.clear();
            current.push(c);
            current_is_word = is_word;
        }
    }
    if !current.is_empty() {
        tokens.push(if current_is_word {
            UtteranceToken::word(&current)
        } else {
            UtteranceToken::separator(&current)
        });
    }
    tokens
}

/// Extracts every number uttered in the text: spelled numbers via
/// [`find_numbers`] (grouping compounds like "one hundred five") plus
/// as-is digit runs like `68`. Ordinals, non-integers ("3.5") and values
/// outside `i16` are dropped; the result is deduplicated in order of
/// appearance.
pub fn extract_numbers(utterance: &str) -> Vec<i16> {
    let tokens = tokenize_utterance(utterance);
    let lang = Language::english();

    let mut numbers = Vec::new();
    let mut seen = HashSet::new();
    for occurrence in find_numbers(tokens.iter(), &lang, 0.0) {
        if !occurrence.is_ordinal {
            push_number(occurrence.value, &mut numbers, &mut seen);
        }
    }
    for value in as_is_numbers(utterance) {
        push_number(value, &mut numbers, &mut seen);
    }
    numbers
}

/// Scans for standalone digit runs (`68`, `12.5`); digits embedded in a word
/// (`co2`) are ignored.
fn as_is_numbers(utterance: &str) -> Vec<f64> {
    let bytes = utterance.as_bytes();
    let is_alnum = |b: u8| b.is_ascii_alphanumeric();
    let mut numbers = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() || (i > 0 && is_alnum(bytes[i - 1])) {
            i += 1;
            continue;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i + 1 < bytes.len() && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < bytes.len() && is_alnum(bytes[i]) {
            continue;
        }
        if let Ok(value) = utterance[start..i].parse::<f64>() {
            numbers.push(value);
        }
    }
    numbers
}

fn push_number(value: f64, numbers: &mut Vec<i16>, seen: &mut HashSet<i16>) {
    if !value.is_finite() || value.fract() != 0.0 {
        return;
    }
    if value < f64::from(i16::MIN) || value > f64::from(i16::MAX) {
        return;
    }
    let value = value as i16;
    if seen.insert(value) {
        numbers.push(value);
    }
}

// --- Question catalog -------------------------------------------------------

/// The 10 typed questions of one pass (limit: 16): one question per object
/// setting — `choice` for value settings (numbers offered as
/// increase/decrease/set options), `noul` for toggles — plus two global
/// flags. Every choice carries a `null` = leave-as-is option.
fn questions(numbers: &[i16]) -> Value {
    json!({
        "emergency_stop": {
            "type": "noul",
            "instructions": "Trigger safety interlock, shut down insufflation/irrigation?",
        },
        "requires_sterile_confirm": {
            "type": "noul",
            "instructions": "Needs sterile confirmation (pressure, tilt, emergency)?",
        },
        "light_brightness": {
            "type": "choice",
            "instructions": "Brightness change for the surgical light",
            "criteria": brightness_criteria(numbers),
        },
        "light_mode": {
            "type": "choice",
            "instructions": "Field mode for the surgical light",
            "criteria": light_mode_criteria(),
        },
        "camera_zoom": {
            "type": "choice",
            "instructions": "Zoom change for the endoscope camera",
            "criteria": zoom_criteria(numbers),
        },
        "camera_irrigation": {
            "type": "noul",
            "instructions": "Toggle endoscope irrigation?",
        },
        "insufflator_pressure": {
            "type": "choice",
            "instructions": "Target pressure change for the CO2 insufflator (mmHg, hard cap 25)",
            "criteria": pressure_criteria(numbers),
        },
        "insufflator_active": {
            "type": "noul",
            "instructions": "Toggle CO2 insufflation on or off?",
        },
        "table_tilt": {
            "type": "choice",
            "instructions": "Tilt change for the operating table (-15 Trendelenburg, +15 anti)",
            "criteria": tilt_criteria(numbers),
        },
        "table_height": {
            "type": "choice",
            "instructions": "Height change for the operating table (cm)",
            "criteria": height_criteria(numbers),
        },
    })
}

fn brightness_criteria(numbers: &[i16]) -> Value {
    let mut criteria = value_criteria_base(numbers, "Brightness", "brightness", "%");
    criteria.insert(
        "IncreaseBrightness".to_string(),
        json!("Increase brightness by one step"),
    );
    criteria.insert(
        "DecreaseBrightness".to_string(),
        json!("Decrease brightness by one step"),
    );
    Value::Object(criteria)
}

fn zoom_criteria(numbers: &[i16]) -> Value {
    let mut criteria = Map::new();
    criteria.insert("null".to_string(), json!("No change"));
    for &n in numbers {
        criteria.insert(format!("ZoomIn:{n}"), json!(format!("Zoom in by {n}")));
        criteria.insert(format!("ZoomOut:{n}"), json!(format!("Zoom out by {n}")));
        criteria.insert(
            format!("SetZoom:{n}"),
            json!(format!("Set zoom to level {n}")),
        );
    }
    criteria.insert("ZoomIn".to_string(), json!("Zoom in by one level"));
    criteria.insert("ZoomOut".to_string(), json!("Zoom out by one level"));
    Value::Object(criteria)
}

fn pressure_criteria(numbers: &[i16]) -> Value {
    let mut criteria = value_criteria_base(numbers, "Pressure", "target pressure", "mmHg");
    criteria.insert(
        "IncreasePressure".to_string(),
        json!("Increase target pressure by one step"),
    );
    criteria.insert(
        "DecreasePressure".to_string(),
        json!("Decrease target pressure by one step"),
    );
    Value::Object(criteria)
}

fn tilt_criteria(numbers: &[i16]) -> Value {
    let mut criteria = value_criteria_base(numbers, "Tilt", "table tilt", "degrees");
    criteria.insert(
        "IncreaseTilt".to_string(),
        json!("Increase table tilt by one step"),
    );
    criteria.insert(
        "DecreaseTilt".to_string(),
        json!("Decrease table tilt by one step"),
    );
    Value::Object(criteria)
}

fn height_criteria(numbers: &[i16]) -> Value {
    let mut criteria = value_criteria_base(numbers, "Height", "table height", "cm");
    criteria.insert(
        "IncreaseHeight".to_string(),
        json!("Raise the table by one step"),
    );
    criteria.insert(
        "DecreaseHeight".to_string(),
        json!("Lower the table by one step"),
    );
    Value::Object(criteria)
}

/// Shared `null` + per-number `Increase<prefix>:N` / `Decrease<prefix>:N` /
/// `Set<prefix>:N` options for a value setting.
fn value_criteria_base(
    numbers: &[i16],
    prefix: &str,
    label: &str,
    unit: &str,
) -> Map<String, Value> {
    let mut criteria = Map::new();
    criteria.insert("null".to_string(), json!("No change"));
    for &n in numbers {
        criteria.insert(
            format!("Increase{prefix}:{n}"),
            json!(format!("Increase {label} by {n} {unit}")),
        );
        criteria.insert(
            format!("Decrease{prefix}:{n}"),
            json!(format!("Decrease {label} by {n} {unit}")),
        );
        criteria.insert(
            format!("Set{prefix}:{n}"),
            json!(format!("Set {label} to {n} {unit}")),
        );
    }
    criteria
}

fn light_mode_criteria() -> Value {
    json!({
        "null": "No change",
        "Normal": "Standard white illumination",
        "CavityFocus": "Focused high-intensity cavity light",
        "AmbientRed": "Red laparoscopy ambient background light",
    })
}

// --- Answer decoding --------------------------------------------------------

#[derive(Deserialize)]
struct SystemOneResponse {
    answers: BTreeMap<String, Value>,
}

/// Decodes the System-One answers into one [`UtteranceDecision`] carrying an
/// explicit decision per room object.
fn decode_outcome(response: &Value, latency: Duration) -> Result<InferenceOutcome, ProviderError> {
    let parsed: SystemOneResponse = serde_json::from_value(response.clone())
        .map_err(|e| ProviderError::Malformed(format!("missing answers: {e}")))?;
    let answers = &parsed.answers;

    let (emergency_stop, _) = decode_noul(answers, "emergency_stop")?;
    let (requires_sterile_confirm, _) = decode_noul(answers, "requires_sterile_confirm")?;

    let light = LightDecision {
        brightness: decode_value(
            answers,
            "light_brightness",
            ActionKind::SetBrightness,
            ActionKind::IncreaseBrightness,
            ActionKind::DecreaseBrightness,
        )?,
        field_mode: decode_light_mode(answers)?,
    };
    let camera = CameraDecision {
        zoom: decode_value(
            answers,
            "camera_zoom",
            ActionKind::SetZoom,
            ActionKind::ZoomIn,
            ActionKind::ZoomOut,
        )?,
        toggle_irrigation: decode_noul(answers, "camera_irrigation")?.0,
    };
    let insufflator = InsufflatorDecision {
        pressure: decode_value(
            answers,
            "insufflator_pressure",
            ActionKind::SetPressure,
            ActionKind::IncreasePressure,
            ActionKind::DecreasePressure,
        )?,
        toggle_insufflation: decode_noul(answers, "insufflator_active")?.0,
    };
    let table = TableDecision {
        tilt: decode_value(
            answers,
            "table_tilt",
            ActionKind::SetTilt,
            ActionKind::IncreaseTilt,
            ActionKind::DecreaseTilt,
        )?,
        height: decode_value(
            answers,
            "table_height",
            ActionKind::SetHeight,
            ActionKind::IncreaseHeight,
            ActionKind::DecreaseHeight,
        )?,
    };

    Ok(InferenceOutcome {
        decision: UtteranceDecision {
            light,
            camera,
            insufflator,
            table,
            emergency_stop,
            requires_sterile_confirm,
        },
        latency,
    })
}

/// Decodes a value-setting `choice`: `null` or a low-confidence answer means
/// leave-as-is; otherwise the choice key (`"<Action>"` or `"<Action>:<N>"`)
/// must name one of the setting's three actions.
fn decode_value(
    answers: &BTreeMap<String, Value>,
    question: &str,
    absolute: ActionKind,
    increase: ActionKind,
    decrease: ActionKind,
) -> Result<Option<ValueChange>, ProviderError> {
    let (choice, confidence) = decode_choice(answers, question)?;
    if NO_CHANGE_KEYS.contains(&choice.as_str()) || confidence < MIN_ACTION_CONFIDENCE {
        return Ok(None);
    }
    let (name, operand) = split_choice(&choice);
    let action = parse_action(name, question)?;
    let value = parse_operand(operand, question)?;

    if action == absolute {
        let value =
            value.ok_or_else(|| inconsistent(question, format!("{name} without a value")))?;
        Ok(Some(ValueChange::Absolute(value)))
    } else if action == increase {
        Ok(Some(ValueChange::Increase(value.unwrap_or(1))))
    } else if action == decrease {
        Ok(Some(ValueChange::Decrease(value.unwrap_or(1))))
    } else {
        Err(inconsistent(
            question,
            format!("unexpected action {name:?}"),
        ))
    }
}

/// Decodes the field-mode `choice`: `null` / low confidence = leave as is.
fn decode_light_mode(
    answers: &BTreeMap<String, Value>,
) -> Result<Option<LightMode>, ProviderError> {
    let (choice, confidence) = decode_choice(answers, "light_mode")?;
    if NO_CHANGE_KEYS.contains(&choice.as_str()) || confidence < MIN_ACTION_CONFIDENCE {
        return Ok(None);
    }
    let mode = match choice.as_str() {
        "Normal" => LightMode::Normal,
        "CavityFocus" => LightMode::CavityFocus,
        "AmbientRed" => LightMode::AmbientRed,
        other => {
            return Err(inconsistent(
                "light_mode",
                format!("unknown mode {other:?}"),
            ));
        }
    };
    Ok(Some(mode))
}

fn split_choice(choice: &str) -> (&str, Option<&str>) {
    match choice.split_once(':') {
        Some((name, operand)) => (name, Some(operand)),
        None => (choice, None),
    }
}

fn parse_action(name: &str, question: &str) -> Result<ActionKind, ProviderError> {
    serde_json::from_value(Value::String(name.to_string()))
        .map_err(|e| inconsistent(question, e.to_string()))
}

fn parse_operand(operand: Option<&str>, question: &str) -> Result<Option<i16>, ProviderError> {
    match operand {
        Some(operand) => operand
            .parse::<i16>()
            .map(Some)
            .map_err(|_| inconsistent(question, format!("bad operand {operand:?}"))),
        None => Ok(None),
    }
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

fn inconsistent(question: &str, detail: impl std::fmt::Display) -> ProviderError {
    ProviderError::InconsistentSlots(format!("{question}: {detail}"))
}

/// Serializes the question catalog for tests / tooling.
#[doc(hidden)]
pub fn question_catalog_json(numbers: &[i16]) -> Value {
    questions(numbers)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- extraction ---

    #[test]
    fn extracts_spelled_numbers_from_noise_words() {
        assert_eq!(
            extract_numbers("bump the light up to about eighty-five percent"),
            vec![85]
        );
        assert_eq!(
            extract_numbers("bring the scope to zoom level three"),
            vec![3]
        );
    }

    #[test]
    fn extracts_as_is_digit_numbers() {
        assert_eq!(extract_numbers("set the pressure to 68"), vec![68]);
        assert_eq!(extract_numbers("raise it to 12.5"), Vec::<i16>::new());
    }

    #[test]
    fn merges_spelled_compound_numbers() {
        assert_eq!(extract_numbers("one hundred five"), vec![105]);
        assert_eq!(extract_numbers("eighty-five"), vec![85]);
    }

    #[test]
    fn keeps_adjacent_numbers_separate_across_punctuation() {
        assert_eq!(
            extract_numbers("set levels one, two, and three"),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn deduplicates_by_value_in_order_of_appearance() {
        assert_eq!(extract_numbers("fifteen and zoom to 15"), vec![15]);
        assert_eq!(extract_numbers("fifteen and zoom to 3"), vec![15, 3]);
    }

    #[test]
    fn drops_non_integers_and_ignores_digits_inside_words() {
        assert_eq!(extract_numbers("three point five"), Vec::<i16>::new());
        assert_eq!(extract_numbers("check the co2 line"), Vec::<i16>::new());
    }

    #[test]
    fn noise_utterance_yields_no_numbers() {
        assert_eq!(
            extract_numbers("how was your morning, you look tired"),
            Vec::<i16>::new()
        );
    }

    // --- question catalog ---

    #[test]
    fn question_catalog_is_ten_questions_under_the_limit() {
        let catalog = questions(&[]);
        let object = catalog.as_object().expect("catalog object");
        assert_eq!(object.len(), 10);
        assert!(object.len() <= 16, "system-one question limit is 16");
    }

    #[test]
    fn brightness_and_mode_are_separate_questions() {
        let catalog = questions(&[]);
        assert_eq!(catalog["light_brightness"]["type"], json!("choice"));
        assert_eq!(catalog["light_mode"]["type"], json!("choice"));
        // Toggles are noul, not choice.
        assert_eq!(catalog["camera_irrigation"]["type"], json!("noul"));
        assert_eq!(catalog["insufflator_active"]["type"], json!("noul"));
    }

    #[test]
    fn every_choice_offers_a_null_leave_as_is_option() {
        let catalog = questions(&[]);
        for question in [
            "light_brightness",
            "light_mode",
            "camera_zoom",
            "insufflator_pressure",
            "table_tilt",
            "table_height",
        ] {
            assert_eq!(
                catalog[question]["criteria"]["null"],
                json!("No change"),
                "{question} lacks null option"
            );
        }
    }

    #[test]
    fn uttered_numbers_become_per_setting_choice_options() {
        let catalog = questions(&[8, 12]);
        let brightness = &catalog["light_brightness"]["criteria"];
        for key in [
            "IncreaseBrightness:8",
            "DecreaseBrightness:8",
            "SetBrightness:8",
            "IncreaseBrightness:12",
            "SetBrightness:12",
        ] {
            assert!(brightness.get(key).is_some(), "missing option {key}");
        }
        // Tilt and height share the numbers independently.
        let tilt = &catalog["table_tilt"]["criteria"];
        assert!(tilt.get("SetTilt:12").is_some());
        let height = &catalog["table_height"]["criteria"];
        assert!(height.get("SetHeight:12").is_some());
        // Generic fallbacks stay available when no number is uttered.
        let plain = questions(&[]);
        assert!(
            plain["light_brightness"]["criteria"]
                .get("IncreaseBrightness")
                .is_some()
        );
        assert!(
            plain["light_brightness"]["criteria"]
                .get("SetBrightness:8")
                .is_none()
        );
    }

    #[test]
    fn score_target_questions_are_gone() {
        let catalog = questions(&[8]);
        for name in [
            "brightness_target",
            "zoom_target",
            "pressure_target",
            "tilt_target",
            "height_target",
            "light_mode_target",
        ] {
            assert!(catalog.get(name).is_none(), "stale question {name}");
        }
    }

    // --- choice decoding ---

    #[test]
    fn decodes_value_changes() {
        assert_eq!(
            decode_value(
                &answers_with("light_brightness", "SetBrightness:60"),
                "light_brightness",
                ActionKind::SetBrightness,
                ActionKind::IncreaseBrightness,
                ActionKind::DecreaseBrightness,
            )
            .expect("decode"),
            Some(ValueChange::Absolute(60))
        );
        assert_eq!(
            decode_value(
                &answers_with("light_brightness", "IncreaseBrightness:8"),
                "light_brightness",
                ActionKind::SetBrightness,
                ActionKind::IncreaseBrightness,
                ActionKind::DecreaseBrightness,
            )
            .expect("decode"),
            Some(ValueChange::Increase(8))
        );
        assert_eq!(
            decode_value(
                &answers_with("table_height", "DecreaseHeight"),
                "table_height",
                ActionKind::SetHeight,
                ActionKind::IncreaseHeight,
                ActionKind::DecreaseHeight,
            )
            .expect("decode"),
            Some(ValueChange::Decrease(1))
        );
    }

    #[test]
    fn low_confidence_value_is_leave_as_is() {
        let mut answers = BTreeMap::new();
        answers.insert(
            "light_brightness".to_string(),
            json!({ "choice": "SetBrightness:60", "confidence": 0.2 }),
        );
        assert_eq!(
            decode_value(
                &answers,
                "light_brightness",
                ActionKind::SetBrightness,
                ActionKind::IncreaseBrightness,
                ActionKind::DecreaseBrightness,
            )
            .expect("decode"),
            None
        );
    }

    #[test]
    fn rejects_malformed_value_choices() {
        for (choice, expect_ok) in [
            ("SetBrightness", false),
            ("IncreaseBrightness:abc", false),
            ("Teleporter", false),
            ("IncreaseBrightness", true),
            ("SetBrightness:60", true),
        ] {
            let result = decode_value(
                &answers_with("light_brightness", choice),
                "light_brightness",
                ActionKind::SetBrightness,
                ActionKind::IncreaseBrightness,
                ActionKind::DecreaseBrightness,
            );
            assert_eq!(result.is_ok(), expect_ok, "choice {choice}");
        }
    }

    #[test]
    fn decodes_light_mode() {
        assert_eq!(
            decode_light_mode(&answers_with("light_mode", "CavityFocus")).expect("decode"),
            Some(LightMode::CavityFocus)
        );
        assert_eq!(
            decode_light_mode(&answers_with("light_mode", "null")).expect("decode"),
            None
        );
        assert!(decode_light_mode(&answers_with("light_mode", "Purple")).is_err());
    }

    fn answers_with(question: &str, choice: &str) -> BTreeMap<String, Value> {
        let mut answers = BTreeMap::new();
        answers.insert(
            question.to_string(),
            json!({ "choice": choice, "confidence": 0.9 }),
        );
        answers
    }
}
