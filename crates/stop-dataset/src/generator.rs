//! Single-call scenario generator (spec `docs/INSTRUCTIONS.md` section 4.3).
//!
//! One request per case: given the scenario, a random utterance type sequence
//! and the initial room state, the model returns an STT-like transcript
//! micro-segment (commands mixed with handlungsneutrale filler/noise
//! utterances) plus the desired room state after each utterance, chained on
//! the previous state. Filler talk keeps the state, self-corrections revert
//! to the corrected state.

use rand::Rng;
use rand::seq::SliceRandom;
use serde::Deserialize;
use serde_json::Value;
use stop_core::RoomState;

use crate::error::DatasetError;
use crate::openrouter::OpenRouterClient;
use crate::schema::{DatasetCase, HistoryEntry};

// --- Utterance types --------------------------------------------------------

/// Planned type of one utterance slot in a case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UtteranceType {
    /// Action-bearing device command (or self-correction).
    Command,
    /// Handlungsneutrales filler: smalltalk, team comments, off-device talk.
    Noise,
}

/// Generator knobs.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratorConfig {
    /// Total utterances per case.
    pub utterances_per_case: usize,
    /// Share of filler/noise utterances (0.0-1.0); 0.0 means all commands.
    pub noise_ratio: f64,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            utterances_per_case: 16,
            noise_ratio: 0.5,
        }
    }
}

/// Builds the utterance type sequence for one case: exact counts after
/// rounding the noise ratio, then a random shuffle.
pub fn build_utterance_type_sequence(
    config: &GeneratorConfig,
    rng: &mut impl Rng,
) -> Vec<UtteranceType> {
    let total = config.utterances_per_case;
    let ratio = config.noise_ratio.clamp(0.0, 1.0);
    let noise = ((total as f64) * ratio).round() as usize;

    let mut types = Vec::with_capacity(total);
    types.extend(std::iter::repeat_n(UtteranceType::Noise, noise));
    types.extend(std::iter::repeat_n(UtteranceType::Command, total - noise));
    types.shuffle(rng);
    types
}

// --- Generator --------------------------------------------------------------

/// Scenario generator over the OpenRouter JSON completion client.
pub struct Generator {
    client: OpenRouterClient,
    config: GeneratorConfig,
}

impl Generator {
    pub fn new(client: OpenRouterClient, config: GeneratorConfig) -> Self {
        Self { client, config }
    }

    pub fn config(&self) -> &GeneratorConfig {
        &self.config
    }

    /// Generates one complete scenario in a single model call: transcript
    /// draft plus per-utterance room-state prediction chained on the
    /// previous state. Draws the prompt variation seed from `rng`.
    pub async fn generate_case(
        &self,
        rng: &mut impl Rng,
        id: &str,
        scenario: &str,
    ) -> Result<DatasetCase, DatasetError> {
        let types = build_utterance_type_sequence(&self.config, rng);
        let variation_seed = rng.next_u32();
        self.generate_case_with_types(&types, id, scenario, variation_seed)
            .await
    }

    /// Same as [`Generator::generate_case`] with an explicit utterance type
    /// sequence and variation seed (concurrent runners pre-plan both from the
    /// per-case RNG to keep the noise placement and phrasing reproducible).
    pub async fn generate_case_with_types(
        &self,
        types: &[UtteranceType],
        id: &str,
        scenario: &str,
        variation_seed: u32,
    ) -> Result<DatasetCase, DatasetError> {
        let (utterances, states) = self.request_case(scenario, types, variation_seed).await?;

        let mut history = Vec::with_capacity(utterances.len());
        for (utterance, mut state) in utterances.into_iter().zip(states) {
            normalize_state(&mut state);
            history.push(HistoryEntry {
                raw_utterance: utterance,
                expected_output_state: state,
            });
        }
        Ok(DatasetCase {
            id: id.to_string(),
            scenario: scenario.to_string(),
            model: self.client.model().to_string(),
            initial_state: RoomState::default(),
            history,
        })
    }

    /// One call producing both the transcript and the chained expected
    /// states (fewer roundtrips than two separate calls).
    async fn request_case(
        &self,
        scenario: &str,
        types: &[UtteranceType],
        variation_seed: u32,
    ) -> Result<(Vec<String>, Vec<RoomState>), DatasetError> {
        let plan = types
            .iter()
            .enumerate()
            .map(|(i, utterance_type)| {
                format!(
                    "{}: {}",
                    i + 1,
                    match utterance_type {
                        UtteranceType::Command => "device command",
                        UtteranceType::Noise => "filler / smalltalk, no device action",
                    }
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let initial_state = serde_json::to_value(RoomState::default())
            .map_err(|e| DatasetError::Malformed(e.to_string()))?;
        let total = types.len();

        let system = build_system_prompt();
        let user = build_user_prompt(scenario, &plan, &initial_state, variation_seed, total);

        let value = self.client.complete_json(system, &user).await?;
        let utterances: Vec<TranscriptUtterance> = parse_list(&value, "utterances")?;
        let states: Vec<StatePrediction> = parse_list(&value, "states")?;

        if utterances.len() != total {
            return Err(DatasetError::Malformed(format!(
                "got {} utterances, expected {total}",
                utterances.len()
            )));
        }
        if states.len() != total {
            return Err(DatasetError::Malformed(format!(
                "got {} states, expected {total}",
                states.len()
            )));
        }
        for (position, state) in states.iter().enumerate() {
            if state.index != position + 1 {
                return Err(DatasetError::Malformed(format!(
                    "state index {} out of order (expected {})",
                    state.index,
                    position + 1
                )));
            }
        }
        Ok((
            utterances.into_iter().map(|u| u.text).collect(),
            states.into_iter().map(|s| s.room_state).collect(),
        ))
    }
}

/// Clamps a predicted state into the physical safety envelope via the state
/// setters (brightness 0-100, zoom 1-5, pressure <= 25 mmHg, tilt +/-15).
pub fn normalize_state(state: &mut RoomState) {
    let brightness = i16::from(state.lighting.primary_intensity_pct);
    state.lighting.set_intensity_pct(brightness);
    let zoom = i16::from(state.endoscope.zoom_level);
    state.endoscope.set_zoom_level(zoom);
    let pressure = i16::from(state.insufflator.target_pressure_mmhg);
    state.insufflator.set_target_pressure_mmhg(pressure);
    let tilt = i16::from(state.table.tilt_degrees);
    state.table.set_tilt_degrees(tilt);
}

// --- Prompt -----------------------------------------------------------------

/// Static system prompt: English-only rule, the complete room-state inventory
/// (all device knobs of [`RoomState`]), the full-coverage requirement, the
/// variation-seed usage and the state-chaining semantics.
fn build_system_prompt() -> &'static str {
    "You draft realistic English speech-to-text transcripts of operating \
    room talk during laparoscopic surgery and predict the room state of a Smart-OP \
    operating room. ENGLISH ONLY: every utterance must be natural operating-room \
    speech, but English only - never any words or sentences in another language. \
    The room state has four devices and a safety interlock; consider and exercise \
    ALL of these fields when writing commands and predicting states: SurgicalLight \
    (lighting.primary_intensity_pct 0-100 %, lighting.field_mode \
    Normal/CavityFocus/AmbientRed), EndoscopeCamera (endoscope.zoom_level 1-5, \
    endoscope.white_balance_locked, endoscope.irrigation_active), Insufflator \
    (insufflator.target_pressure_mmhg hard cap 25 mmHg, insufflator.gas_flow_l_min, \
    insufflator.is_active), OperatingTable (table.tilt_degrees -15..+15 degrees, \
    table.height_cm) and safety_interlock_active. Commands within a case must \
    spread across ALL of these device knobs - brightness and light mode, zoom, \
    white balance, irrigation, insufflator pressure, gas flow, insufflator active, \
    table tilt and height, safety interlock - not just a favorite few. \
    Each request carries a variation seed: use it to vary phrasing, speech style \
    and word choice so different seeds yield genuinely different transcripts for \
    the same scenario. Surgeons, nurses and assistants speak naturally: fillers \
    (uhm, uh), self-corrections (wait no - actually...), confirmations of already \
    executed commands, smalltalk and team comments with no device connection. For \
    each utterance in order, predict the full room state after that utterance has \
    been fully processed, chained on the previous state: handlungsneutrale filler \
    talk keeps the state unchanged, self-corrections revert to the corrected \
    state. Reply with JSON only."
}

/// User prompt: scenario, variation seed, transcript plan, initial state and
/// the JSON reply contract.
fn build_user_prompt(
    scenario: &str,
    plan: &str,
    initial_state: &Value,
    variation_seed: u32,
    total: usize,
) -> String {
    format!(
        "Scenario: {scenario}.\n\n\
        Variation seed: {variation_seed}\n\n\
        Transcript plan (write one natural utterance per slot, in order):\n{plan}\n\n\
        Initial room state:\n{initial_state}\n\n\
        Reply as JSON: {{\"utterances\": [{{\"index\": 1, \"text\": \"...\"}}, ...], \
        \"states\": [{{\"index\": 1, \"room_state\": {{...}}}}, ...]}} with exactly {} \
        utterances and exactly {} states (same shape as the initial state), \
        indices 1..{total} in both arrays.",
        total, total
    )
}

// --- Model payload helpers --------------------------------------------------

#[derive(Deserialize)]
struct TranscriptUtterance {
    #[allow(dead_code)]
    index: usize,
    text: String,
}

#[derive(Deserialize)]
struct StatePrediction {
    #[allow(dead_code)]
    index: usize,
    room_state: RoomState,
}

fn parse_list<T: serde::de::DeserializeOwned>(
    value: &Value,
    key: &str,
) -> Result<Vec<T>, DatasetError> {
    let list = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| DatasetError::Malformed(format!("missing {key} array")))?;
    serde_json::from_value(Value::Array(list.clone()))
        .map_err(|e| DatasetError::Malformed(format!("{key}: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn type_sequence_respects_count_and_ratio() {
        let config = GeneratorConfig {
            utterances_per_case: 10,
            noise_ratio: 0.5,
        };
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..16 {
            let types = build_utterance_type_sequence(&config, &mut rng);
            assert_eq!(types.len(), 10);
            let noise = types.iter().filter(|t| **t == UtteranceType::Noise).count();
            assert_eq!(noise, 5, "50:50 split expected");
        }
    }

    #[test]
    fn type_sequence_without_noise_is_all_commands() {
        let config = GeneratorConfig {
            utterances_per_case: 6,
            noise_ratio: 0.0,
        };
        let mut rng = StdRng::seed_from_u64(1);
        let types = build_utterance_type_sequence(&config, &mut rng);
        assert!(types.iter().all(|t| *t == UtteranceType::Command));
    }

    #[test]
    fn type_sequence_rounds_ratio_and_clamps() {
        let config = GeneratorConfig {
            utterances_per_case: 5,
            noise_ratio: 0.5,
        };
        let mut rng = StdRng::seed_from_u64(7);
        let types = build_utterance_type_sequence(&config, &mut rng);
        // 5 * 0.5 = 2.5 rounds half away from zero: 3 noise slots.
        assert_eq!(
            types.iter().filter(|t| **t == UtteranceType::Noise).count(),
            3
        );

        let config = GeneratorConfig {
            utterances_per_case: 3,
            noise_ratio: 1.5,
        };
        let types = build_utterance_type_sequence(&config, &mut rng);
        assert!(types.iter().all(|t| *t == UtteranceType::Noise));
    }

    #[test]
    fn normalize_state_clamps_to_safety_envelope() {
        let mut state = RoomState::default();
        state.lighting.primary_intensity_pct = 150;
        state.endoscope.zoom_level = 9;
        state.insufflator.target_pressure_mmhg = 40;
        state.table.tilt_degrees = -30;

        normalize_state(&mut state);

        assert_eq!(state.lighting.primary_intensity_pct, 100);
        assert_eq!(state.endoscope.zoom_level, 5);
        assert_eq!(state.insufflator.target_pressure_mmhg, 25);
        assert_eq!(state.table.tilt_degrees, -15);
    }

    #[test]
    fn normalize_state_keeps_valid_state_untouched() {
        let mut state = RoomState::default();
        let before = state.clone();
        normalize_state(&mut state);
        assert_eq!(state, before);
    }

    #[test]
    fn system_prompt_requires_english_only() {
        let prompt = build_system_prompt();
        assert!(prompt.contains("speech-to-text transcripts"));
        assert!(
            prompt.contains("English only"),
            "prompt must demand English-only utterances"
        );
    }

    #[test]
    fn system_prompt_enumerates_all_device_knobs() {
        let prompt = build_system_prompt();
        for knob in [
            "lighting.primary_intensity_pct",
            "lighting.field_mode",
            "Normal/CavityFocus/AmbientRed",
            "endoscope.zoom_level",
            "endoscope.white_balance_locked",
            "endoscope.irrigation_active",
            "insufflator.target_pressure_mmhg",
            "insufflator.gas_flow_l_min",
            "insufflator.is_active",
            "table.tilt_degrees",
            "table.height_cm",
            "safety_interlock_active",
        ] {
            assert!(prompt.contains(knob), "prompt misses knob {knob}");
        }
        assert!(
            prompt.contains("ALL of these device knobs"),
            "prompt must demand full knob coverage per case"
        );
    }

    #[test]
    fn user_prompt_contains_variation_seed_scenario_and_initial_state() {
        let initial_state = serde_json::to_value(RoomState::default()).expect("state");
        let user = build_user_prompt(
            "laparoscopic_cholecystectomy",
            "1: device command",
            &initial_state,
            424_242,
            3,
        );

        assert!(user.contains("Variation seed: 424242"));
        assert!(user.contains("Scenario: laparoscopic_cholecystectomy"));
        assert!(user.contains("1: device command"));
        assert!(user.contains("Initial room state"));
        assert!(user.contains("\"safety_interlock_active\""));
        assert!(user.contains("\"height_cm\""));
    }
}
