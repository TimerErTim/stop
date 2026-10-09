//! Two-step scenario generator (spec `docs/INSTRUCTIONS.md` section 4.3).
//!
//! Step 1 drafts an STT-like transcript micro-segment (commands mixed with
//! handlungsneutrale filler/noise utterances) following a random utterance
//! type sequence. Step 2 lets the model predict the desired room state after
//! each utterance, based on the previous state. Noise and self-corrections
//! are handled naturally there: filler talk keeps the state, corrections
//! revert to the corrected state.

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

    /// Generates one complete scenario: transcript draft, then per-utterance
    /// room-state prediction on top of the previous state.
    pub async fn generate_case(
        &self,
        rng: &mut impl Rng,
        id: &str,
        scenario: &str,
    ) -> Result<DatasetCase, DatasetError> {
        let types = build_utterance_type_sequence(&self.config, rng);
        let utterances = self.request_transcript(scenario, &types).await?;
        let states = self.request_states(&utterances).await?;

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

    /// Step 1: STT-like transcript micro-segment matching the type sequence.
    async fn request_transcript(
        &self,
        scenario: &str,
        types: &[UtteranceType],
    ) -> Result<Vec<String>, DatasetError> {
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

        let system = "You draft realistic English speech-to-text transcripts of operating \
            room talk during laparoscopic surgery. Surgeons, nurses and assistants speak \
            naturally: fillers (uhm, uh), self-corrections (wait no - actually...), \
            confirmations of already executed commands, smalltalk and team comments with \
            no device connection. Reply with JSON only.";
        let user = format!(
            "Scenario: {scenario}.\n\nWrite one short transcript segment with exactly {} \
            utterances in this order (one utterance per line, natural wording):\n{plan}\n\n\
            Reply as JSON: {{\"utterances\": [{{\"index\": 1, \"text\": \"...\"}}, ...]}} \
            with one entry per planned utterance, indices 1..{}.",
            types.len(),
            types.len()
        );

        let value = self.client.complete_json(system, &user).await?;
        let utterances: Vec<TranscriptUtterance> = parse_list(&value, "utterances")?;
        if utterances.len() != types.len() {
            return Err(DatasetError::Malformed(format!(
                "transcript step returned {} utterances, expected {}",
                utterances.len(),
                types.len()
            )));
        }
        Ok(utterances.into_iter().map(|u| u.text).collect())
    }

    /// Step 2: predicts the desired room state after each utterance, based on
    /// the previous state (starting from the initial state).
    async fn request_states(&self, utterances: &[String]) -> Result<Vec<RoomState>, DatasetError> {
        let numbered = utterances
            .iter()
            .enumerate()
            .map(|(i, text)| format!("{}: {text}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let initial_state = serde_json::to_value(RoomState::default())
            .map_err(|e| DatasetError::Malformed(e.to_string()))?;

        let system = "You predict the room state of a Smart-OP operating room with four \
            devices after spoken utterances: SurgicalLight (brightness 0-100 %, light mode \
            Normal/CavityFocus/AmbientRed), EndoscopeCamera (zoom 1-5, irrigation), \
            Insufflator (target pressure, hard cap 25 mmHg), OperatingTable (tilt -15..+15 \
            degrees). For each utterance in order, output the full room state after that \
            utterance has been fully processed, starting from the given initial state and \
            chaining: each state is based on the previous one. Handlungsneutrale filler \
            talk keeps the state unchanged; self-corrections revert to the corrected \
            state. Reply with JSON only.";
        let user = format!(
            "Initial room state:\n{initial_state}\n\n\
            Utterances in order:\n{numbered}\n\n\
            Reply as JSON: {{\"states\": [{{\"index\": 1, \"room_state\": {{...}}}}, ...]}} \
            with one full room_state (same shape as the initial state) per utterance, \
            indices 1..{}.",
            utterances.len()
        );

        let value = self.client.complete_json(system, &user).await?;
        let states: Vec<StatePrediction> = parse_list(&value, "states")?;
        if states.len() != utterances.len() {
            return Err(DatasetError::Malformed(format!(
                "state step returned {} states, expected {}",
                states.len(),
                utterances.len()
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
        Ok(states.into_iter().map(|s| s.room_state).collect())
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
}
