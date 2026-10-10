//! Whole-case rollout: per-utterance fresh and rolling predictions with
//! retries (spec `docs/INSTRUCTIONS.md` section 5).
//!
//! Every utterance is processed twice:
//!
//! - **Fresh** (ground-truth chaining): the utterance is applied to the
//!   previous *expected* state. Independent of the model's own errors —
//!   isolates per-utterance accuracy.
//! - **Rolling** (self-chaining): the utterance is applied to the previous
//!   *predicted* state, starting at the case `initial_state`. Measures
//!   error accumulation across a case. Once a rolling prediction fails
//!   (all [`MAX_ATTEMPTS`] exhausted), every later rolling prediction
//!   fails too: there is no recovered state to chain on.
//!
//! An utterance attempt is retried up to [`MAX_ATTEMPTS`] times on
//! failure; every attempt's latency and raw answers are recorded.

use std::collections::BTreeMap;
use std::future::Future;
use std::time::{Duration, Instant};

use serde_json::Value;
use stop_core::RoomState;
use stop_dataset::DatasetCase;

use crate::raw::{RawCase, RawPass, RawPrediction, RawUtterance};

/// Attempts per utterance before it is recorded as failed.
pub const MAX_ATTEMPTS: usize = 3;
/// Base backoff between attempts; doubles per attempt, capped at [`BACKOFF_MAX`].
pub const BACKOFF_BASE: Duration = Duration::from_secs(2);
/// Upper bound for a single backoff wait.
pub const BACKOFF_MAX: Duration = Duration::from_secs(30);

/// Outcome of one utterance attempt: the newest room state, the pass
/// latency and the raw decision answers snapshot, or an error message.
pub type AttemptResult = Result<(RoomState, Duration, BTreeMap<String, Value>), String>;

/// Backoff before retrying after the given 1-based attempt (doubling, capped).
pub fn backoff_for_attempt(attempt: usize) -> Duration {
    (BACKOFF_BASE * 2u32.pow(attempt.saturating_sub(1) as u32)).min(BACKOFF_MAX)
}

/// Runs one whole case: both prediction variants per utterance.
///
/// `process` performs one inference pass for the given input room state and
/// utterance (the live provider in `main.rs`, a scripted fake in tests).
/// Each variant retries its attempts up to [`MAX_ATTEMPTS`]; the variants
/// run sequentially and independently.
pub async fn run_case<F, Fut>(case: &DatasetCase, model_name: &str, mut process: F) -> RawCase
where
    F: FnMut(RoomState, String) -> Fut,
    Fut: Future<Output = AttemptResult>,
{
    // Rolling chain state; the fresh variant never mutates it.
    let mut rolling_state = case.initial_state.clone();
    // Rolling prediction failed before => stays failed: no state to chain on.
    let mut rolling_broken = false;
    let mut entries = Vec::with_capacity(case.history.len());

    for (entry_index, entry) in case.history.iter().enumerate() {
        // Fresh variant: input is the previous expected state.
        let fresh_input = previous_expected(case, entry_index);
        let fresh = run_prediction(
            case,
            entry_index,
            fresh_input.clone(),
            &entry.raw_utterance,
            &mut process,
        )
        .await;

        // Rolling variant: input is the previous predicted state; once
        // broken, every later prediction fails without any inference call.
        let rolling = if rolling_broken {
            RawPrediction {
                state: Err("rolling prediction failed earlier in the case".to_string()),
                inference_passes: Vec::new(),
                wall_latency_ms: 0.0,
            }
        } else {
            let prediction = run_prediction(
                case,
                entry_index,
                rolling_state.clone(),
                &entry.raw_utterance,
                &mut process,
            )
            .await;
            match &prediction.state {
                Ok(new_room) => rolling_state = new_room.clone(),
                Err(_) => rolling_broken = true,
            }
            prediction
        };

        entries.push(RawUtterance {
            entry_index,
            raw_utterance: entry.raw_utterance.clone(),
            expected_state: entry.expected_state.clone(),
            fresh_prediction: fresh,
            rolling_prediction: rolling,
        });
    }

    RawCase {
        case_id: case.id.clone(),
        scenario: case.scenario.clone(),
        model_name: model_name.to_string(),
        initial_state: case.initial_state.clone(),
        entries,
    }
}

/// The expected state feeding the fresh variant of `entry_index`: the
/// previous history entry's expected state, or the case `initial_state`.
fn previous_expected(case: &DatasetCase, entry_index: usize) -> RoomState {
    if entry_index == 0 {
        case.initial_state.clone()
    } else {
        case.history[entry_index - 1].expected_state.clone()
    }
}

/// Processes one utterance variant: up to [`MAX_ATTEMPTS`] retries with
/// backoff, recording every attempt's latency and raw answers.
async fn run_prediction<F, Fut>(
    case: &DatasetCase,
    entry_index: usize,
    input_state: RoomState,
    utterance: &str,
    process: &mut F,
) -> RawPrediction
where
    F: FnMut(RoomState, String) -> Fut,
    Fut: Future<Output = AttemptResult>,
{
    let started = Instant::now();
    let mut inference_passes: Vec<RawPass> = Vec::with_capacity(MAX_ATTEMPTS);
    let mut attempt = 0usize;
    let state = loop {
        attempt += 1;
        let attempt_started = Instant::now();
        match process(input_state.clone(), utterance.to_string()).await {
            Ok((new_room, latency, answers)) => {
                inference_passes.push(RawPass {
                    latency_ms: latency.as_secs_f64() * 1000.0,
                    answers,
                });
                break Ok(new_room);
            }
            Err(err) => {
                // A failed attempt still consumed a pass: record it with
                // its wall-clock latency and no answers.
                inference_passes.push(RawPass {
                    latency_ms: attempt_started.elapsed().as_secs_f64() * 1000.0,
                    answers: BTreeMap::new(),
                });
                if attempt >= MAX_ATTEMPTS {
                    break Err(err);
                }
                let backoff = backoff_for_attempt(attempt);
                tracing::warn!(
                    case = %case.id,
                    entry_index,
                    attempt,
                    retry_in = ?backoff,
                    error = %err,
                    "utterance attempt failed, retrying"
                );
                tokio::time::sleep(backoff).await;
            }
        }
    };
    RawPrediction {
        state,
        inference_passes,
        wall_latency_ms: started.elapsed().as_secs_f64() * 1000.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::VecDeque;
    use std::rc::Rc;
    use stop_dataset::schema::HistoryEntry;

    fn case_with_utts(utterances: &[&str]) -> DatasetCase {
        DatasetCase {
            id: "case_test".to_string(),
            scenario: "test".to_string(),
            model: "test-model".to_string(),
            initial_state: RoomState::default(),
            history: utterances
                .iter()
                .map(|u| HistoryEntry {
                    raw_utterance: u.to_string(),
                    expected_state: RoomState::default(),
                })
                .collect(),
        }
    }

    fn changed(base: &RoomState, pct: u8) -> RoomState {
        let mut state = base.clone();
        state.lighting.primary_intensity_pct = pct;
        state
    }

    #[tokio::test(start_paused = true)]
    async fn retries_transient_errors_until_success() {
        // First attempt fails, second succeeds: exactly two attempts.
        let attempts = Rc::new(RefCell::new(0usize));
        let counter = Rc::clone(&attempts);
        let result = run_case(
            &case_with_utts(&["dim"]),
            "test-model",
            move |state, _utt| {
                let counter = Rc::clone(&counter);
                let state = state.clone();
                async move {
                    *counter.borrow_mut() += 1;
                    if *counter.borrow() == 1 {
                        Err("transport error".to_string())
                    } else {
                        Ok((
                            changed(&state, 60),
                            Duration::from_millis(7),
                            BTreeMap::new(),
                        ))
                    }
                }
            },
        )
        .await;

        assert_eq!(*attempts.borrow(), 3); // fresh 2 attempts, rolling 1.
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.model_name, "test-model");
        assert_eq!(
            result.entries[0].fresh_prediction.state,
            Ok(changed(&RoomState::default(), 60))
        );
        assert_eq!(
            result.entries[0].rolling_prediction.state,
            Ok(changed(&RoomState::default(), 60))
        );
        // Both fresh attempts (one failed, one successful) recorded; the
        // rolling variant only needed its first attempt.
        assert_eq!(result.entries[0].fresh_prediction.inference_passes.len(), 2);
        assert_eq!(
            result.entries[0].rolling_prediction.inference_passes.len(),
            1
        );
    }

    #[tokio::test(start_paused = true)]
    async fn records_err_after_attempts_exhausted() {
        let attempts = Rc::new(RefCell::new(0usize));
        let counter = Rc::clone(&attempts);
        let result = run_case(
            &case_with_utts(&["dim"]),
            "test-model",
            move |_state, _utt| {
                let counter = Rc::clone(&counter);
                async move {
                    *counter.borrow_mut() += 1;
                    Err("provider timeout".to_string())
                }
            },
        )
        .await;

        assert_eq!(*attempts.borrow(), MAX_ATTEMPTS * 2);
        assert_eq!(
            result.entries[0].fresh_prediction.state,
            Err("provider timeout".to_string())
        );
        assert_eq!(
            result.entries[0].rolling_prediction.state,
            Err("provider timeout".to_string())
        );
        assert_eq!(
            result.entries[0].fresh_prediction.inference_passes.len(),
            MAX_ATTEMPTS
        );
    }

    #[tokio::test(start_paused = true)]
    async fn fresh_variant_chains_from_expected_states() {
        // The fresh variant always sees the previous expected state,
        // regardless of what the model predicted before.
        let inputs: Rc<RefCell<Vec<u8>>> = Rc::new(RefCell::new(Vec::new()));
        let recorded = Rc::clone(&inputs);
        // Every utterance predicts 60 (wrong for the middle one).
        let result = run_case(
            &case_with_utts(&["a", "b", "c"]),
            "test-model",
            move |state, _utt| {
                let recorded = Rc::clone(&recorded);
                let state = state.clone();
                async move {
                    recorded
                        .borrow_mut()
                        .push(state.lighting.primary_intensity_pct);
                    Ok((
                        changed(&state, 60),
                        Duration::from_millis(1),
                        BTreeMap::new(),
                    ))
                }
            },
        )
        .await;

        // Fresh inputs: initial 80, then expected 80, then expected 80.
        // Rolling inputs interleaved: 80 (initial), then 60 (own
        // prediction), then 60 again.
        assert_eq!(*inputs.borrow(), vec![80, 80, 80, 60, 80, 60]);
        for entry in &result.entries {
            assert_eq!(
                entry.fresh_prediction.state,
                Ok(changed(&RoomState::default(), 60))
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn rolling_variant_chains_from_predicted_states() {
        // The rolling variant chains on its own predicted states.
        let inputs: Rc<RefCell<Vec<u8>>> = Rc::new(RefCell::new(Vec::new()));
        let recorded = Rc::clone(&inputs);
        let result = run_case(
            &case_with_utts(&["a", "b"]),
            "test-model",
            move |state, _utt| {
                let recorded = Rc::clone(&recorded);
                let state = state.clone();
                async move {
                    recorded
                        .borrow_mut()
                        .push(state.lighting.primary_intensity_pct);
                    Ok((
                        changed(&state, 60),
                        Duration::from_millis(1),
                        BTreeMap::new(),
                    ))
                }
            },
        )
        .await;

        // Calls in order: fresh0(80 initial), roll0(80), fresh1(80
        // expected), roll1(60 own prediction).
        assert_eq!(*inputs.borrow(), vec![80, 80, 80, 60]);
        assert_eq!(
            result.entries[1].rolling_prediction.state,
            Ok(changed(&RoomState::default(), 60))
        );
    }

    #[tokio::test(start_paused = true)]
    async fn broken_rolling_chain_fails_all_later_entries() {
        // Utterance 0 rolls fine at 70; utterance 1 exhausts all attempts;
        // utterance 2 must fail without any inference call.
        let inputs: Rc<RefCell<Vec<u8>>> = Rc::new(RefCell::new(Vec::new()));
        let recorded = Rc::clone(&inputs);
        let script: Rc<RefCell<VecDeque<AttemptResult>>> = Rc::new(RefCell::new(
            vec![
                // Fresh 0, rolling 0.
                Ok((
                    changed(&RoomState::default(), 70),
                    Duration::from_millis(1),
                    BTreeMap::new(),
                )),
                Ok((
                    changed(&RoomState::default(), 70),
                    Duration::from_millis(1),
                    BTreeMap::new(),
                )),
                // Fresh 1 succeeds, rolling 1 fails all three attempts.
                Ok((
                    changed(&RoomState::default(), 60),
                    Duration::from_millis(1),
                    BTreeMap::new(),
                )),
                Err("boom".to_string()),
                Err("boom".to_string()),
                Err("boom".to_string()),
                // Fresh 2 still runs; rolling 2 fails without a call.
                Ok((
                    changed(&RoomState::default(), 60),
                    Duration::from_millis(1),
                    BTreeMap::new(),
                )),
            ]
            .into(),
        ));
        let script = Rc::clone(&script);

        let result = run_case(
            &case_with_utts(&["a", "b", "c"]),
            "test-model",
            move |state, _utt| {
                let recorded = Rc::clone(&recorded);
                let script = Rc::clone(&script);
                async move {
                    recorded
                        .borrow_mut()
                        .push(state.lighting.primary_intensity_pct);
                    script.borrow_mut().pop_front().expect("script exhausted")
                }
            },
        )
        .await;

        // Calls: fresh0(80), roll0(80), fresh1(80), roll1(70)x3, fresh2(80).
        assert_eq!(*inputs.borrow(), vec![80, 80, 80, 70, 70, 70, 80]);
        assert!(result.entries[0].rolling_prediction.state.is_ok());
        assert!(result.entries[1].rolling_prediction.state.is_err());
        // Utterance 2: rolling broken, no passes recorded.
        assert!(result.entries[2].rolling_prediction.state.is_err());
        assert!(
            result.entries[2]
                .rolling_prediction
                .inference_passes
                .is_empty()
        );
        // Fresh variant unaffected throughout.
        assert!(result.entries[2].fresh_prediction.state.is_ok());
        assert_eq!(result.entries[2].fresh_prediction.inference_passes.len(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn fresh_uses_expected_not_predicted_inputs() {
        // Expected states differ from the initial state: fresh must see
        // them; rolling must not.
        let mut case = case_with_utts(&["a", "b"]);
        case.history[0].expected_state = changed(&RoomState::default(), 40);
        case.history[1].expected_state = changed(&RoomState::default(), 50);

        let inputs: Rc<RefCell<Vec<u8>>> = Rc::new(RefCell::new(Vec::new()));
        let recorded = Rc::clone(&inputs);
        let result = run_case(&case, "test-model", move |state, _utt| {
            let recorded = Rc::clone(&recorded);
            let state = state.clone();
            async move {
                recorded
                    .borrow_mut()
                    .push(state.lighting.primary_intensity_pct);
                Ok((
                    changed(&state, 60),
                    Duration::from_millis(1),
                    BTreeMap::new(),
                ))
            }
        })
        .await;

        // Calls in order: fresh0(80 initial), roll0(80), fresh1(40 expected),
        // roll1(60 predicted).
        assert_eq!(*inputs.borrow(), vec![80, 80, 40, 60]);
        assert_eq!(
            result.entries[0].fresh_prediction.state,
            Ok(changed(&changed(&RoomState::default(), 40), 60))
        );
        assert_eq!(
            result.entries[1].fresh_prediction.state,
            Ok(changed(&changed(&RoomState::default(), 50), 60))
        );
    }

    #[test]
    fn backoff_doubles_and_caps() {
        assert_eq!(backoff_for_attempt(1), Duration::from_secs(2));
        assert_eq!(backoff_for_attempt(2), Duration::from_secs(4));
        assert_eq!(backoff_for_attempt(3), Duration::from_secs(8));
        assert_eq!(backoff_for_attempt(10), BACKOFF_MAX);
    }
}
