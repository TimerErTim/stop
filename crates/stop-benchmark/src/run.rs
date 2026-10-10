//! Whole-case rollout: chained predicted room states with per-utterance
//! retries (spec `docs/INSTRUCTIONS.md` section 5).
//!
//! The predicted rollout is independent of the dataset's expected states:
//! it starts at the case `initial_state` and each utterance sees the room
//! state predicted for the previous utterance. An utterance is retried up to
//! [`MAX_ATTEMPTS`] times on failure; every attempt's latency is recorded.
//! When all attempts fail, the entry carries `Err(last error)` and the
//! rollout continues from the latest `Ok()` room state (or `initial_state`).

use std::future::Future;
use std::time::{Duration, Instant};

use stop_core::RoomState;
use stop_dataset::DatasetCase;

use crate::raw::{RawCase, RawPass, RawUtterance};

/// Attempts per utterance before it is recorded as failed.
pub const MAX_ATTEMPTS: usize = 3;
/// Base backoff between attempts; doubles per attempt, capped at [`BACKOFF_MAX`].
pub const BACKOFF_BASE: Duration = Duration::from_secs(2);
/// Upper bound for a single backoff wait.
pub const BACKOFF_MAX: Duration = Duration::from_secs(30);

/// Outcome of one utterance attempt: the newest room state, the pass
/// latency and the raw decision answers snapshot, or an error message.
pub type AttemptResult = Result<(RoomState, Duration, Option<RawPass>), String>;

/// Backoff before retrying after the given 1-based attempt (doubling, capped).
pub fn backoff_for_attempt(attempt: usize) -> Duration {
    (BACKOFF_BASE * 2u32.pow(attempt.saturating_sub(1) as u32)).min(BACKOFF_MAX)
}

/// Runs one whole case as an independent predicted rollout.
///
/// `process` performs one inference pass for the given input room state and
/// utterance (the live provider in `main.rs`, a scripted fake in tests). Every
/// attempt of an utterance is retried up to [`MAX_ATTEMPTS`]; a failed
/// utterance never corrupts the chain because the next utterance starts from
/// the latest `Ok()` predicted state.
pub async fn run_case<F, Fut>(case: &DatasetCase, model_name: &str, mut process: F) -> RawCase
where
    F: FnMut(RoomState, String) -> Fut,
    Fut: Future<Output = AttemptResult>,
{
    let mut state = case.initial_state.clone();
    let mut entries = Vec::with_capacity(case.history.len());

    for (entry_index, entry) in case.history.iter().enumerate() {
        let started = Instant::now();
        let mut pass_latencies_ms = Vec::with_capacity(MAX_ATTEMPTS);
        let mut attempt = 0usize;
        let predicted = loop {
            attempt += 1;
            let attempt_started = Instant::now();
            match process(state.clone(), entry.raw_utterance.clone()).await {
                Ok((new_room, latency, pass_answers)) => {
                    pass_latencies_ms.push(latency.as_secs_f64() * 1000.0);
                    break Ok((new_room, pass_answers));
                }
                Err(err) => {
                    // A failed attempt still consumed a pass: record its
                    // wall-clock time for the latency distribution.
                    pass_latencies_ms.push(attempt_started.elapsed().as_secs_f64() * 1000.0);
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

        let (predicted, pass_answers) = match predicted {
            Ok((new_room, pass_answers)) => (Ok(new_room), pass_answers),
            Err(err) => (Err(err), None),
        };

        if let Ok(new_room) = &predicted {
            // Chain on the predicted state only; expected states never feed
            // the rollout.
            state = new_room.clone();
        }

        entries.push(RawUtterance {
            entry_index,
            raw_utterance: entry.raw_utterance.clone(),
            expected_output_state: entry.expected_output_state.clone(),
            predicted_output_state: predicted,
            pass_answers,
            wall_latency_ms: started.elapsed().as_secs_f64() * 1000.0,
            pass_latencies_ms,
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
                    expected_output_state: RoomState::default(),
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
                        Ok((changed(&state, 60), Duration::from_millis(7), None))
                    }
                }
            },
        )
        .await;

        assert_eq!(*attempts.borrow(), 2);
        assert_eq!(result.entries.len(), 1);
        assert_eq!(result.model_name, "test-model");
        assert_eq!(
            result.entries[0].predicted_output_state,
            Ok(changed(&RoomState::default(), 60))
        );
        // Both attempts (one failed, one successful) recorded.
        assert_eq!(result.entries[0].pass_latencies_ms.len(), 2);
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

        assert_eq!(*attempts.borrow(), MAX_ATTEMPTS);
        assert_eq!(
            result.entries[0].predicted_output_state,
            Err("provider timeout".to_string())
        );
        assert_eq!(result.entries[0].pass_latencies_ms.len(), MAX_ATTEMPTS);
    }

    #[tokio::test(start_paused = true)]
    async fn failed_entry_chains_from_latest_ok_state() {
        // Utterance 0 succeeds at 70%, utterance 1 fails, utterance 2 must
        // therefore see 70% (the latest Ok state), not the initial 80%.
        let inputs: Rc<RefCell<Vec<u8>>> = Rc::new(RefCell::new(Vec::new()));
        let recorded = Rc::clone(&inputs);
        let script: Rc<RefCell<VecDeque<AttemptResult>>> = Rc::new(RefCell::new(
            vec![
                Ok((
                    changed(&RoomState::default(), 70),
                    Duration::from_millis(1),
                    None,
                )),
                Err("boom".to_string()),
                Err("boom".to_string()),
                Err("boom".to_string()),
                Ok((
                    changed(&RoomState::default(), 70),
                    Duration::from_millis(1),
                    None,
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

        // Utterance 0 sees initial 80; utterance 1 sees 70; utterance 2 sees 70.
        assert_eq!(*inputs.borrow(), vec![80, 70, 70, 70, 70]);
        assert_eq!(result.entries.len(), 3);
        assert!(result.entries[0].predicted_output_state.is_ok());
        assert!(result.entries[1].predicted_output_state.is_err());
        assert!(result.entries[2].predicted_output_state.is_ok());
    }

    #[test]
    fn backoff_doubles_and_caps() {
        assert_eq!(backoff_for_attempt(1), Duration::from_secs(2));
        assert_eq!(backoff_for_attempt(2), Duration::from_secs(4));
        assert_eq!(backoff_for_attempt(3), Duration::from_secs(8));
        assert_eq!(backoff_for_attempt(10), BACKOFF_MAX);
    }
}
