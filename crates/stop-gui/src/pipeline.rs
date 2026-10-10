//! Tokio pipeline wiring: input sources, executor consumer, event output.
//!
//! The GUI render thread never touches the executor; all cross-thread
//! traffic goes through the command broadcast and the event mpsc channel.

use stop_core::{
    ExecutionError, InferencePort, RoomState, SinglePassExecutor, systemone::SystemOneClient,
};

use crate::events::GuiEvent;
use crate::source::{BroadcastSource, CommandStreamSource};

/// GUI-facing pipeline handles: command broadcast and event channel.
///
/// Clonable senders; the executor task, the input sources, and the window
/// all keep their own clones.
#[derive(Clone)]
pub struct Pipeline {
    pub commands_tx: tokio::sync::broadcast::Sender<String>,
    pub events_tx: tokio::sync::mpsc::UnboundedSender<GuiEvent>,
    pub initial_room: RoomState,
}

impl Pipeline {
    pub fn new() -> (Self, tokio::sync::mpsc::UnboundedReceiver<GuiEvent>) {
        let (commands_tx, _command_rx) = tokio::sync::broadcast::channel(64);
        let (events_tx, events_rx) = tokio::sync::mpsc::unbounded_channel();
        (
            Self {
                commands_tx,
                events_tx,
                initial_room: RoomState::default(),
            },
            events_rx,
        )
    }

    /// Spawns the sequential consumer over the real System-One client: one
    /// utterance at a time, in order, awaiting the single-pass executor.
    /// The shared mutable state lives inside this task; the GUI only
    /// receives snapshots.
    ///
    /// Concrete `SystemOneClient` argument: the async-fn-in-trait future of
    /// [`InferencePort::single_pass`] is only `Send` for a known
    /// implementation, so the spawned task requires the real client type
    /// (tests drive the generic [`run_executor_loop`] directly instead).
    pub fn spawn_executor(
        &self,
        executor: SinglePassExecutor<SystemOneClient>,
    ) -> tokio::task::JoinHandle<()> {
        let events_tx = self.events_tx.clone();
        let commands_tx = self.commands_tx.clone();
        let initial_room = self.initial_room.clone();
        tokio::spawn(async move {
            let source = BroadcastSource::new(commands_tx.subscribe());
            run_executor_loop(&executor, source, &events_tx, initial_room).await;
        })
    }

    /// Spawns a stdin source publishing onto the command broadcast.
    pub fn spawn_stdin(&self) -> tokio::task::JoinHandle<()> {
        let commands_tx = self.commands_tx.clone();
        tokio::spawn(async move {
            let mut source = crate::source::StdinCliSource;
            loop {
                match source.next_command().await {
                    Ok(Some(cmd)) => {
                        let _ = commands_tx.send(cmd);
                    }
                    Ok(None) => {
                        tracing::info!("stdin EOF");
                        break;
                    }
                    Err(e) => {
                        tracing::error!("stdin source failed: {e}");
                        break;
                    }
                }
            }
        })
    }

    /// Spawns the microphone STT pipeline (feature `mic`).
    #[cfg(feature = "mic")]
    pub fn spawn_mic(
        &self,
    ) -> Result<std::sync::Arc<std::sync::atomic::AtomicBool>, crate::stt::SttError> {
        crate::stt::spawn_stt_pipeline(self.events_tx.clone(), self.commands_tx.clone())
    }
}

/// Sequential executor consumer, generic over the inference port.
///
/// Drives the source until it closes: for each utterance one
/// `PromptReceived`, then either `ExecutionFinished` (state snapshot plus
/// report) or `ExecutionFailed`.
pub async fn run_executor_loop<I>(
    executor: &SinglePassExecutor<I>,
    mut source: impl CommandStreamSource,
    events_tx: &tokio::sync::mpsc::UnboundedSender<GuiEvent>,
    mut room: RoomState,
) where
    I: InferencePort,
{
    loop {
        let Some(utterance) = (match source.next_command().await {
            Ok(cmd) => cmd,
            Err(e) => {
                tracing::error!("command source failed: {e}");
                break;
            }
        }) else {
            tracing::info!("command stream closed");
            break;
        };

        let _ = events_tx.send(GuiEvent::PromptReceived(utterance.clone()));
        match executor.process_utterance(&room, &utterance).await {
            Ok(result) => {
                room = result.new_room;
                let _ = events_tx.send(GuiEvent::ExecutionFinished {
                    utterance,
                    new_room: room.clone(),
                    report: result.report,
                });
            }
            Err(error) => {
                let detail = describe_error(&error);
                tracing::warn!("utterance failed: {detail}");
                let _ = events_tx.send(GuiEvent::ExecutionFailed {
                    utterance,
                    error: detail,
                });
            }
        }
    }
}

fn describe_error(error: &ExecutionError) -> String {
    match error {
        ExecutionError::Provider(inner) => format!("provider: {inner}"),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use stop_core::{
        InferenceInput, InferenceOutcome, LightDecision, ProviderError, UtteranceDecision,
        ValueChange,
    };

    /// Mock port: two configured outcomes, then transport errors.
    struct MockPort {
        decision: UtteranceDecision,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl InferencePort for MockPort {
        async fn single_pass(
            &self,
            _input: &InferenceInput<'_>,
        ) -> Result<InferenceOutcome, ProviderError> {
            let n = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if n < 2 {
                Ok(InferenceOutcome {
                    decision: self.decision.clone(),
                    latency: Duration::from_millis(19),
                    answers: Default::default(),
                })
            } else {
                Err(ProviderError::Transport("mock exhausted".into()))
            }
        }
    }

    /// Scripted command source for loop tests.
    struct ScriptedSource {
        commands: Vec<String>,
    }

    impl CommandStreamSource for ScriptedSource {
        async fn next_command(&mut self) -> Result<Option<String>, crate::source::SourceError> {
            if self.commands.is_empty() {
                Ok(None)
            } else {
                Ok(Some(self.commands.remove(0)))
            }
        }
    }

    async fn drain_until(
        events_rx: &mut tokio::sync::mpsc::UnboundedReceiver<GuiEvent>,
        finished: usize,
    ) -> (usize, Vec<RoomState>) {
        let mut prompts = 0;
        let mut rooms = Vec::new();
        while rooms.len() < finished {
            match events_rx.recv().await.unwrap() {
                GuiEvent::PromptReceived(_) => prompts += 1,
                GuiEvent::ExecutionFinished { new_room, .. } => rooms.push(new_room),
                other => panic!("unexpected event: {other:?}"),
            }
        }
        (prompts, rooms)
    }

    #[tokio::test]
    async fn executor_loop_runs_sequentially_and_emits_events() {
        let decision = UtteranceDecision {
            light: LightDecision {
                brightness: Some(ValueChange::Decrease(2)),
                ..Default::default()
            },
            ..Default::default()
        };
        let executor = SinglePassExecutor::new(MockPort {
            decision,
            calls: std::sync::atomic::AtomicUsize::new(0),
        });

        let (pipeline, mut events_rx) = Pipeline::new();
        let source = ScriptedSource {
            commands: vec!["dim the light by two steps".into(), "and again".into()],
        };
        run_executor_loop(&executor, source, &pipeline.events_tx, RoomState::default()).await;

        let (prompts, rooms) = drain_until(&mut events_rx, 2).await;
        assert_eq!(prompts, 2);
        let baseline = RoomState::default().lighting.primary_intensity_pct;
        assert_eq!(rooms[0].lighting.primary_intensity_pct, baseline - 2);
        assert_eq!(rooms[1].lighting.primary_intensity_pct, baseline - 4);
    }

    #[tokio::test]
    async fn failures_surface_as_execution_failed() {
        struct FailingPort;
        impl InferencePort for FailingPort {
            async fn single_pass(
                &self,
                _input: &InferenceInput<'_>,
            ) -> Result<InferenceOutcome, ProviderError> {
                Err(ProviderError::Transport("down".into()))
            }
        }

        let (pipeline, mut events_rx) = Pipeline::new();
        let executor = SinglePassExecutor::new(FailingPort);
        let source = ScriptedSource {
            commands: vec!["any".into()],
        };
        run_executor_loop(&executor, source, &pipeline.events_tx, RoomState::default()).await;

        match events_rx.recv().await.unwrap() {
            GuiEvent::PromptReceived(_) => {}
            other => panic!("expected prompt first: {other:?}"),
        }
        match events_rx.recv().await.unwrap() {
            GuiEvent::ExecutionFailed { error, .. } => {
                assert!(error.contains("down"), "error detail: {error}");
            }
            other => panic!("expected failure: {other:?}"),
        }
    }
}
