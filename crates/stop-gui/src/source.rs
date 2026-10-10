//! Input seam for command sources: stdin CLI and microphone STT.
//!
//! Driving-adapter concern of the GUI crate (spec section 7 seam, hosted in
//! `stop-gui`): anything that produces utterance strings feeds the tokio
//! consumer task through [`CommandStreamSource`].

use thiserror::Error;

/// Failures of a command source.
#[derive(Debug, Error)]
pub enum SourceError {
    #[error("stdin read failed: {0}")]
    Stdin(std::io::Error),

    #[error("microphone source failed: {0}")]
    Microphone(String),
}

/// Async stream of user utterances. Implementations block while waiting for
/// the next command; `Ok(None)` means the source is exhausted (EOF).
///
/// Native `async fn` in trait: the consumer is generic over the source, so
/// no `async-trait` boxing is needed (mirrors [`stop_core::InferencePort`]).
#[allow(async_fn_in_trait)]
pub trait CommandStreamSource: Send {
    async fn next_command(&mut self) -> Result<Option<String>, SourceError>;
}

/// Standard-input source: one utterance per line, EOF ends the stream.
pub struct StdinCliSource;

impl StdinCliSource {
    /// Blocking line read moved to a spawned thread so the async caller
    /// never blocks the runtime's worker threads.
    fn read_line_blocking() -> std::io::Result<String> {
        std::thread::spawn(|| {
            let mut buffer = String::new();
            std::io::stdin().read_line(&mut buffer)?;
            Ok(buffer)
        })
        .join()
        .map_err(|_| std::io::Error::other("stdin reader thread panicked"))?
    }
}

impl CommandStreamSource for StdinCliSource {
    async fn next_command(&mut self) -> Result<Option<String>, SourceError> {
        loop {
            let buffer = tokio::task::spawn_blocking(Self::read_line_blocking)
                .await
                .map_err(|e| SourceError::Stdin(std::io::Error::other(e.to_string())))?
                .map_err(SourceError::Stdin)?;
            let trimmed = buffer.trim();
            if !trimmed.is_empty() {
                return Ok(Some(trimmed.to_string()));
            }
            // Blank line: keep reading instead of ending the stream.
        }
    }
}

/// Commands are fanned out to consumers (executor feed, CLI echo, HUD) as
/// a shared broadcast stream; every source adapter converts into it.
pub type CommandBroadcast = tokio::sync::broadcast::Sender<String>;

/// Wraps a [`tokio::sync::broadcast::Receiver`] as a command source. Used by
/// the executor consumer; the GUI HUD drains its own receiver for the
/// prompt display.
pub struct BroadcastSource {
    rx: tokio::sync::broadcast::Receiver<String>,
}

impl BroadcastSource {
    pub fn new(rx: tokio::sync::broadcast::Receiver<String>) -> Self {
        Self { rx }
    }
}

impl CommandStreamSource for BroadcastSource {
    async fn next_command(&mut self) -> Result<Option<String>, SourceError> {
        loop {
            match self.rx.recv().await {
                Ok(cmd) if cmd.trim().is_empty() => continue,
                Ok(cmd) => return Ok(Some(cmd)),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("command broadcast lagged, dropped {n} commands");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => return Ok(None),
            }
        }
    }
}

/// Publishes one utterance on the command broadcast.
pub fn publish_command(broadcast: &CommandBroadcast, utterance: &str) {
    let _ = broadcast.send(utterance.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scripted source for trait-contract tests.
    struct ScriptedSource {
        lines: Vec<Option<String>>,
    }

    impl CommandStreamSource for ScriptedSource {
        async fn next_command(&mut self) -> Result<Option<String>, SourceError> {
            if self.lines.is_empty() {
                Ok(None)
            } else {
                Ok(self.lines.remove(0))
            }
        }
    }

    #[tokio::test]
    async fn scripted_source_yields_lines_then_eof() {
        let mut source = ScriptedSource {
            lines: vec![Some("dim the light".into()), Some("zoom in".into()), None],
        };
        assert_eq!(
            source.next_command().await.unwrap(),
            Some("dim the light".into())
        );
        assert_eq!(source.next_command().await.unwrap(), Some("zoom in".into()));
        assert_eq!(source.next_command().await.unwrap(), None);
    }

    #[tokio::test]
    async fn broadcast_source_skips_blank_and_reports_lag_free_flow() {
        let (tx, rx) = tokio::sync::broadcast::channel(4);
        let mut source = BroadcastSource::new(rx);

        tx.send("  ".into()).unwrap();
        tx.send("set light to 40".into()).unwrap();
        drop(tx);

        assert_eq!(
            source.next_command().await.unwrap(),
            Some("set light to 40".into())
        );
        assert_eq!(source.next_command().await.unwrap(), None);
    }
}
