# mise Tooling for `stop`

This document describes how the project manages tools, environment variables, and runnable tasks via [mise](https://mise.jdx.dev/). The setup follows the conventions of the reference project `yams` (`mise.toml`, `mise.dev.toml`, `mise.local.toml`, plus a `tasks/` directory of included task files).

## Configuration Files

| File | Purpose | Versioned |
|------|---------|-----------|
| `mise.toml` | Global config: tool versions, task includes, shared vars | yes |
| `mise.dev.toml` | Development environment variables (`MISE_ENV=dev`) | yes |
| `mise.local.toml` | Local developer overrides (env vars only) | no (gitignored) |
| `tasks/*.toml` | Task definitions, included via `[task_config].includes` | yes |

### `mise.toml`

- `[settings] experimental = true` — enables task usage flags and other experimental task features.
- `[tools]`:
  - `rust = "nightly"` — Rust toolchain (edition 2024 workspace).
  - `"cargo:cargo-nextest" = "latest"` — test runner.
- `[task_config].includes` — all files under `tasks/`.

### `mise.dev.toml`

Development environment (applied when the `dev` environment is active):

```toml
[env]
RUST_LOG = "info,stop_core=debug,stop_gui=debug"
JEV_API_BASE_URL = "http://localhost:8080"
OPENROUTER_API_KEY = ""
```

- `RUST_LOG` — tracing filter for the workspace.
- `JEV_API_BASE_URL` — local JevK5 GPU instance.
- `OPENROUTER_API_KEY` — used by `stop-dataset` only; leave empty in versioned config.

### `mise.local.toml`

Machine-local overrides, never committed. Example:

```toml
[env]
CARGO_BUILD_JOBS = "2"
```

## Task Taxonomy

Tasks live in `tasks/` and follow a `category:target` naming scheme:

| File | Tasks | Behavior |
|------|-------|----------|
| `format.toml` | `fmt:rust`, `fmt:newline` | Formatting; `--check` flag (env `STOP_FMT_CHECK`) turns them into checks |
| `linting.toml` | `lint:crates` | `cargo clippy --workspace --all-targets`; `--fix` (env `STOP_LINT_FIX`) applies suggestions, `--allow-dirty` (env `STOP_LINT_FIX_DIRTY`) permits a dirty working directory |
| `check.toml` | `check:format`, `check:lint`, `check` | Aggregate checks; `check:format` runs `fmt:?*` with `STOP_FMT_CHECK=true` |
| `fixes.toml` | `fix:fmt`, `fix:lint`, `fix` | Aggregate fixes; `fix:lint` runs `lint:?*` with `STOP_LINT_FIX=true` |
| `tests.toml` | `test:crates`, `test` | `cargo nextest run --all-targets` |
| `build.toml` | `build` | `cargo build --workspace` |
| `dev.toml` | `dev:gui` | Runs the interactive demo (GUI binary) |
| `misc.toml` | `dataset:generate`, `bench:run`, `bench:eval-accuracy\|eval-latency` | Entry points wired to the dataset generator and benchmark binaries. `dataset:generate` passes `--noise-ratio 0.5` (noise transcripts, see `docs/INSTRUCTIONS.md` 4.3) |

Common entry points:

```bash
mise run fix     # apply all lint suggestions + formatting
mise run check   # format check + lint check (CI gate)
mise run test    # run all tests via nextest
mise run build   # release-free workspace build
```

## Environment Runnables

Environment variables are managed in three layers (later layers override earlier ones):

1. `mise.toml` — nothing environment-specific.
2. `mise.dev.toml` — development defaults (`RUST_LOG`, `SYSTEMONE_API_BASE_URL`, `SYSTEMONE_MODEL`, `OPENROUTER_API_KEY`).
3. `mise.local.toml` — per-developer overrides, gitignored.

Task-local `env` blocks (e.g. `STOP_FMT_CHECK`, `STOP_LINT_FIX`) are set via task usage flags and only exist for the duration of the task invocation.

Runtime entry points:

- `mise run dev:gui` — interactive demo (still a placeholder binary).
- `mise run dataset:generate` runs `generate-data` with `--count 250` and `--noise-ratio 0.5` (noise transcript step, `docs/INSTRUCTIONS.md` 4.3); `OPENROUTER_API_KEY` required.
- `mise run bench:run`, `mise run bench:eval-accuracy|eval-latency`

## Project Status

Workspace and core foundation:

- mise configuration (global, dev, local) and the task files listed above.
- Cargo workspace with four crates; `stop-core` carries the domain model.
- `stop-core`: `RoomState`, per-object decision types (`DeviceDecision` / `UtteranceDecision`), deterministic `apply_action_to_state` with safety caps, serde round-trip and clamp tests.

Inference and single-pass engine:

- `stop-core::engine`: `InferencePort` trait (native `async fn`, generic-only), `InferenceInput` (room state + utterance, no history — token minimization), `InferenceOutcome` (all device decisions + latency), `ProviderError`.
- `stop-core::executor`: `SinglePassExecutor` — exactly one inference call per utterance; the response already carries every object's decision (`null` = no change) plus absolute targets, applied with confidence gating (`MIN_ACTION_CONFIDENCE` / `MIN_VALUE_CONFIDENCE` = 0.5, low confidence or missing target = no change) and emergency-stop precedence. `UtteranceReport` / `ExecutionResult` output (latency feeds the GUI HUD and the benchmark raw output). `MultiPassExecutor` removed: multi-pass is not testable against the high-latency provider and is no longer needed (closed object set resolves several actions in one pass).
- `stop-core::systemone`: `SystemOneClient` against `POST {SYSTEMONE_API_BASE_URL}/v1/systemone` (documented System-One contract, JevK5 server-compatible). One pass = one request with 12 typed questions (noul/choice/score, limit 16): one action group per room object (`light_action`, `camera_action`, `insufflator_action`, `table_action`, each with a `null` = no-change option) plus conditional absolute targets (`brightness_target`, `light_mode_target`, `zoom_target`, `pressure_target`, `tilt_target`, `height_target`) and the global flags `emergency_stop` / `requires_sterile_confirm`. Table height is decidable (`SetTableHeight`, clamped 70-130 cm). Hard 15s request timeout mapping to `ProviderError::Timeout`. Optional `SYSTEMONE_API_KEY` bearer, `SYSTEMONE_MODEL` override (default `jev-latest`).
- Tests: `tests/executor.rs` (scripted `MockDecisionEngine` defined tests-only, exactly-one-call invariant, null semantics, confidence gating, emergency stop, error propagation) and `tests/systemone_http.rs` (wiremock-canned responses, request shape, per-object decode rules, optional targets, HTTP/parse/timeout error mapping). HTTP tests need no live instance; one `#[ignore]`d live round trip runs against `SYSTEMONE_API_BASE_URL`. `tests/state_delta.rs` covers serde round-trips and the safety clamps (incl. table height).

Dataset generation (`stop-dataset`):

- Dataset schema shared with `stop-benchmark`: `DatasetCase { id, scenario, model, initial_state, history }`, `HistoryEntry { raw_utterance, expected_output_state }` — one JSONL line per case. `model` records the generating model as provenance.
- `OpenRouterClient` (`OPENROUTER_API_KEY`, optional `OPENROUTER_MODEL` / `OPENROUTER_BASE_URL`) producing JSON-only chat completions; default model `inclusionai/ling-3.0-flash-vl:floor`.
- Single-call generator per `docs/INSTRUCTIONS.md` 4.3: one completion per case produces the transcript micro-segment with handlungsneutrale filler/noise utterances following a random utterance type sequence, plus per-utterance room-state prediction chained on the previous state (filler keeps the state, self-corrections revert to the corrected state). Predicted states are clamped to the safety envelope via the state setters.
- `generate-data` is the package main binary (`--count`, `--output`, `--scenarios` (default: the five laparoscopic scenarios `laparoscopic_cholecystectomy`, `laparoscopic_hernia_repair`, `laparoscopic_appendectomy`, `laparoscopic_sleeve_gastrectomy`, `laparoscopic_fundoplication`), `--utterances-per-case` (default 16), `--noise-ratio` (default 0.5; 0.0 = all commands), `--concurrency` (default 8), `--seed`): noise/command slots at the given ratio, randomly interleaved per case. Output appends to an existing JSONL and resumes: case ids already present are skipped. Failed cases retry up to 3 times with backoff honoring `Retry-After`.
- Tests cover the full pipeline over wiremock-canned OpenRouter responses, plus response-mapping tests.

Benchmark (`stop-benchmark`):

- `run-benchmark` executes every dataset utterance once against the live System-One provider (state chains within a case) and appends one raw JSONL line per utterance (`RawEntry`: expected vs predicted final state, wall-clock and single-pass latencies, error strings). Crash-safe: entries persist as they complete.
- `eval-accuracy` computes per-utterance state exact match, an action/no-change entry split (derived from expected states; predicted changes on no-change entries are false positives), and Sequence Exact Match.
- `eval-latency` computes P50/P95/P99 and mean per pass and per utterance.
- Evaluation works purely on the raw output; tests use a hand-written fixture with golden metrics and smoke-run both analyzer binaries.

Not yet implemented:

- Real dataset generation into `data/test_suite.jsonl` (needs a live `OPENROUTER_API_KEY`); `data/benchmark_results.jsonl` from live runs likewise.
- `eval-roc` (spec `docs/INSTRUCTIONS.md` 5.1 lists it): deferred — scoring is state-based and per-slot confidences are not persisted.
- `dev:gui` becomes a real windowed runnable; STT/audio seam.
