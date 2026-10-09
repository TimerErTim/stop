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
| `misc.toml` | `dataset:generate`, `bench:run`, `bench:eval-accuracy\|eval-roc\|eval-latency` | Phase 3/4 entry points wired to the binaries; tasks exist now, binaries land in their phases. `dataset:generate` passes `--include-noise` (noise transcripts, see `docs/INSTRUCTIONS.md` 4.3) |

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
2. `mise.dev.toml` — development defaults (`RUST_LOG`, `JEV_API_BASE_URL`, `OPENROUTER_API_KEY`).
3. `mise.local.toml` — per-developer overrides, gitignored.

Task-local `env` blocks (e.g. `STOP_FMT_CHECK`, `STOP_LINT_FIX`) are set via task usage flags and only exist for the duration of the task invocation.

Runtime entry points (grown per phase):

- Phase 1: `mise run dev:gui`
- Phase 3: `mise run dataset:generate` runs `generate-data` with `--count 250` and `--include-noise` (noise transcript step, `docs/INSTRUCTIONS.md` 4.3); `OPENROUTER_API_KEY` required.
- Phase 4: `mise run bench:run`, `mise run bench:eval-accuracy|eval-roc|eval-latency`

## Phase 1 Scope and Open Points

Implemented in phase 1:

- mise configuration (global, dev, local) and the task files listed above.
- Cargo workspace with four crates; `stop-core` carries the domain model.
- `stop-core`: `RoomState`, decision slots, deterministic `apply_action_to_state` with safety caps, serde round-trip and clamp tests.

Not yet implemented (later phases):

- Phase 2: `DecisionEngineProvider` trait, mock engine, multi-pass executor, JevK5 HTTP client. HTTP client tests will need `JEV_API_BASE_URL` pointing at a live instance or a mock.
- Phase 3: `dataset:generate` task wiring the `generate-data` binary (`OPENROUTER_API_KEY` required). Noise transcript step added to the spec (`docs/INSTRUCTIONS.md` 4.3).
- Phase 4: `bench:*` tasks (`run-benchmark`, `eval-accuracy`, `eval-roc`, `eval-latency`).
- Phase 5: `dev:gui` becomes a real windowed runnable; STT/audio seam.
