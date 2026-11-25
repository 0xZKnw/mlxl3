# MLXL3 Engine 2.0.37

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. **Agent turns survive interruptions better**: a stopped or failed turn keeps the tool calls that had finished, a tool server lost to a stopped or timed-out call restarts by itself, and long prompts report their progress on the paths that used to stay silent.

## Fixed

- **A tool server is no longer lost after one failed call.** When a tool call timed out, was stopped (Stop during a command) or broke its connection, the engine dropped that MCP server and never reconnected it: every later call answered `MCP server unavailable: sandbox` until the model was reloaded. The server is now started again at its next call. If it cannot restart, each call says why (`it could not be restarted: …`). The restarted server is a new process: what the old one kept in memory (the JavaScript kernel's `$` values, background jobs) is gone, and the tool catalogue offered to the model is not read again.
- **An error answered by a tool server no longer disconnects it.** A JSON-RPC `error` reply (a quota, an invalid argument) is reported to the model and the connection is kept.
- **A stopped or failed turn keeps its finished tool calls.** `cancelled` and `error` events now carry `turn_context` when at least one tool result existed: every call that received its result, in order, closed by an empty assistant message, in the same form as a paused turn (`agent_yield`). A call that was written but never ran is left out. Without tool results the events are exactly what they were. JSON-schema turns never carry it.

## New

- **Progress while a long context is read, on every default path.** `generation_status` events with `Reading context · N / M tokens` (at most one every 5 seconds) are now also sent by the Inkling 2 prefill with its DSpark/DFlash2 drafter and by the Qwen prefill with an MTP head. Before, only the plain prefill reported it, so a long context looked frozen. Flash Next with an MTP head is unchanged and still silent.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security. The bridge protocol is unchanged (1): older Desktop versions ignore the new `turn_context` field. [Desktop 2.0.33](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.33) bundles this engine and uses it.

## Verification and limits

- **No test suite was re-run for this release.** The checks below were run on the sources of [#210](https://github.com/0xZKnw/mlxl3/pull/210) before it was merged, on an Apple M5, without loading a model.
- `cargo fmt --check`; `cargo clippy --all-targets -- -D warnings` with `mlx,chat`, `chat` and `direct-metal,chat`; `cargo test --no-default-features --features chat`: 587 passed, 0 failed, 13 ignored (library).
- New deterministic tests: a scripted MCP server (stop during a call then restart with another process, 1-second deadline then restart, dead process then restart, refusal keeping the same process, restart impossible reported at each call) and a scripted model for the turn loop (stop between two calls of one message, engine error after two calls then replay of the next turn, stop before any result, unreadable call, and 72 stop shapes that never leave a call without its result). 11 temporary mutants of this code were all detected.
- **Not verified:** any run with a real model. The progress events of the DSpark and MTP prefill were compiled and read, never executed (no test reaches them without a GPU). Real chat templates (Inkling, Qwen) were not rendered with the history of an interrupted turn. Kani was not run.
- CI of `main` had not finished when this release was published.

## Artifacts

| File | Size | SHA-256 |
|---|---|---|
| `MLXL3-Engine-v2.0.37-arm64.tar.gz` | ENGINE_SIZE bytes | `ENGINE_SHA` |
| `MLXL3-Engine-v2.0.37-arm64.tar.gz.sha256` | 100 bytes | |

Source: `SOURCE_SHA` (tag `engine-v2.0.37`). ENGINE_CHECKS

## Thanks

- [0xZKnw](https://github.com/0xZKnw) for the Agent fixes ([#210](https://github.com/0xZKnw/mlxl3/pull/210)).
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), whose scripted-adapter tests the turn-loop tests follow.
- Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [minijinja](https://github.com/mitsuhiko/minijinja).
