# MLXL3 Engine 2.0.34

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. One fix: **`/v1/chat/completions` keeps `reasoning_content` and accepts empty `tool_calls`**.

## Fixed

Both come from [#198](https://github.com/0xZKnw/mlxl3/pull/198), for issue [#195](https://github.com/0xZKnw/mlxl3/issues/195).

- **`reasoning_content` was silently dropped.** Thinking that a client sent back on an earlier assistant turn never reached the chat template. It is now forwarded as written, and the template trims it as the reference `apply_chat_template` does.
- **`"tool_calls": null` and `"tool_calls": []` were rejected with a 400**, as was `"function_call": null`. The OpenAI Python SDK sends them on plain text turns. They now mean "no tool call".
- A real tool call in the request is still rejected with the same 400. A `reasoning_content` that is not a string, or one on a `system`, `user` or `tool` turn, is rejected with a 400 instead of being dropped; `null` is accepted.

Prompt tokens on the issue's history, from the pull request (`max_tokens: 1`):

| Request | 2.0.31 | 2.0.34 | Hugging Face reference |
|---|---|---|---|
| plain history | 41 | 41 | 41 |
| assistant turn with `reasoning_content` | 41 | 81 | 81 |
| `reasoning_content` and `"tool_calls": null` | 400 | 81 | 81 |
| `"tool_calls": []` | 400 | 41 | 41 |
| a real tool call | 400 | 400 | n/a |

Nothing else changes in the engine since 2.0.33.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security. The bridge protocol is unchanged (1). [Desktop 2.0.30](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.30) bundles this engine.

## Verification and limits

- The fix was verified in its pull request: new tests in `native/src/openai.rs` that fail before the change and pass after it, and the pull request's CI passed.
- **The test suites, Clippy and any model run were not repeated for this release** (published quickly at the maintainer's request); CI runs the suites on `main`.
- **Not verified:** the server driven by a real OpenAI client on this build, Kani for this change beyond the pull request's CI.

## Artifacts

| File | Size | SHA-256 |
|---|---|---|
| `MLXL3-Engine-v2.0.34-arm64.tar.gz` | 69,298,110 bytes | `ec918e639116ed3842a6834fa198fe3bb32c9edf1bfaa7fbdd205d7f84e20e93` |
| `MLXL3-Engine-v2.0.34-arm64.tar.gz.sha256` | 100 bytes | |

Source: `6836e664b` (tag `engine-v2.0.34`). The archive was extracted into a new directory: `mlxl3` passes a strict signature check and is arm64, its manifest reports 2.0.34 and `engine-v2.0.34`, and its five files are identical to those bundled in Desktop 2.0.30. The binary itself was not started for this release.

## Thanks

- [EnriqueGalindo](https://github.com/EnriqueGalindo) for the report and the reproduction in [#195](https://github.com/0xZKnw/mlxl3/issues/195).
- Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [minijinja](https://github.com/mitsuhiko/minijinja).
