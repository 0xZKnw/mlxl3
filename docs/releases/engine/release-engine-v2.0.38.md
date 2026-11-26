# MLXL3 Engine 2.0.38

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. **A long Agent conversation no longer ends on the context limit**: when earlier turns fill the context, the model summarizes them and the turn continues.

## New

- **Earlier turns are summarized when they fill the context.** Until now the turn loop only trimmed old tool outputs and summarized the older steps of the *current* turn; a turn that started in a context already filled by earlier questions and answers ended on `Context limit reached`. Now, when the prompt is above 75% of the context after trimming and the current turn has nothing to give up, the oldest turns are summarized by the model (at most 3,072 tokens, nothing streamed) and replaced by that summary, written at the head of the first user message kept.
  - The cut is always before a user message with text, so a turn is summarized whole. The newest turns that fit in 20% of the context are kept word for word; the last user message always is. If the summary step of the preferred cut does not fit the context, an older cut is tried.
  - Nothing changes when the summary is empty, cut, a tool call, or not shorter than what it replaces: the messages are kept and the status says so.
  - It needs tools to be offered in the request (the Agent loop) and at least 6,144 tokens left in the turn's budget. JSON-schema turns are never summarized.
- **`history_summary` bridge event.** After such a summary the engine sends `{"type":"history_summary","request_id":…,"kept_from":N,"prefix":"…"}`: `kept_from` is the index, in the request's `messages`, of the first message kept, and `prefix` is the exact text now at the head of that message. A client that stores it sends, from the next turn on, `prefix` + that message and the following ones, and no longer the summarized ones. A stopped or failed turn after a summary reports only its own tool exchanges in `turn_context`.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security. The bridge protocol is unchanged (1): older Desktop versions ignore the new event and keep sending the whole conversation, which the engine then summarizes again at each turn. [Desktop 2.1.0](https://github.com/0xZKnw/mlxl3/releases/tag/v2.1.0) bundles this engine and stores the summary.

## Verification and limits

Checks run on these sources on an Apple M5, before publication.

- `cargo fmt --check`; `cargo clippy --all-targets -- -D warnings` with `mlx,chat`, `chat` and `direct-metal,chat`; `cargo test --no-default-features --features chat`: 593 passed, 0 failed, 13 ignored (library).
- New deterministic tests with a scripted model: earlier turns summarized when they fill the context, with and without tool history; a conversation that never used a tool; a summary that cannot be used changes nothing; a turn interrupted after a summary keeps only its own exchanges; a single turn or a small conversation is never summarized; the cut and its fallbacks to older cuts. 9 temporary mutants of this code were all detected.
- **One real-model run with the binary of this archive** and the sandbox server of Desktop 2.1.0: Inkling 2 Mini EXL3 3.33 bpw, temperature 1, 16,384-token context, six fabricated exchanges then a question. One `history_summary` (`kept_from` 11, ten messages replaced, prompt 13,139 → 4,331 tokens), `complete` with `eos`; the next turn, rebuilt from the event, had 4,671 prompt tokens and completed. The summary kept the two planted facts and both answers gave them. Script and report: `docs/measurements/agent-history-summary-2026-11-26/`.
- **Not verified:** any other model (Qwen, Flash Next), a cut across images or tool history on a real model, the quality of summaries in general (one run, one conversation, temperature 1), a second summary in the same request on a real model. Kani was not run.
- CI of `main` for this source had not finished when these notes were written; see the Desktop release for the state read before tagging.

## Artifacts

| File | Size | SHA-256 |
|---|---|---|
| `MLXL3-Engine-v2.0.38-arm64.tar.gz` | 69,321,677 bytes | `b3ab6b63223c62dd26aa14480d81fed3476b50c795d8df41765e074001636a47` |
| `MLXL3-Engine-v2.0.38-arm64.tar.gz.sha256` | 100 bytes | |

Source: `20473fd12` (tag `engine-v2.0.38`). The archive was extracted into a new directory: `mlxl3` passes a strict signature check and is arm64 (SHA-256 `7ca8ce1444b2ea5f46d90c58fee59190d175e9a513f241e30b4b6867f493bbfb`), its manifest reports 2.0.38 and `engine-v2.0.38`, and its five files are identical to those bundled in Desktop 2.1.0. That binary is the one used in the real-model run above.

## Thanks

- [0xZKnw](https://github.com/0xZKnw) for the summary of earlier turns.
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), whose summary of a long turn this extends to earlier turns.
- Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [minijinja](https://github.com/mitsuhiko/minijinja).
