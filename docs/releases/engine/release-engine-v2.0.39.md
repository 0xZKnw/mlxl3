# MLXL3 Engine 2.0.39

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. **Inkling 2 no longer reads a whole Agent conversation again in the middle of a turn**, and an Agent step can now go on from the state the model decoded with instead of reading its own output again.

## Fixed

- **The Inkling 2 prompt cache no longer loses its snapshot.** With the DSpark or DFlash2 drafter, the state saved at the end of a prefill could exceed the 256 MiB prompt cache budget and was then dropped without a word: the next step or turn read the whole conversation again from the first token. Its size depended on how the prompt had been split into prefill calls, not on what the state holds: the drafter's context was saved as a view that also kept the rows added by the last prefill (above about 2,650 kept rows the snapshot no longer fit). In the conversation this was found in, one step of each Agent turn started from zero (9,619 tokens read again in a 117 s turn).
  - The drafter's context is now copied to its own rows, and each sliding-attention layer saves its 1,023 live rows instead of its whole buffer. The snapshot is a constant ≈ 215 MB whatever the prompt length and split. No computed value changes: a resumed prompt is still identical to the bit to a cold prefill.
  - A snapshot refused by the budget is now reported on stderr with the size of its parts. `MLXL3_PROMPT_CACHE_TRACE=1` reports every capture; `MLXL3_CACHE_COMPACTION=0` keeps the previous layout for comparisons.

## New

- **Live prompt cache (`live_prompt_cache`, Inkling 2 with a DSpark/DFlash2 session; off unless requested).** Until now each Agent step put the model's answer back in the prompt in a canonical form, without its reasoning, so the tokens the model had just decoded were never a prefix of the next prompt and were read again. With `"live_prompt_cache": true` in a `generate` request:
  - each step that calls tools keeps its text exactly as the model wrote it (reasoning and call) in the dialogue and in `turn_context` (a `raw` field next to `content` and `tool_calls`), and a final answer stored raw is written back as it was instead of being wrapped in a second text message;
  - when the next prompt starts with exactly the previous prompt and the tokens the model committed while decoding, the prefill starts at the end of that live state: only what is new (tool results, the next user message) is read. Otherwise the engine resumes the saved snapshot as before.
  - The `ready` event announces `live_prompt_cache_supported`; step and turn statistics report `live_prompt_tokens`.
  - **This is not the bit-exact contract of the default cache.** Rows computed while decoding do not have the shapes of prefilled rows: in the oracle below the last logits differ from a cold prefill by 0.8% to 3.0% (relative L2), with the same top token. The model also sees its own reasoning from earlier steps and turns, as Codex, Claude and DeepSeek's tool loops do. Without the field nothing changes, prompts included.

## Performance

Inkling 2 Mini EXL3 3.33 bpw with its DFlash2 drafter, Apple M5 24 GB, real bridge, other sessions running on the machine. Token counts are exact; times are single passes and indicative.

- **Snapshot fix**, conversation without tools, 5 growing turns, greedy: turns that resume the cache went from 2 of 4 to 4 of 4, and the prefill of the five turns from 43.0 s (2.0.38) to 16.2 s. Token hashes and texts are identical with the cache off, with 2.0.38 and with this engine.
- **Live prompt cache**, two Agent turns with real tools (fix a failing Python test, then add a function and its test), temperature 1, three passes per mode: every step after the first prompt went on from the live state (8/8, 9/9, 8/8). After the first prompt, 92 to 139 tokens were read per step instead of 259 to 314, and the prefill took 10.5 s and 11.6 s (two passes on this binary) instead of 25.1 s to 29.5 s. The task passed in all six passes. The two modes do not follow the same path (sampling, reasoning kept), so whole-turn times are not compared.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security. The bridge protocol is unchanged (1): a client that does not send `live_prompt_cache` gets the snapshot fix and nothing else. [Desktop 2.1.2](https://github.com/0xZKnw/mlxl3/releases/tag/v2.1.2) bundles this engine and sends the field in Agent turns.

## Verification and limits

Checks run on these sources on an Apple M5, before publication. **The final verification pass was stopped at the maintainer's request to publish**, so the list of what did not run is longer than usual.

- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` with `mlx,chat`, `chat` and `direct-metal,chat` on the released sources.
- Snapshot fix: both GPU resume tests on the real model (resumed prompt identical to the bit to a cold prefill; snapshot size independent of the call before its capture; a shifted copy detected), two temporary mutants restoring the previous behaviour detected, full suites at that point (`chat`: 710 passed, 0 failed, 15 ignored; `mlx,chat`: 808 passed, 0 failed, 418 ignored).
- Live prompt cache: 65 targeted CPU tests passed before the last change to the prefill spans (not rerun after it); GPU oracle on the real model, live state against a cold prefill of the same prompt, 4 cases, same top token, relative L2 error 0.8% to 3.0% (criterion: 5% at most); a state built from a text shifted by one token is 12 to 14 times further when 8 tokens follow, but only 1.4 and 3.1 times when 200 follow, so the criterion registered beforehand for 200 tokens was **not** met. Without the field, a 5-turn greedy conversation gives the token hashes of 2.0.38's fixed cache, and the bit-exact DSpark resume test still passes.
- **Not run on the released sources:** the full `cargo test` suites, the grouped-prefill and plain Inkling resume GPU tests, mutants of the live prompt cache guards. Kani was not run. CI of `main` for this source had not finished when this release was published.
- **Not verified:** the live prompt cache without a drafter session (it resumes the snapshot), any other model (Qwen, LFM, Flash Next are unchanged and were not measured), answer quality with reasoning kept beyond the trial task, conversations stored by older clients (their first turn is read from the start).
- Scripts, reports and logs: `docs/measurements/prompt-cache-budget-2026-11-26/` and `docs/measurements/agent-live-cache-2026-11-26/`.

## Artifacts

| File | Size | SHA-256 |
|---|---|---|
| `MLXL3-Engine-v2.0.39-arm64.tar.gz` | ARCHIVE_SIZE bytes | `ARCHIVE_SHA` |
| `MLXL3-Engine-v2.0.39-arm64.tar.gz.sha256` | SUM_SIZE bytes | |

Source: `SOURCE_SHA` (tag `engine-v2.0.39`). ARCHIVE_CHECKS

## Thanks

- [0xZKnw](https://github.com/0xZKnw) for the report of the 68% cache figure, the decision to go on from the decoded state and the request to publish.
- OpenAI's [Unrolling the Codex agent loop](https://openai.com/index/unrolling-the-codex-agent-loop/), Anthropic's [prompt caching guide](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) and DeepSeek's [thinking mode guide](https://api-docs.deepseek.com/guides/thinking_mode), whose exact-prefix prompts and reasoning kept across tool calls this follows.
- Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [minijinja](https://github.com/mitsuhiko/minijinja).
