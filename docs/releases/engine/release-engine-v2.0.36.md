# MLXL3 Engine 2.0.36

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. **Long prompts on Inkling 2 Mini are read up to about 1.5× faster**, with the same output bit for bit, and the Agent loop can now take a new instruction between two tool calls.

## Performance

- **Inkling 2 Mini prefill.** The prompt is still read in chunks of 128 tokens, but the experts of up to 8 consecutive chunks are now computed in one batch instead of one chunk at a time. Everything else (routing, attention, caches) keeps its per-chunk shape, so logits, drafter features and caches are **identical bit for bit** to 2.0.35. This is now the default; `MLXL3_INKLING_MOE_GROUP=1` restores the previous per-chunk prefill.
- Measured on an Apple M5 (10-core GPU, 24 GB), MLX 0.32.2, `Inkling-2-Mini-EXL3-3.33bpw`, on battery:
  - In-process prefill harness, 1,024 tokens, best of 3 after warmup, alternating order: 180.7 → **316.2 tok/s** (5.67 s → 3.24 s).
  - `mlxl3-rs run`, prompt of about 1,135 tokens, fresh process per run, same generated text: 179–199 tok/s → **269–294 tok/s**; time to first token 5.7–6.3 s → **3.9–4.2 s**.
  - Prompts of 128 tokens or fewer, and the last partial chunk of any prompt, take the same path as before. Gains on short prompts are therefore small.
  - The first long prompt on a machine compiles new kernel variants once (about +1.7 s observed once).

## New

- **Agent guidance during a task.** When Desktop sends a new instruction while an Agent turn is running, the loop finishes the current tool call, stops before the next one with `stop_reason: "agent_yield"`, and replays only the tool calls that finished. Stop always takes priority over a queued instruction. The direct Metal bridge advertises `agent_yield_supported: false`, so Desktop delivers the instruction at the end of the turn there.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security. The bridge protocol is unchanged (1). [Desktop 2.0.32](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.32) bundles this engine.

## Verification and limits

- **No test suite was re-run for this release**; the checks below were run on the pull requests' sources before they were merged.
- Grouped prefill ([#208](https://github.com/0xZKnw/mlxl3/pull/208)): bit-for-bit equality on 19 chunkings (256 to 1,300 tokens, groups 2, 3, 4 and 8) for end-of-span logits, all-row features, every K/V and convolution state and 4 decode steps; prompt-cache resume identical to a cold prefill; 12 identical DSpark comparisons (last logits, drafter context, verified rows, 48 greedy draft tokens). Clippy strict `mlx,chat` passed; library suite 645 passed, 0 failed, 397 ignored after the rebase. **GPU tests were not re-run after the rebase.**
- Agent yield ([#204](https://github.com/0xZKnw/mlxl3/pull/204)): Rust tests for pausing before, between and after tool calls, unreadable calls and compacted history, plus a targeted Kani proof of Stop's priority at the tool boundary.
- **Not verified:** the grouped prefill through the Desktop bridge over multi-turn conversations with prompt cache (about 8,000 tokens), cancellation in the middle of a group, the DFlash2 drafter end to end (DSpark only), peak memory, mains power.

## Artifacts

To be completed after the build.

## Thanks

- [HENK0O](https://github.com/HENK0O) for the Agent guidance loop and its direct Metal fallback ([#204](https://github.com/0xZKnw/mlxl3/pull/204)).
- [0xZKnw](https://github.com/0xZKnw) for the grouped Inkling prefill ([#208](https://github.com/0xZKnw/mlxl3/pull/208)).
- Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [minijinja](https://github.com/mitsuhiko/minijinja).
