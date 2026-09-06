# v1.0.0 validation and compatibility

This is a hardening release, not a claim that every EXL3 architecture, Mac or
Markdown extension has been tested. The original findings are preserved in
[the pre-fix audit](audit-v1-2026-09-06.md). No speculative decoding was added.

## Audit resolution

| Findings | Change and check |
| --- | --- |
| A1 | Corrupt/unsupported histories cannot be overwritten silently. Visible errors, previous-file backup, explicit recovery preserving unreadable files, ordered atomic writes. Injected-corruption and write-order checks. |
| A2–A6 | Dead engines are not Ready; active-chat deletion cancels; instance IDs discard stale process events; model import/load guarded during generation; model identity comes from the loaded engine; interrupted tools are finalized. Fake-engine lifecycle checks. |
| B1–B4 | Execute only structured final-channel tool calls, never examples/reasoning. Strict JSON Schema validation without remote schema resolution. Cancellation checks between tools and during protocol waits. Native Gemma parser. Structured assistant/tool transcript persisted and replayed. Regression fixtures. |
| B5–B7 | Tool result token budget, visible engine TTFT versus whole-turn latency, bounded stdio queues/messages/deadlines/pagination. Privacy text distinguishes local inference from network-capable tools. A remote action already accepted cannot be undone; HTTP cancellation can take up to its socket timeout. |
| C1–C4 | Validate descriptor, safetensors header inventory, duplicate tensors, shards, ordinary weights, expert count and kernel shapes/dtypes. Unsupported mapped MoE K=7 is rejected, not silently reinterpreted. Dense K=7 remains supported. |
| C2, C5 | Failed prefill invalidates mutated state. Session budgets pruned on success/error/cancel, including oversized active sessions between turns. Compiled-module ownership and benchmark cleanup release the preceding model. |
| C6–C7 | Shared CLI/GUI Gemma attention default; environment overrides and metadata hashes recorded by benchmark; compatibility scope below. |
| D1–D4 | Bounded lazy layout for long code, linear chunk splitting, virtualized message list, serialized compact background saves, token fade including thinking, Reduce Motion, variable fences, preserved long-table headers, multiline lexical context, currency/code/math fixtures and bounded LaTeX cache. |
| D5 | Ctrl+Enter inserts at cursor, per-chat drafts within the session, full-content search, import/export, undo last deletion, recovery, model-matched widget stats and formula accessibility text. |
| E1–E4 | HTTP/Xet byte progress, atomic locked registry, shard/tokenizer validation before registration, rollback if registration fails, common CLI/GUI variant-only downloader, explicit folder relocation. Interrupted-download fixture covers resume and no overwrite. |
| E5 | Refresh/TTL, incremental search limit, multi-term filtering, persisted interrupted jobs, explicit partial-file cleanup, token sign-in in Settings, config/architecture preflight before large shards and disk budget checks. Remote metadata cannot prove kernel compatibility or peak RAM; load-time validation remains necessary. |
| F1–F2 | Build-aware update comparison, background staging, drained output and bounded helper wait, frozen runtime smoke check before stopping the old app; retain previous app for recovery. |
| F3 | Ad-hoc signing retained by explicit release-owner choice. **No Developer ID, notarization or authenticated Apple publisher identity.** GitHub digest and bundle checks are not a substitute for notarization. |
| F4–F6 | Synchronized version/build, pinned release constraints and dependency manifest, physical-Mac regressions, protocol/desktop CI, benchmark metadata and reusable two-turn frozen-runtime smoke test. |

## Coverage

| Model/format | Text generation on this release source | Multi-turn | MCP | Device |
| --- | --- | --- | --- | --- |
| LFM2.5 2.6B EXL3 4 bpw | Tested | Two-turn project-code recall tested | Structured protocol fixtures; no live Exa request in this campaign | M5 |
| Qwen3.6 35B A3B EXL3 2.49 bpw | Tested | Cache regression fixtures | Qwen structured parser fixtures | M5 |
| Qwen3.8 27B EXL3 2.75 bpw | Tested | Cache regression fixtures | No live tool campaign | M5 |
| Gemma 4 26B A4B EXL3 | Prior-release text validation; no weights available for a new full-model campaign | State/kernel fixtures | Native parser fixture, not full-model live MCP validation | Prior M5 tests |
| LFM2.5 8B A1B EXL3 | Existing checkpoint/adapter coverage | Not rerun as a full-model conversation in this campaign | Not validated here | M5 |
| Other upstream architectures / Ling | Experimental until their actual checkpoint is tested | Not guaranteed | Not guaranteed | Not guaranteed |

Apple Silicon M1–M4 use the non-TensorOps fallback; physical cross-device testing
has not been performed for this release. Intel, Windows and Linux Desktop are
not supported. Image/audio input and mapped MoE K=7 are not supported.

## Reproduce

```sh
python -m pytest
scripts/check-desktop.sh
python scripts/smoke-release.py '/path/MLXL3 Desktop.app/Contents/Resources/runtime/mlxl3' MODEL
mlxl3 benchmark MODEL --prompt-tokens 128 --max-tokens 64 --repeats 5 --output results.json
```

The physical M5 campaign exercised three checkpoints sequentially without an
out-of-memory failure after the cleanup fix. A short battery-powered run gave
peaks of 1.92 GB (LFM 2.6B), 12.43 GB (Qwen MoE) and 13.07 GB (Qwen dense).
These are MLX peak allocations, not total system memory. Timing is not a clean
before/after comparison: some runs overlapped compilation/tests, and power and
temperature were not controlled. No performance improvement is claimed here.

The desktop regression fixture covers 300k-character code, long tables, lifecycle
failures, draft switching, corrupted storage and interrupted tool states. A
separate `--ui-preview --ui-preview-stress` scenario streams a large HTML file
without loading a model or touching real history. The benchmark measures engine
callbacks, not displayed frame rate; two repetitions do not establish a P95.

## Deliberate limits

- Markdown highlighting is a display lexer, not a compiler or a complete
  CommonMark/LaTeX implementation. Unsupported math falls back to readable text.
- Histories remain a compact JSON archive with serialized writes, not an
  incremental database. Extremely large archives still require proportional
  decode/encode memory; drafts and Undo are session-local.
- No exhaustive VoiceOver, clean-user install, automatic-update rollback after a
  GUI-only crash, long-duration soak, or multi-Mac certification is claimed.
- A local MCP subprocess is trusted code with the user's permissions; it is not
  sandboxed by MLXL3. MCP is off by default and activation persists by choice.

Use the retained backup app if a new GUI fails after installation. Export chats
before experimenting with another build; do not manually discard unreadable
history copies until recovery is complete.
