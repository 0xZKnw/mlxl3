# MLXL3 Desktop 1.4.3 — build 25

This update bundles engine 1.4.3 and integrates [PR #31](https://github.com/0xZKnw/mlxl3/pull/31), [PR #32](https://github.com/0xZKnw/mlxl3/pull/32) and [PR #33](https://github.com/0xZKnw/mlxl3/pull/33).

## New features

- **DFlash2 setup is available in Generation settings** for Qwen3.6-35B-A3B and Qwen3.8-27B. One switch downloads the matching pinned Q4 draft, checks its files and loads it. Greedy settings apply automatically; OFF releases the draft. Target verification remains mandatory.
- **Tune DFlash2** compares baseline, Auto (2/5 proposals), fixed 2 and fixed 7, using code and French prompts. It saves a profile for this model/draft/runtime/context/Mac and retains baseline unless an eligible mode exceeds it by 3%. Invalid or cancelled tests preserve the previous setting.
- MTP and DFlash2 each have a toolbar Tune button with progress, phase, stop and result details. Tuning keeps the conversation visible and preserves an already-open inspector.
- **Memory saving**, optional and off by default, releases the saved prompt checkpoint and unused buffers after generation or Tune. Weights and history stay loaded; the following message may recompute more context.
- The inspector offers a conservative **context recommendation** based on engine memory, model cache geometry and physical RAM. Applying it requires the user's Save action.

## Engine changes and defaults

Saved prefill checkpoints now count all retained buffers, including draft/head state, logits and history, against a **256 MiB default budget**. Wasteful views are detached by an exact copy at checkpoint boundaries. Kernel-cache hits avoid copying their keys. See the [engine changelog](release-engine-v1.4.3.md) for options and numerical scope.

Adaptive MTP, packed embeddings, additional Qwen27 execution paths and 35B context-copy drafting remain experimental and off by default. The target's precision and mandatory speculative verification are retained. MTP and DFlash2 cannot be enabled simultaneously.

## Performance scope

No new global tokens/s gain or superiority over MTP is established. The key-view CPU microbenchmark measured about 25% less lookup time on its archived harness; this is not a 25% generation boost. The later 35B production comparison had exact outputs but excessive timing drift. Historical 27B RAM measurements concern specific settings and prompts, and are not added together. [DFlash2 measurements](dflash2-key-view-tps-isolated-2026-10-10.md), [memory measurements](adaptive-memory-qwen-20261009.md).

## Fixes

DFlash2 tuning forwards the Memory saving policy through the production Desktop/NDJSON boundary. Calibration identities distinguish allocator policy and experimental execution options. Preparation handles silent, malformed and failed replies; changing the target invalidates stale draft selections.

## Install and compatibility

Download **MLXL3-Desktop-v1.4.3-b25-Apple-Silicon.dmg**. Requires Apple Silicon and macOS 26.2+. MLX 0.32.2 and engine 1.4.3 are bundled; model and draft weights are not included. The app is ad-hoc signed, not notarized. The engine updater remains independent; update Desktop to use the new controls.

This release is the latest Desktop update. The separate engine release is published with latest=false so it does not replace the DMG in the app updater. Installed copies and user histories are not replaced by the build process.

## Verification and limits

Validation covers formatting, strict lint, shipped mlx/chat compilation, Rust CPU tests, Python and the complete Desktop fixture suite, including one-click setup, cancellation, saved calibration, memory-policy wiring and rendered Tune controls. Physical 35B checks compare logits, states and caches with independent exact oracles and exercise the production bridge. Commands, counts, hashes and CI identities are in [the release validation report](release-v1.4.3-validation.md).

The latest local campaign deliberately excludes 27B GPU inference; its availability uses pinned-geometry checks and earlier qualification. Source-level checks are bounded CPU verification, not a proof of MLX/Metal/concurrency or universal generation correctness. C++/Swift/Python formal-verifier limitations and the inherited direct-Q4 fixture failure remain recorded in the linked evidence.

## Thanks

- [@HENK0O](https://github.com/HENK0O) for the memory, context-advice and adaptive-MTP work in [#31](https://github.com/0xZKnw/mlxl3/pull/31) and [#32](https://github.com/0xZKnw/mlxl3/pull/32).
- [@0xZKnw](https://github.com/0xZKnw) for the MLXL3 engine/Desktop integration and DFlash2 investigation.
- [z-lab / DFlash](https://github.com/z-lab/dflash), [IncoAI](https://huggingface.co/IncoAI) and [Splash / Goekdeniz](https://huggingface.co/Goekdeniz-Guelmez) for the draft work and conversion ecosystem; [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) and [mlx-node](https://github.com/mlx-node/mlx-node) for implementations examined during optimization. Their published results are not presented as MLXL3 measurements.
- [MLX](https://github.com/ml-explore/mlx), [MTPLX](https://github.com/youssofal/MTPLX) and [TensorFold](https://github.com/ashhart/TensorFold) for upstream inference work.
