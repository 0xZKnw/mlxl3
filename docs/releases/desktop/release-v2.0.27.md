# MLXL3 Desktop 2.0.27

**Inkling 2 Mini answers faster by default: Desktop downloads its DFlash2 drafter and turns speculative decoding on.**

Desktop **2.0.27 / build 80** bundles MLXL3 engine **2.0.32**: the same files as [`engine-v2.0.32`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.32).

## New

### Inkling 2 Mini: DFlash2 on by default

- The first time Inkling 2 Mini loads, Desktop **downloads its DFlash2 drafter in the background** ([`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2), about **990 MB**, pinned revision and checksums). Answers decode normally meanwhile.
- Once the drafter is installed, the acceleration **turns itself on**. The DSpark control in the toolbar shows the progress, an error with a Retry button, and the acceptance of the latest answer.
- **Your choice wins:** turning the switch off cancels the download and keeps the acceleration off; turning it on starts the download if the drafter is missing. If you never touched the switch, it follows the engine.
- The output is unchanged in law: the target model verifies every proposal, greedy answers are identical to plain decoding, and sampled answers keep the model's sampling distribution.

### Faster on predictable text

Measured through the bridge on a MacBook Pro M5 (10-core GPU, 24 GB), Inkling 2 Mini EXL3 3.33 bpw, temperature 1, an English coding prompt, one turn of 1,200 tokens, 4 passes: **37.6 tokens/s with DFlash2 against 28.4 without** (passes 32.9–40.2 against 28.3–28.5). French prose, long contexts and later turns were **not measured**; where the drafter predicts poorly the engine falls back to plain steps. Details in the [engine 2.0.32 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.32).

## Also included from `main`

Merged since 2.0.26 and shipped here, not re-verified for this release beyond the checks below: Studio Lab's native workspace with model usage, generation history, response timing and activity ([#194](https://github.com/0xZKnw/mlxl3/pull/194)); dictated drafts kept across pauses and punctuation spacing by language ([#193](https://github.com/0xZKnw/mlxl3/pull/193)); native reasoning levels per model in Studio ([#192](https://github.com/0xZKnw/mlxl3/pull/192)); faster Qwen Image rendering at 512 px with a lower memory peak ([#197](https://github.com/0xZKnw/mlxl3/pull/197)).

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The Windows app is unchanged in this release (it runs the ExLlamaV3 engine, which has no DFlash2 drafter).

## Unchanged

- Models, conversations, pins, projects, Agent projects and settings.
- Agent mode and requests with tools decode without the drafter, as before.
- With top-k, top-p or penalties, the drafter's proposals are checked with the earlier point rule (exact, lower acceptance).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Engine checks:** see the [engine 2.0.32 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.32) (bit-identical verification, sampling law, test suites, default path through the bridge).
- **App logic:** the switch rule (user choice first, else the engine default) and the download trigger (Inkling 2 Mini only, engine 2.0.32+, never after the user turned the switch off) are pure functions with a truth-table check in `tests/speculative-sampling-check.swift`; the app builds with them.
- **Desktop suite (`scripts/check-desktop.sh`): not run to completion for this release.** A run before the merge with `main` stopped on an unrelated home-texture check already fixed on `main`; the run on the release source was cut short to publish quickly. The new Swift checks were therefore **not executed** here.
- **Not verified:**
  - the real 990 MB drafter download from the app (the engine's installer was exercised with the pinned files already present);
  - the DSpark panel's progress, error and retry states driven by hand;
  - the app driven by hand on a real model;
  - what `main` merged since 2.0.26 (listed above).

## Source and download

- Source: [`c806094f6`](https://github.com/0xZKnw/mlxl3/commit/c806094f6fcd2379468d0ba5e13adaba9f96c919) (tag `v2.0.27`).
- `MLXL3-Desktop-v2.0.27-b80-Apple-Silicon.dmg`: 253,375,899 bytes, SHA-256 `1a50f34ea831a3c138e12cc5fd419677a348713a882478e5d801835e58a8632e`.
- `MLXL3-Desktop-v2.0.27-b80-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.32, the same files as [`engine-v2.0.32`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.32).
- This DMG: `hdiutil verify` valid, `codesign --deep --strict` passed on the app, version 2.0.27 / build 80, built with the macOS 26.5 SDK; `build-info` has commit `c806094f6` with no tracked changes; the five runtime files are identical to the published engine archive.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for asking that the drafter be downloaded automatically and on by default.
- [IncoAI](https://huggingface.co/incoai) for the [Inkling 2 Mini DFlash2 drafter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) and [Splash](https://github.com/incoai/splash); [SGLang](https://github.com/sgl-project/sglang) and [MTPLX](https://github.com/youssofal/MTPLX) for the reference implementation and the exact-verification ideas credited in the engine notes.
- Thanks to [HENK0O](https://github.com/HENK0O) for Studio Lab ([#194](https://github.com/0xZKnw/mlxl3/pull/194)), dictation ([#193](https://github.com/0xZKnw/mlxl3/pull/193)), per-model reasoning ([#192](https://github.com/0xZKnw/mlxl3/pull/192)) and Qwen Image ([#197](https://github.com/0xZKnw/mlxl3/pull/197)).
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
