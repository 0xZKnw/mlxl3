# MLXL3 Desktop 2.0.5

**MTP now works with your sampling settings, and each recommended catalog model offers its official settings in one click.**

Desktop **2.0.5 / build 58**, bundling MLXL3 engine **2.0.12**.

## New

- **MTP is no longer greedy-only.** Before 2.0.5, turning MTP on locked temperature, top-k and the repetition penalty to greedy decoding. With engine 2.0.12, MTP verifies drafts with exact speculative sampling. Your sampling settings now apply with MTP on, and the replies follow the same distribution as without MTP. Greedy settings behave exactly as before.
- **Recommended settings for catalog models.** The generation inspector has a new **Recommended settings** menu. It appears only for the recommended models of the catalog, and only lists the settings published by each model's authors:
  - **Qwen3.6-35B-A3B:**
    - Thinking · general: temperature 1.0, top-k 20, top-p 0.95, presence 1.5;
    - Thinking · precise coding: 0.6, top-k 20, top-p 0.95;
    - Instruct · non-thinking: 0.7, top-k 20, top-p 0.8, presence 1.5.
  - **Qwen3.8-27B and Qwen3.8 Flash Next:**
    - Thinking: 1.0, top-k 20, top-p 0.95;
    - Instruct · non-thinking: 0.7, top-k 20, top-p 0.8, presence 1.5.
  - **LFM3 1.7B (8, 4.49 and 4 bpw):**
    - General: 0.1, top-k 50, repetition 1.1;
    - Math & STEM: 1.0, top-k 20, top-p 0.95.
  - **LFM3 1.7B Heretic NX and LFM3-VL 2B:** 0.1, top-k 50, repetition 1.1.
  - **LFM2.5-24B-A2B:** 0.1, top-k 50, repetition 1.05.
  
  Models you add yourself, outside the catalog, have no such menu. The menu shows which preset is active, or "Custom" when you changed a value. It is disabled during a generation and when sampling is locked.
- **Two new sampling controls: Top P and Presence.** Top P ranges from 0.05 to 1, and Presence from 0 to 2. Both are saved with conversations and profiles. Older conversations and profiles open with the neutral values (top-p 1, presence 0).

## Unchanged

- **DFlash2 stays greedy.** With DFlash2 on, the sampling controls stay locked as before.
- **Older engines keep MTP greedy.** With an engine that does not report sampled-MTP support, the app keeps MTP greedy as in 2.0.4. This includes engines before 2.0.12 and the direct-Metal engine.
- **MTP auto-tune is unchanged.** It still measures with greedy decoding.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Full Desktop suite:** `scripts/check-desktop.sh` passed on the release changes. Two earlier runs failed and are kept in the evidence:
  - one run was invalid because a source file changed during the build;
  - one failure came from a mistake in the new check itself, which waited for a send button that needs a draft.
  
  The Python/pytest part of the final run used the project's virtual environment, because the system Python has no pytest. Results: pytest 39, Studios 58, sandbox 25, protocol 29, workspace 44.
- **New check `tests/speculative-sampling-check.swift`:** 2,863 assertions passed. It covers:
  - a 648-state grid of MTP / DFlash2 / engine-capability combinations, checking which controls are locked and what the request sends;
  - a sampled MTP run against the test engine;
  - persistence of top-p and presence through save, relaunch and older snapshots;
  - the exact preset values for every catalog repository;
  - matching of local model names, including older `--variant` folder names.
  
  Two deliberate bugs were each caught: restoring the old MTP lock, and a wrong top-p in a Qwen preset.
- **Engine:** see the [engine 2.0.12 notes](../engine/release-engine-v2.0.12.md) for the sampling-law tests, the GPU parity runs on Qwen3.6-35B-A3B and Qwen3.8-27B, and the check of the released engine archive.
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum.
  - After mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.5 / 58.
  - Its `build-info.json` names source `3075413a5bc4`, no uncommitted changes, and engine 2.0.12 with the same `mlxl3` SHA-256 as the engine release.
  - The bundled engine's `runtime-info` reports 2.0.12.
- **Not verified:**
  - The new menu and sliders were not checked by hand in a running app with a real model; they are covered by the automated checks above.
  - No speed measurement of sampled MTP.
  - Qwen3.8 Flash Next with sampled MTP was not run on a GPU.
  - The Windows app is not part of this release.
  - SwiftUI behavior has no formal proof.
- **Hosted CI (added after publication)** on the release commit `a08acf560`:
  - Passed: Flash profile compatibility, EXL3 conversion contracts, native coverage inventory, `mlx-package`, `native (ubuntu-latest)`, the Kani allocator and speculative shards, and the Metal and loader contracts.
  - Failed or cancelled, exactly as on the previous `main` commit `9eabe8eda`:
    - `desktop-and-protocol` and `desktop-preferences`: the hosted Swift compiler times out type-checking `ChatComponents.swift:390`, a file this release does not touch;
    - the Kani `core`, `routes`, `flash` and main shards: an internal Kani compiler panic plus harness timeouts in files this release does not touch;
    - `native (macos-26)` and `guard-proofs`: cancelled, as on every recent `main` commit.
  - These failures were not introduced by this release, but they mean hosted CI is **not green**. The local checks above are the release verification.

## Source and download

- Compiled source `3075413a5bc41a6e10a9015c23f329dc4ee953e1`.
- `MLXL3-Desktop-v2.0.5-b58-Apple-Silicon.dmg` · **253,147,954 bytes**.
- SHA-256 `c29ca208e4d586fee57a3ceb332e455606416f7a18478e70e380a291001e5012`.
- `MLXL3-Desktop-v2.0.5-b58-Apple-Silicon.dmg.sha256` · 109 bytes, SHA-256 `63939f313f55f6d1cc49c4f18e63fe29b8b331aa6e1fbb5d4a54c82eb0c0d935`.
- Bundled engine: [engine 2.0.12](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.12) (`mlxl3` SHA-256 `3574c82d…828adf`). The Swift app was built with the Command Line Tools macOS 26.5 SDK.
- Detailed records: `opti.md` (entry `MTP-SAMPLING-20261116`) and `docs/measurements/mtp-sampling-20261116/`.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the feature request and the release decision. Thanks to [Qwen](https://huggingface.co/Qwen) and [Liquid AI](https://huggingface.co/LiquidAI) for the models and the recommended settings published on their model cards, to [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [MTPLX](https://github.com/youssofal/mtplx), [TensorFold](https://github.com/ashhart/TensorFold) and [SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, model format, MTP research and math rendering.
