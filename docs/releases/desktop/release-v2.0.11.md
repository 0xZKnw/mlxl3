# MLXL3 Desktop 2.0.11

**Adds Inkling 2 Mini to the recommended models, with its DSpark drafter.**

Desktop **2.0.11 / build 64** bundles MLXL3 engine **2.0.20**.

## New

### Inkling 2 Mini in the recommended models

- **A new family, "Inkling".** [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) from Thinking Machines is a 26B looped mixture-of-experts model (about 3.4B parameters active per token). The card installs [`0xzknw/Inkling-2-Mini-EXL3-3.33bpw`](https://huggingface.co/0xzknw/Inkling-2-Mini-EXL3-3.33bpw), pinned to revision `d1248f8a`: a 14.8 GB download with experts at 3.33 bits per weight.
- **DSpark is included.** The download contains the model's drafter, and engine 2.0.20 uses it by default. There is nothing to configure: with the recommended sampling preset it was 10–25% faster than plain decoding on an Apple M5, and the sampled text follows the same law.
- **Recommended sampling preset.** The model's own settings are selected automatically: temperature 1.0 over the whole vocabulary.
- **Thinking.** Thinking and answer are shown separately, as for the other thinking models. Turning thinking off asks the model for no reasoning (`reasoning_effort` 0).

### Engine 2.0.20

The bundled engine adds the Inkling architecture and changes nothing for the other models: greedy token hashes are identical to engine 2.0.19 on Qwen3.6-35B-A3B, Qwen3.8 Flash Next and d1-2B. See the [engine 2.0.20 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.20).

## Performance

Apple M5 MacBook with 24 GB, AC power, 8192-token context, three prompts, single passes:

| Mode | Plain decoding | DSpark (default) |
|---|---|---|
| Sampling, temperature 1.0 (the app's preset) | 17.0–18.1 tok/s | 18.1–22.5 tok/s |
| Greedy | 19.6–19.7 tok/s | 23.4–25.9 tok/s |

The first reply after loading starts in 3.4–3.6 s.

## Limits for Inkling 2 Mini

- **Memory.** The model keeps about 15 GB resident and was run on a 24 GB Mac only. 16 GB Macs are not expected to hold it.
- **No image input**, although the original model accepts images.
- **No MCP tools**: the model answers without tools even when MCP is enabled.
- The MTP and DFlash2 panels do not apply to this model; DSpark has no panel and no Tune.

## Unchanged

- Everything else is as in [2.0.10](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.10).
- Conversations, settings, models and saved Tune results are kept.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Desktop checks** (`scripts/check-desktop.sh`, Command Line Tools toolchain, Swift 6.4.2, with the macOS 26.5 SDK):
  - The two halves that CI runs (`--preferences-only` and `--skip-preferences`, together the full suite once) both passed, including speculative sampling (10,019 assertions over 1,944 capability and intent states), the workspace checks (291) and the recommended-models UI.
  - The recommended list now has four families (LFM, Decision, Qwen, Inkling) and ten models; the fake engine has an Inkling fixture.
  - An earlier single full run stopped at the long-visibility check ("visible tail persistently missing") while the machine was busy with other builds. The same check passed in the rerun. It is kept here as an unstable check under load, not as a pass.
- **Engine at the bridge**, with the released binary: Inkling 2 Mini, downloaded by the engine's hub into an empty folder, answered three greedy and three sampled prompts and a second turn, with thinking and answer separated and no control token in the text (DSpark 19.0–28.6 tok/s, single passes). Greedy token hashes of Qwen3.6-35B-A3B, Qwen3.8 Flash Next and d1-2B are identical to engine 2.0.19. Details are in the engine notes.
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum;
  - after mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.11 / 64;
  - its `build-info.json` names source `8a67ba993c59`, with no uncommitted changes, and engine 2.0.20;
  - its four runtime files are byte-identical to those of the engine 2.0.20 archive (`mlxl3` SHA-256 `6d77721e…2a2a`).
- **Not verified:** Inkling 2 Mini was not run by hand in the shipped app: the model card, download plan and preset were checked with the fake engine, and generation with the real model at the bridge. SwiftUI behavior has no formal proof.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. It contains none of these changes: the Windows app uses its own engine and does not run Inkling 2 Mini.

## Source and download

- Compiled source `8a67ba993c59bee266fc1e78f891e64c955aa281`.
- `MLXL3-Desktop-v2.0.11-b64-Apple-Silicon.dmg` · **244,011,609 bytes**, SHA-256 `98f3c8ed7d8d5756ce048f9cc65d9e27b71587ad9b5b04137581e859d63e39f3`.
- `MLXL3-Desktop-v2.0.11-b64-Apple-Silicon.dmg.sha256` · 110 bytes, SHA-256 `22b3201d7ddbbc702c6caa0ec4b02cf427a5352f185e7f919a1985bb4ad61efd`.
- Bundled engine 2.0.20 (`mlxl3` SHA-256 `6d77721e393636f2f1a787e93d4743c100219e247a27b6eeef574c14da152a2a`, source `8a67ba993c59`). The Swift app was built with the Command Line Tools toolchain (Swift 6.4.2) and its macOS 26.5 SDK.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the request to add the model to the recommended list with DSpark, and for the release decision.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini and its DSpark drafter.
- Thanks also to [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [Apple MLX](https://github.com/ml-explore/mlx), [PonyExl3](https://github.com/beamivalice/PonyExl3) and [SwiftMath](https://github.com/mgriebling/SwiftMath), for the EXL3 format and quantizer, the runtime, the Metal EXL3 port, and math rendering.
