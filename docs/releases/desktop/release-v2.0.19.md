# MLXL3 Desktop 2.0.19

**Faster Inkling 2 Mini conversations: a follow-up no longer waits for the whole conversation to be read again, and answers that copy earlier text (code, rewrites) come out faster.**

Desktop **2.0.19 / build 72** bundles MLXL3 engine **2.0.28**: the same sources as [`engine-v2.0.28`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.28).

## New

### Inkling 2 Mini answers follow-ups much sooner

- **The conversation is not read again at each turn.** The engine resumes the previous prompt and reads only the previous answer and your new message. After a 3,000-token document, the second and third questions started in **1.4–2.5 s instead of about 18 s** (M5, 24 GB).
- **Same answers.** The resumed state is identical to the bit to reading the whole conversation; greedy answers were the same with and without the cache.

### Faster answers that repeat earlier text

- **Prompt lookup** proposes the tokens that followed an earlier copy of what the model is writing, and the model checks them in one step. It helps most when the answer quotes or edits text from the conversation: **+20 % to +34 %** tok/s on a Rust file, **+7 % to +60 %** on a rewritten passage, and no change on plain prose (temperature 1).
- It is on by default for Inkling 2 Mini outside Agent mode and DSpark. Sampled answers keep the model's distribution. With greedy settings, an answer can differ slightly from plain decoding, as with DSpark.

## Fixes

- **`pip` in the Agent sandbox no longer goes through `xcrun`.** `pip3` starts with `#!/usr/bin/xcrun python3`; with Xcode selected, each call started `xcodebuild` inside the sandbox and mixed its warnings into the output. The sandbox's private home now has `pip3` and `pip` wrappers (`python3 -m pip`) first on the `PATH`, and commands get `DEVELOPER_DIR` and an `xcrun` cache in the private home.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The Windows app is unchanged since 2.0.16.

## Unchanged

- The Agent screen of 2.0.18, models, conversations, settings and Agent projects.
- Everything else is as in [2.0.18](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.18).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Engine 2.0.28:** see its [release notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.28) for the parity tests, measurements and limits of the prompt cache and prompt lookup.
- **This DMG:**
  - `hdiutil verify` valid, `codesign --deep --strict` passed;
  - version 2.0.19 / build 72;
  - `build-info` has commit `be09e2c0c` with no tracked changes;
  - the bundled engine reports 2.0.28 (`engine-v2.0.28`), and its five files are identical to the published engine archive.
- **Desktop suite: passed** on the release source (548 s, macOS SDK 26.5): Agent sandbox 318 checks, Agent protocol 66, Studios 59, workspace 44, and the end-to-end lifecycle, file imports, bridge, streaming and CLI checks. The live network checks were not run (they need `MLXL3_NETWORK_TESTS=1`). A first attempt with the default macOS 27.2 SDK of the Command Line Tools did not compile (`SwiftUIMacros` plugin not found), a toolchain issue unrelated to this release.
- **Bundled engine in the bridge:** a real Inkling 2 Mini tool call through the Agent sandbox helper of this build, DSpark on demand, and a 3-turn conversation resuming its prompt cache.
- **Not verified:** the prompt cache in Agent tool rounds; the Windows app does not use this engine.

## Source and download

- Source: [`be09e2c0c`](https://github.com/0xZKnw/mlxl3/commit/be09e2c0c3b6480e2b06a7217d53ec42f756a539) (tag `v2.0.19`).
- `MLXL3-Desktop-v2.0.19-b72-Apple-Silicon.dmg`: 253,214,633 bytes, SHA-256 `23fa3abb227cfde6cb1c5913a0a94903d9d6d147c074e08ce266e7c007746091`.
- `MLXL3-Desktop-v2.0.19-b72-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.28, the same files as [`engine-v2.0.28`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.28).

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for pointing at Project Maya and for the prompt-lookup decision.
- Thanks to [Project Maya](https://github.com/mw00/project-maya) by [mw00](https://github.com/mw00) and its contributors, whose Inkling 2 engine inspired the prompt cache and prompt lookup (ideas reimplemented for MLX, no code taken).
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini, and to the GitHub Actions macOS runner, whose Xcode setup exposed the `pip`/`xcrun` noise.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
