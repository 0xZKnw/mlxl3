# MLXL3 Desktop 2.0.28

**The Inkling 2 Mini acceleration is now called DFlash2, and it is on for everyone once its drafter is installed.**

Desktop **2.0.28 / build 81** bundles MLXL3 engine **2.0.32**, rebuilt from this release's source; the Rust sources are unchanged since [`engine-v2.0.32`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.32).

## Fixed

- **The switch stayed off after updating to 2.0.27** for anyone who had once turned the old DSpark switch off: that old choice was still applied to DFlash2. The DFlash2 switch now has its own remembered choice, so it starts from the default — **on as soon as the drafter is installed**. Turn it off once and it stays off.
- **The control still said "DSpark".** The toolbar entry, the inspector card, the switch, the per-answer statistic and the model catalogue now say **DFlash2**.
- **The official DSpark drafter is no longer used by the app.** While the DFlash2 drafter is being downloaded, answers decode normally; the acceleration starts with DFlash2 only.

## Unchanged

- The engine, the drafter download and everything else are as in [2.0.27](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.27).
- The Windows app (its installer is built and attached by the `windows-desktop` workflow after publication).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Published without running the test suites**, at the maintainer's request, to fix 2.0.27 quickly: the app compiles and the DMG checks below passed, nothing else was run for this release.
- **Not verified:** the Desktop suite, the switch and labels in the running app, the drafter download from the app.

## Source and download

- Source: [`0e62d4ea7`](https://github.com/0xZKnw/mlxl3/commit/0e62d4ea7) (tag `v2.0.28`).
- `MLXL3-Desktop-v2.0.28-b81-Apple-Silicon.dmg`: 252,695,832 bytes, SHA-256 `86dc6ce28e752b750a777045db2ce5da9caf2081b5c32eb7e6a3aefe25ff4e9a`.
- `MLXL3-Desktop-v2.0.28-b81-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.32 rebuilt from `0e62d4ea7` (Rust sources identical to `engine-v2.0.32`; the binary is not byte-identical to the published archive, its `runtime-info` carries this commit).
- This DMG: `hdiutil verify` valid, `codesign --deep --strict` passed on the app, version 2.0.28 / build 81, `build-info` has commit `0e62d4ea7` with no tracked changes.

## Known limit

With **tools enabled in chat (MCP) or in Agent mode**, the engine decodes without the drafter, as before: DFlash2 applies to requests without tools.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the report, with screenshots, minutes after 2.0.27.
- [IncoAI](https://huggingface.co/incoai) for the [Inkling 2 Mini DFlash2 drafter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2). Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
