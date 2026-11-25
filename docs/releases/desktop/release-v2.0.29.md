# MLXL3 Desktop 2.0.29

**DFlash2 now also accelerates Inkling 2 Mini when tools are enabled: MCP tools in chat and Agent mode.**

Desktop **2.0.29 / build 82** bundles MLXL3 engine **2.0.33**: the same files as [`engine-v2.0.33`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.33).

## Fixed

- With MCP tools enabled in chat, or in Agent mode, answers were decoded without the drafter even with the DFlash2 switch on. The engine now speculates in those requests too; tool calls go through the same filter and stops as before.

## Unchanged

- The app itself: everything else is as in [2.0.28](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.28).
- The Windows app (its installer is built and attached by the `windows-desktop` workflow after publication).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **One real tool round** through the bridge with this app's sandbox server (see the [engine notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.33)): the tool call succeeded with speculation on.
- **The test suites were not run for this release**; the app compiles and the DMG checks below passed.
- **Not verified:** the app driven by hand with MCP tools or in Agent mode, long Agent tasks.

## Source and download

- Source: [`7cbdba545`](https://github.com/0xZKnw/mlxl3/commit/7cbdba545) (tag `v2.0.29`).
- `MLXL3-Desktop-v2.0.29-b82-Apple-Silicon.dmg`: 252,111,304 bytes, SHA-256 `d25568115670603a781327d94dfa14d8b8177bfa42fe22b7baa4bc48c6abd848`.
- `MLXL3-Desktop-v2.0.29-b82-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.33, the same files as [`engine-v2.0.33`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.33).
- This DMG: `hdiutil verify` valid, `codesign --deep --strict` passed on the app, version 2.0.29 / build 82, `build-info` has commit `7cbdba545` with no tracked changes; the five runtime files are identical to the published engine archive.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the report.
- [IncoAI](https://huggingface.co/incoai) for the [Inkling 2 Mini DFlash2 drafter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2). Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
