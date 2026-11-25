# MLXL3 Desktop 2.0.31

**Agent mode with Inkling 2 Mini no longer gets stuck on "Unreadable tool call".**

Desktop **2.0.31 / build 84** bundles MLXL3 engine **2.0.35**: the same files as [`engine-v2.0.35`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.35).

## Fixed

- **Tool calls refused one after the other.** Inkling 2 Mini sometimes wrote the tool name correctly before a call and only its first part inside it (`sandbox` for `sandbox.javascript`). Each call then failed with "Unreadable tool call", the model tried the same call again, and the turn went nowhere. These calls now run. A call written with the bare name `javascript` runs too.
- When a call really names two different tools, it is still refused, and the model is told what differs.

## Unchanged

- The app itself: everything else is as in [2.0.30](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.30).
- The Windows app (its installer is built and attached by the `windows-desktop` workflow after publication).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- The refused call is now a regression test of the engine, taken as the model wrote it; the engine's test suite passed (see the [engine notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.35)).
- A build with this fix read 12 real tool calls out of 12 on Inkling 2 Mini with this app's sandbox server.
- **Not explained:** why the model shortened the name; it did not happen again in 28 real calls. This release makes that slip harmless; it does not prevent it.
- **The Desktop test suite was not run for this release**; the app compiles and the DMG checks below passed.

## Source and download

- Source: [`a0d6a2e66`](https://github.com/0xZKnw/mlxl3/commit/a0d6a2e66) (tag `v2.0.31`).
- `MLXL3-Desktop-v2.0.31-b84-Apple-Silicon.dmg`: 256,831,039 bytes, SHA-256 `5636b62e85050ce30f433692cb1da2bf717975948cd810034e9a2419309e2190`.
- `MLXL3-Desktop-v2.0.31-b84-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.35, the same files as [`engine-v2.0.35`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.35).
- This DMG: `hdiutil verify` valid, `codesign --deep --strict` passed on the app, version 2.0.31 / build 84, `build-info` has commit `a0d6a2e66` with no tracked changes; the five runtime files are identical to the published engine archive.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the report.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
