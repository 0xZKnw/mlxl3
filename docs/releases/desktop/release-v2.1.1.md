# MLXL3 Desktop 2.1.1

**The line counting failed and succeeded commands under an Agent answer is gone.**

Desktop **2.1.1 / build 89** bundles MLXL3 engine **2.0.38**, rebuilt from this source. The Rust sources are identical to [`engine-v2.0.38`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.38); the binary is not byte-identical to that archive because it records another source commit. The engine itself is not republished.

## Changed

- **No more `Commands · 1 failed · 4 succeeded` under an answer.** The same summary for test commands (`Test commands · …`) is removed too. Each command and its result is still shown in the steps of the turn.
- The **Review N files** button and the turn's document stay, when the turn changed files. A turn that only ran commands now shows nothing in that place.

Everything else is [Desktop 2.1.0](https://github.com/0xZKnw/mlxl3/releases/tag/v2.1.0): one conversation mode, summarized conversations, and the confirmation before deleting in your folder.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

**Windows:** the Windows app never had this line; its installer for this tag has the same content as 2.1.0.

## Verification and limits

- Every source of the app was recompiled with warnings as errors together with the Studio modes check, which passed (2,534 checks, fake engine); the task workspace check passed (96), which still verifies that command receipts are recorded.
- No test read these labels, so none was changed. The app binary of this DMG no longer contains the removed label text.
- **Not verified:** the app of this DMG was not launched, so the row was not looked at on screen. No engine test and no real-model run were repeated for this release: the Rust sources are those of engine 2.0.38. CI of `main` for this source had not finished when this release was published.

## Source and download

- Source: [`2dfb37057`](https://github.com/0xZKnw/mlxl3/commit/2dfb37057) (tag `v2.1.1`).
- `MLXL3-Desktop-v2.1.1-b89-Apple-Silicon.dmg`: 254,248,731 bytes, SHA-256 `d6dbc465238ea60dde14a6c398bfd0b5fbdfeed766d2d442d21f6d8ce395839a`.
- `MLXL3-Desktop-v2.1.1-b89-Apple-Silicon.dmg.sha256`: 109 bytes.
- Bundled engine: MLXL3 engine 2.0.38 rebuilt from this source (same Rust sources as `engine-v2.0.38`).
- This DMG: `hdiutil verify` valid, `codesign --deep --strict` passed on the app of the mounted DMG, version 2.1.1 / build 89, `build-info` has commit `2dfb37057` with no tracked changes, MLX 0.32.2 and engine 2.0.38 (`engine-v2.0.38`).

## Acknowledgements

- [0xZKnw](https://github.com/0xZKnw) for the request and the change.
- [HENK0O](https://github.com/HENK0O) for the per-turn review this row belongs to ([#204](https://github.com/0xZKnw/mlxl3/pull/204)).
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
