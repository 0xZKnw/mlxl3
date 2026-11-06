# MLXL3 Desktop 2.0.0-beta.7 — build 38

This Desktop-only beta builds the current interface while retaining the published beta 6 engine **2.0.2** and converter byte for byte. Desktop's internal version remains **2.0.0**, now **build 38**. The engine currently under development is excluded.

Desktop source: `accabf48a8a6c2a57fb1daa05703adc7f41be330`, based on main `ed3bfdb8ff45e650fd8347b1a8f00931187284bf`. Reused engine and converter source: beta 6, `9df28b6d2a10e0d646f8a36532ae69c88c9ec3e8`.

## Features

- Choose System, Light, Dark or Midnight Blue appearance. The setting persists across launches; Dark remains the default.
- Pin conversations to keep them at the top of the history. Pins survive saves, restarts and history import/export.
- Switch between centered and wide reading layouts, and adjust response text size from appearance settings without losing the current conversation or draft.
- Choose whether a model loads on startup and select the startup model. Automatic loading remains enabled by default.
- Enable optional completion notifications for background generations, with cancellation, errors and foreground activity handled separately.
- Access MTP from the chat toolbar or inspector. Each model remembers its own ON/OFF choice; new models default to OFF. Flash Next uses its integrated checkpoint head.

## Fixes

- Keep the MTP depth controls readable in narrow inspector panels.
- Improve logo contrast in Light appearance and keep rich response rendering consistent with the selected theme and text size.
- Preserve cancellation when a bridge start is waiting for a previous process to retire.
- Accept the separate native Metal engine archive format in the updater. This package still bundles the existing MLX engine 2.0.2.

## Performance

No new engine, inference optimization or converter is included. No new speed or memory measurement is claimed.

## Installation and compatibility

Apple Silicon and macOS **26.2 or later**. Open the DMG and drag MLXL3 Desktop into Applications. Build **37 → 38** advances the Desktop build number while keeping the internal version at 2.0.0. This build is signed ad hoc and is not notarized.

The converter and its architecture compatibility limits remain those of beta 6. Existing Agent Studio requirements, including Command Line Tools Git, remain applicable.

## Verification and limitations

The complete Desktop checks passed in retained segments, including lifecycle, preferences, bridge, imports, rendering, previews and Studio workspaces. Package helpers: 94 tests passed; Studio Python checks: 39 passed. Per-model sampling: 969 assertions over 216 capability states; layout: 16 rendered cases and an old-layout mutation rejected; long-response native bitmap/OCR: 30 frames. The signed, read-only DMG passed the production updater build37→38 check and 15 updater regression cases. All five runtime entries and 3,050 converter entries match the published beta 6 payloads. See [the build report](https://github.com/0xZKnw/mlxl3/blob/codex/desktop-v2-beta7/docs/desktop-beta7-build-2026-11-06.md) for commands and retained failures.

No model inference, conversion or fresh engine benchmark was run for this Desktop package. Swift source proof is unavailable; tests are not proofs. At the publication snapshot, no GitHub CI run exists for this exact source; the complete local Desktop checks passed. The source is provided on the release tag and the preparation branch. This release does not install or replace the local application.

## Artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| MLXL3-Desktop-v2.0.0-beta.7-b38-Apple-Silicon.dmg | 247050083 | `431e7de3f8a1d6df0aad4944c9817b6230dc86d56dc1844fb523268f3a9e3385` |
| MLXL3-Desktop-v2.0.0-beta.7-b38-Apple-Silicon.dmg.sha256 | 116 | `43d27c8eae9864561d6bc2b32a7ca8acf2b4b199095d535f64396b5d4a30126c` |
| MLXL3-v2.0.0-beta.7-validation.json | 2347 | `b2550829d1d618671681bb1ff9666b990e52ea2432907cc8232567dd2215a61b` |
| MLXL3-v2.0.0-beta.7-verification.tar.gz | 1072810 | `db327d8365f504db89d2ea3e0f6de140e6349f01788499358bc3766d22fe1aec` |

The DMG is accompanied by its SHA-256 file, `MLXL3-v2.0.0-beta.7-validation.json` and `MLXL3-v2.0.0-beta.7-verification.tar.gz`. This Desktop package produces no standalone engine artifact.

## Acknowledgements

Thanks to [Lucas Bauduin / HENK0O](https://github.com/HENK0O) for appearance, startup, pinned chats, reading controls and per-model MTP work; [0xZKnw](https://github.com/0xZKnw) for MLXL3 and the Desktop feature reports; [Apple MLX](https://github.com/ml-explore/mlx) for the retained tensor runtime; [beamivalice / PonyExl3](https://github.com/beamivalice/PonyExl3) for the retained EXL3 converter; [MTPLX](https://github.com/youssofal/mtplx) for the existing MTP work; and [SwiftMath](https://github.com/mgriebling/SwiftMath) for equation rendering. Their existing notices remain in the package.
