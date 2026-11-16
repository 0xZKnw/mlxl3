# MLXL3 Desktop 2.0.6

**A compact sidebar that shows more than twice as many conversations.**

Desktop **2.0.6 / build 59**, bundling MLXL3 engine **2.0.13**.

## New

- **Compact sidebar.** The controls above and below the conversation list take much less room, so the list gets the space back. In a 760 pt tall window, the list now starts about 205 pt from the top instead of about 390 pt, and about 15 conversations are visible instead of about 6.
  - **Mode tabs.** Chat, Agent, Server and Lab are one row of four tabs, each with its icon above a short label, instead of four stacked rows. The selected tab is highlighted, and the tabs stay disabled during a generation as before.
  - **Search and new conversation on one row.** The search field and an icon-only **New conversation** button (**New task** in Agent mode) share one row. The button keeps its tooltip and ⌘N shortcut.
  - **Icon footer.** Projects and profiles, MCP, Models and Settings are one row of four icons instead of four labelled rows. Each icon has a tooltip and an accessibility label with its full name. The update badge stays in the footer.
  - **Logo only.** The header shows the MLXL3 logo without the "MLXL3" text next to it. The logo keeps the accessibility label "MLXL3".
  - Conversation rows and the project picker are 30 pt tall, and the section headers are tighter.

## Unchanged

- Conversations, pins, projects, search and every page behave as before. Only the layout of the sidebar changed.
- The engine is unchanged at source level: Desktop 2.0.6 bundles engine 2.0.13 (see the limits below about the binary).

## Windows

The same compact sidebar is on `main` for the Windows app ([PR 175](https://github.com/0xZKnw/mlxl3/pull/175)): four mode tabs, search with an icon-only new-conversation button, a footer of four icons and the logo without the wordmark. No Windows installer is published with this release.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Full Desktop suite:** `scripts/check-desktop.sh` passed on the release changes (the Python part used the project's virtual environment). It includes Studio modes 2,324, workspace 276 and 44, Studios 58, agent sandbox 25 and agent protocol 29 checks.
- **Sidebar checks in `tests/desktop-experience-check.swift`** (3,212 assertions passed):
  - clicking the search field does not create a conversation, and clicking the new-conversation button creates one;
  - each of the four footer icons opens its own page (Projects and profiles, MCP, Models, Settings), with exactly one page open at a time.
  
  An earlier run failed on the MCP icon because the check clicked while the previous page was still fading out. The check now waits for the 0.2 s fade to finish; the app was not changed for this.
- **Test bundle helper.** `scripts/copy-test-runtime.py` now also finds engines installed as `engine-v<version>-<archive SHA-256>`, the layout used by recent Desktop releases. Before this fix, `scripts/build-desktop-test.sh` failed on a Mac with such an engine installed.
- **Visual check.** The sidebar was rendered off screen in light and dark mode and checked by hand, then tried in a test build of the app. The space figures above come from these renders.
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum.
  - After mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.6 / 59.
  - Its `build-info.json` names source `9bbdf795115a`, no uncommitted changes, and engine 2.0.13.
  - The bundled engine's `runtime-info` reports 2.0.13, commit `9bbdf795115a`.
- **Bundled engine binary.** The bundled `mlxl3` was compiled from this release's source. Its engine sources (`native/`, `Cargo.toml`, `Cargo.lock`, `build.rs`) are identical to the [engine 2.0.13](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.13) source `c5626a134537`, but the binary is **not** byte-identical to the published one: its SHA-256 is `8b3522e9…32a9fc` instead of `644c2e08…888e00`. Without code signatures, the two binaries differ in 45 bytes: the embedded build revision (two copies, 24 bytes), the Mach-O UUID (16 bytes), and 5 bytes in the code whose origin was not traced.
- **Not verified:**
  - No model was run with this build: the change is interface-only, and the engine is unchanged at source level.
  - The Windows app is not part of this release, and no Windows installer was rebuilt.
  - SwiftUI behavior has no formal proof.
- **Hosted CI:** on the last `main` commits, `desktop-and-protocol`, `desktop-preferences`, several Kani shards, `native (macos-26)` and `guard-proofs` were already failing or cancelled. The Desktop jobs fail because the hosted Swift compiler times out on `ChatComponents.swift:390`, a file this release does not touch. Hosted CI is **not green**. The local checks above are the release verification.

## Source and download

- Compiled source `9bbdf795115a6c5e879b5c8afa30074c8d60fc85`.
- `MLXL3-Desktop-v2.0.6-b59-Apple-Silicon.dmg` · **250,929,391 bytes**.
- SHA-256 `7f2f7837a8c8d8b64e01e9f8daaf86e6fd661cd3475d05a3e32930775dad46a5`.
- `MLXL3-Desktop-v2.0.6-b59-Apple-Silicon.dmg.sha256` · 109 bytes, SHA-256 `2e03cb74aa9a1e1a94086027eff1f878b3333d7f94963a3ba4730c31be3cae21`.
- Bundled engine: 2.0.13, built from the source above (`mlxl3` SHA-256 `8b3522e9e4fd5a7e417dc7e92f1245c28379b0527c9d2b76067d475ef832a9fc`). The Swift app was built with the Command Line Tools macOS 26.5 SDK.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the feature request, the design feedback and the release decision. Thanks to [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [MTPLX](https://github.com/youssofal/mtplx), [TensorFold](https://github.com/ashhart/TensorFold) and [SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, model format, MTP research and math rendering, and to [Qwen](https://huggingface.co/Qwen) and [Liquid AI](https://huggingface.co/LiquidAI) for the models.
