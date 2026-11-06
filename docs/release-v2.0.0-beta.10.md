# MLXL3 Desktop 2.0.0-beta.10

Desktop prerelease **v2.0.0-beta.10**, internal version **2.0.0 / build 41**. Desktop source: **5e815940cf84f4004eff35ecd41e17c595cf48f3**. Bundled engine **2.0.4**, retained from the verified beta 9 DMG and independently published as [engine-v2.0.4-beta.1](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.4-beta.1).

## Settings and navigation

- Settings now use the whole window, with separate General, Appearance, Models & memory, Conversations, Connections, Privacy and Updates sections.
- Models, MCP and Projects/profiles also use full-window views, larger controls and a shared Back button. Closing a menu returns to the previous screen while retaining the workspace.
- The compact sidebar keeps the logo and places the new-conversation action beside it. Clear Liquid Glass is limited to toolbar actions; navigation and content remain solid.
- Short menu/tab fades and press feedback respect Reduce Motion.

## Appearance

- Eight themes: System, Light, Dark, Midnight blue, Slate, Forest, Plum and Paper.
- Six accent choices and three message-spacing settings, alongside the existing response width and 12–24 pt text-size controls.
- Appearance changes are saved locally. Restore default appearance resets only visual preferences.
- Existing startup model selection, completion notifications, automatic unloading, memory saving, update channels and model acceleration controls remain available.

## Fixes

- Top K keeps integer values without drawing overlapping native tick marks.
- Invalid context-limit input now disables saving instead of retaining a stale valid number.
- Larger model-inspector sections keep sampling lock explanations readable.
- MCP edits in the UI preview use an isolated temporary configuration. The preview explicitly refuses real connection checks.

## Engine and performance

No engine, quantization or inference optimization is included in this Desktop update. The beta 9 engine, converter environment and math fonts are retained and verified against the published DMG. No new speed or memory improvement is claimed.

## Installation and compatibility

Apple Silicon, macOS **26.2+**. Use the DMG or the Desktop beta update channel. The installed app and the user's engine pointer were not replaced by this task. Signing is ad hoc; this build is not notarized.

## Validation and limits

Swift debug/release compilation, compiler SIL invariant checks, targeted formatting and diff checks were used. The user reviewed the running isolated UI preview and approved the final toolbar/menu direction. Preview MCP connection refusal was observed.

At the user's request, full automated suites, exhaustive theme/keyboard/control checks and actual model loading were not run. Regression assertions were added to the existing Desktop preference check but were not executed locally. No complete antislop Delivery Gate or behavioral SwiftUI proof is claimed. Remote CI status and exact artifact checks are recorded in the attached validation file; an absent workflow run is not a passing run.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the redesign direction, screenshots and live UI review; [HENK0O](https://github.com/HENK0O) for the retained appearance and reading foundations; [Apple SwiftUI](https://developer.apple.com/documentation/swiftui) for the native interface and Liquid Glass APIs; [antislop](https://github.com/miqdadbadjuber/anti-slop) for the UI review guidance; [SwiftMath](https://github.com/mgriebling/SwiftMath) for equations; and [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [LiquidAI](https://huggingface.co/LiquidAI), [PonyExl3](https://github.com/beamivalice/PonyExl3) and [MTPLX](https://github.com/youssofal/mtplx) for the retained engine and conversion foundations. Existing notices remain included.

## Artifacts

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| MLXL3-Desktop-v2.0.0-beta.10-b41-Apple-Silicon.dmg | 256286105 | `5c331a3fbddc4fd14f071fb92586447ecd4361737a753faefede95bfd6149fe3` |
| MLXL3-Desktop-v2.0.0-beta.10-b41-Apple-Silicon.dmg.sha256 | 117 | `776774fa6fe2b8ebdc1054049bc122a2a41f8500b31adf7c4c7915ac8e7118a2` |
| MLXL3-v2.0.0-beta.10-validation.json | 2249 | `bc267186ef618ecf31c2063d8097565086f8d99d7c08807dab14a963fb6b92cc` |
| MLXL3-v2.0.0-beta.10-verification.tar.gz | 110261 | `1e37eac435a736f1b626bd0040bb0193f1f6a86d61011b787ffc76d788918059` |

The mounted DMG passed its internal checksum, arm64 app signature, bundle identity, engine manifest and signed executable comparisons. All 3,048 converter files and the retained runtime/fonts match beta 9 in SHA-256 after packaging. Both app self-checks passed; the physical Metal subcheck was disabled. No CI run exists for the pushed feature-branch source. Three packaging-tool/oracle corrections are preserved in the attached logs, with no product-source changes.
