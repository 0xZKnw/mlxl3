# MLXL3 Desktop 2.0.4

**Fixes the dot that bounced around next to the reply header while a response was being generated.**

Desktop **2.0.4 / build 57**. The bundled engine is unchanged: MLXL3 engine **2.0.11 beta 3**, as in Desktop 2.0.3.

## Fixes

- **No more bouncing dot during generation.** The small pulsing dot next to the "MLXL3" label of a streaming reply could drift away from the label and keep bouncing between two positions until the reply finished. Its endless pulse animation was started when the row appeared and also picked up the row's layout move, so the dot's position looped forever. The repeating animation now only changes the dot's opacity and size; the dot stays in place and pulses as intended.
- The three dots of the "Thinking" placeholder had the same flaw and get the same fix.
- With Reduce Motion enabled, both indicators stay still, as before.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- This is a small interface fix released at the maintainer's request **without running the test suites** on the release source. `scripts/check-desktop.sh` was not run for 2.0.4.
- The same change was checked on an older local branch before the release, where the Swift build and the full `scripts/check-desktop.sh` suite passed. That branch is not the release source.
- The release app built with `swift build --configuration release` and passed `codesign --verify --deep --strict`, before and after mounting the DMG. The bundled engine answered `mlxl3 list --json`. `hdiutil verify` reported a valid checksum.
- **Not verified:** the fix was not observed in a running app during a real generation, and no automated check covers these animations. SwiftUI behavior has no formal proof.

## Source and download

- Compiled source `457ccceb23a87dadaf444b494c9532b63e2d137b`.
- `MLXL3-Desktop-v2.0.4-b57-Apple-Silicon.dmg` · **251,075,288 bytes**.
- SHA-256 `efeda0834a1e8a0baf5ad7ec555a2534414b2e2ca1515b7daec98828b7be350d`.
- Only the Desktop app changed. The bundle was assembled from the published Desktop 2.0.3 DMG (SHA-256 `0af39d25…6e79eef`, verified). Across the app bundle, only `Info.plist`, the app binary, `build-info.json` and the code signature differ from 2.0.3. The engine is the [engine 2.0.11 beta 3](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.11-beta.3) build (`mlxl3` SHA-256 `b2dab05b…bd655`), and its manifest was revalidated against the payload when `build-info.json` was written.
- The Swift app was built with the Xcode macOS SDK instead of the Command Line Tools 26.5 SDK used for 2.0.3.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for spotting the bouncing dot and the release decision. Thanks to [Apple MLX](https://github.com/ml-explore/mlx), [Qwen](https://huggingface.co/Qwen), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [MTPLX](https://github.com/youssofal/mtplx), [TensorFold](https://github.com/ashhart/TensorFold) and [SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, models, model format, MTP research and math rendering.
