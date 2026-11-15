# MLXL3 Desktop 2.0.3

**A cleaner interface, new welcome greetings, and a lighter animated background.**

Desktop **2.0.3 / build 56**. The bundled engine is unchanged: MLXL3 engine **2.0.11 beta 3**, as in Desktop 2.0.2.

## New

- **Welcome greetings.** The empty chat now greets you according to the time of day ("Good morning", "Good evening", "Still up?"…), with an occasional any-hour line such as "What’s on your mind?". The greeting stays the same while a conversation is open and can change with a new conversation or a relaunch. French and English.

## Interface

- **Sidebar.**
  - Mode rows, the project picker, search and conversation rows share one icon column and inset.
  - The project picker is a sidebar row instead of the system pop-up.
  - The selection fill covers the whole conversation row, pin included. An unpinned row reserves room for the pin only while hovered.
  - The conversation list shows its count and fades out at its edges instead of being cut off.
- **Chat.**
  - The transcript fades out under the toolbar and above the composer.
  - The composer has continuous corners, a soft shadow and a focus outline.
  - User bubbles use continuous corners.
  - The send/stop button is a solid disc with a press effect.
- **Full-screen pages.** Settings, Models, Projects and MCP now fade in over the chat instead of showing the chat through them.
- **Models page.**
  - The download indicator sits in the page header.
  - Load and Locate folder use the app's own button styles.
  - Model cards have continuous corners.
  - Refresh, Show in Finder and Remove model have VoiceOver labels.

## Performance

The animated background (the light moving across the type cells) looks exactly the same and costs less:

- **Transparent areas are skipped.** About 40% of the background is fully transparent. The shader used to compute the full light effect there anyway; it now leaves those pixels early. An off-screen render of the old and new shaders, 4,584,000 pixels over 2 window sizes and 5 animation times, gave identical bytes.
- **No drawing while the window is covered.** Desktop 2.0.2 kept animating the background at 60 fps when another app's window covered it, for example while you worked elsewhere during a reply. The background now pauses when the window cannot be seen and resumes where it stopped.

Measured on an Apple M5 (macOS 27.2, on battery) with the app's preview mode, no model loaded, the welcome screen open, 10 s warmup and 30 s windows. Values are the app's GPU time per 30 s from the IOKit GPU accounting, A = 2.0.2 background, B = 2.0.3 background, same interface code otherwise:

| Window | 2.0.2 (A) | 2.0.3 (B) | Runs |
| --- | --- | --- | --- |
| Visible | 4.34 s median | 3.84 s median (−11%) | 4 + 4, order A B B A B A A B; B lower in every pair |
| Covered by another window | 3.56 s | 0 s | 2 + 2, order A B B A |
| Minimized | 0 s | 0 s | 2 + 2; macOS already stops drawing |

With the window covered, the app's CPU time also dropped from 1.1 s to 0.02 s per 30 s. WindowServer GPU time did not change measurably with the window visible. These are idle measurements of the interface. The effect on token generation speed was **not measured**.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- The full `scripts/check-desktop.sh` suite passed on the release source (73 passing check summaries), including the desktop experience check (3,212 assertions, with the new greeting checks), preferences (384), Studio modes (2,324), response UI and text size, Markdown and streaming, the 14 aurora background checks, updater and installer boundaries, the app updater preflight (15 cases) and the new aurora shader parity check. The Studios (58 + 39 Python tests), agent sandbox (25), agent protocol (29) and workspace (44) suites passed.
  - The first full run failed in the long-response visibility check because its test window was hidden during the run (`window.isVisible` false); the rerun passed. The second run stopped before the Studios suite because the system Python lacks pytest; that suite was run with the project's Python and passed.
- The shader parity check fails if a single pixel changes (a mutant threshold changed 333,973 pixels by up to 5/255) or if the shaders do not compile.
- The background measurements are described under Performance. Their raw data and protocol are logged in `opti.md` (APP-2026-11-15-AURORA). Pause and resume were also checked on one running app: rendering resumed after a screen unlock, stopped under a covering window and resumed when it was removed.
- **Not verified:** the effect on generation speed; the pixel parity in dark mode and at Retina ×2 (the check renders the light artwork at scale 1); Macs other than Apple M5 and macOS 27.2; hosted CI rendering of the new check. The release app was not launched with a model; the interface was reviewed in the app's preview mode built from the same source. SwiftUI behavior has no formal proof.

## Source and download

- Compiled source `5579ccb9de8f64305335c9b544cdf1fda38a8981`.
- `MLXL3-Desktop-v2.0.3-b56-Apple-Silicon.dmg` · **250,959,667 bytes**.
- SHA-256 `0af39d25910fb48a88dad2f037c060c7813eec2186cd4eb0afc1633cf6e79eef`.
- Only the Desktop app changed. The bundled engine and quantizer were taken unchanged from the published Desktop 2.0.2 DMG (SHA-256 `0f93bc5c…7d8ce0`, verified). Across the 3,118 files and links of the app bundle, only `Info.plist`, the app binary, `build-info.json` and the code signature differ from 2.0.2. The engine is the [engine 2.0.11 beta 3](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.11-beta.3) build (`mlxl3` SHA-256 `b2dab05b…bd655`), and its manifest was revalidated against the payload.
- `hdiutil verify` reported a valid checksum. The mounted app passed `codesign --verify --deep --strict` and is identical to the built app.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the design review, the greeting idea and the release decisions. Thanks to [Apple MLX](https://github.com/ml-explore/mlx), [Qwen](https://huggingface.co/Qwen), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [MTPLX](https://github.com/youssofal/mtplx), [TensorFold](https://github.com/ashhart/TensorFold) and [SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, models, model format, MTP research and math rendering.
