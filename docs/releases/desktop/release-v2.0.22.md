# MLXL3 Desktop 2.0.22

**A calmer Desktop with four styles to choose from, and an Agent mode without a 32,768-token cap.**

Desktop **2.0.22 / build 75** bundles MLXL3 engine **2.0.30** unchanged: the same files as [`engine-v2.0.30`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.30).

## New

### Styles

Settings → Appearance → **Style** now offers four looks. The style is separate from the System / Light / Dark choice.

- **Graphite** (new default): warm neutrals taken from the logo's off-white. Text is never pure white in dark mode, the sidebar sits a step darker than the conversation, and titles use SF Pro. The composer is the only raised surface. Labels, tool rows and metrics are quieter, and thinking is shown in proportional type. The empty home screen has a faint monochrome character texture; it runs at 30 fps and pauses when the window is hidden or Reduce Motion is on.
- **Minimal**: Graphite with nothing behind the work.
- **Aurora ASCII**: the Desktop 2.0 look, unchanged, with the violet light printed in characters behind every view.
- **Aurora**: the first, soft aurora from the 2.0 betas, without characters.

Choosing a style also applies the accent it is designed around: Neutral for Graphite and Minimal, Iris for both Aurora styles. You can pick another accent afterwards, and a custom colour is never replaced.

**After the update** everyone starts on Graphite. If you never chose an accent, you also get the Neutral accent; an accent you chose is kept. To get the previous look back, pick **Aurora ASCII**.

### Agent mode: no total token cap

With the bundled engine 2.0.30, an Agent turn no longer stops at 32,768 output tokens or 24 tool rounds. It has no total output budget and up to 256 tool rounds. The engine prunes old context and stops a loop of identical tool calls. When a turn still reaches an output limit, the message now says so instead of quoting 32,768. With an older engine, Desktop keeps the previous limits.

### Generate an image on Apple M5

On Macs with an Apple M5-family GPU and macOS 26.3 or later, the sidebar has a **Generate an image** button. It runs Qwen-Image-2.1 text-to-image from a local EXL3 denoiser folder. The first launch needs internet access and `uv`; see [Generate an image on Apple M5](https://github.com/0xZKnw/mlxl3/blob/main/docs/studio-image21.md). The button is hidden on other Macs.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The Windows app is unchanged since 2.0.16.

## Unchanged

- Engine 2.0.30, models, conversations, projects and Agent projects. Other settings are unchanged; only the default style, and the accent for people who never chose one, are new.
- Everything else is as in [2.0.21](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.21).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Engine 2.0.30:** see its [release notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.30) for the Agent loop checks and limits.
- **Desktop checks:** the preferences, notifications, logo and response UI part of `scripts/check-desktop.sh` passed on the release commit with the macOS 26.5 SDK (467 preference assertions). The new checks cover each style's persistence and accent, WCAG contrast of every style in light and dark (body ≥ 12:1, secondary ≥ 6:1, metadata ≥ 4.5:1 on the canvas and sidebar), and the rendered settings in Aurora ASCII and Graphite. The rest of the Desktop suite, including the new style-background check, was not run again for this release; CI runs it.
- **Visual check:** the four styles were captured in the app's preview mode in light and dark (home, conversation with tools and code, settings, models, Agent, inspector). Rendered colours match the design values, and Aurora ASCII matches 2.0.21.
- **This DMG:**
  - `hdiutil verify` valid, `codesign --deep --strict` passed;
  - version 2.0.22 / build 75, built with the macOS 26.5 SDK;
  - `build-info` has commit `0bf26a024` with no tracked changes;
  - the bundled engine reports 2.0.30 (`engine-v2.0.30`, commit `dc20d6693583`), and its four runtime files are identical to the published engine archive.
- **Not verified:** the GPU cost of the home texture during inference (not measured; it stops in conversations and pauses when the window is hidden); image generation, which needs an Apple M5, was not run for this release; the app was not driven by hand with a real model.

## Source and download

- Source: [`0bf26a024`](https://github.com/0xZKnw/mlxl3/commit/0bf26a0248c63458a58edb455f4a6e9513807c1a) (tag `v2.0.22`).
- `MLXL3-Desktop-v2.0.22-b75-Apple-Silicon.dmg`: 252,574,608 bytes, SHA-256 `0ab166b3fa226a0d36f52c5df0617587c829c1a9051b4f6b261d154f494f9812`.
- `MLXL3-Desktop-v2.0.22-b75-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.30, the same files as [`engine-v2.0.30`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.30).

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for asking for a Desktop that feels crafted, for keeping the Aurora looks as choices, and for the uncapped Agent loop.
- Thanks to [HENK0O](https://github.com/HENK0O) for image generation on Apple M5 ([#189](https://github.com/0xZKnw/mlxl3/pull/189)).
- Design references: Anthropic's [`frontend-design` skill](https://github.com/anthropics/skills/blob/main/skills/frontend-design/SKILL.md), [Linear's 2026 interface refresh](https://linear.app/now/behind-the-latest-design-refresh) and Apple's [Human Interface Guidelines](https://developer.apple.com/design/human-interface-guidelines/dark-mode).
- Image generation: [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) by Qwen, [MFLUX](https://github.com/mflux-community/mflux) and the [MLX Community auxiliary weights](https://huggingface.co/mlx-community/Qwen-Image-2.1-mflux-q4).
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
