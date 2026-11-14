# MLXL3 Desktop 2.0.1

**Management pages that follow the v2 workspace, richer code colors, and calmer animation.**

Desktop **2.0.1 / build 54**, bundling MLXL3 engine **2.0.10**.

## Interface

- **Models, MCP and Projects** now share Settings’ ASCII aurora, translucent navigation, serif headings, spacing and Back controls.
- **Agent Studio** adopts the same visual language, with project actions below the title and a sidebar that follows the chosen appearance width. The editor keeps its own reading surface.
- The model library, Hugging Face browser, Quantization Studio, project instructions, document lists and profiles follow the shared surfaces.
- Generation animation now settles at **2× the resting speed**, with continuous acceleration and a smooth return to idle.

## Code highlighting

- HTML files now highlight the **CSS inside style blocks** and the **JavaScript inside script blocks**.
- Inline style attributes and event handlers receive CSS and JavaScript colors. TypeScript script blocks, JSON data blocks and import maps use their matching rules.
- HTML comments protect their contents from tag and code coloring. Incomplete embedded blocks, mixed-case tags and Unicode keep their context.
- CSS properties, selectors, custom properties, colors, dimensions and functions have distinct colors in stylesheets and HTML.
- Streaming Chat code retains embedded-language colors across layout chunks. The same lexer serves completed Chat code and the Agent editor.
- The code palette uses Iris, amber and rose tones with separate light/dark values. All six tested syntax roles meet a **4.5:1** contrast target against the editor canvas.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings. The v2.0.0/build53 → v2.0.1/build54 upgrade passed the production updater preflight.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- Inspected 20 native page captures across **820/1280-point windows** and **light/dark appearances**, plus four final HTML-editor captures.
- Four management pages passed rendered sidebar checks and actual Back-button clicks in both appearances.
- Desktop experience passed **55 assertions**, plus **14 background checks**. Generation playback checks cover the 2× rate, continuous transitions and twelve rapid restarts.
- Preferences passed **384 assertions**, notifications **17**, logo rendering **19 standalone / 23 packaged**, Agent workspace checks **44**, and recommended-model layouts passed at 480/840 points.
- Syntax and responsiveness passed **2,529 checks**, including **289** for embedded languages, color contrast, comments, Unicode, Chat chunks, cache updates and the production editor.
- Streaming regression checks passed **141** plain-text cases, **1,194** code fragments and **4,845** Markdown fragments. The existing long-response native test completed its 30 checks.
- A temporary opaque model-page variant failed the new visual regression test. The previous syntax lexer failed the embedded-CSS assertion.
- Swift 6 compilation and the source inventory passed. The signed DMG passed image, signature, runtime-integrity, app self-check and updater checks. Engine and converter payload hashes match v2.0.0.

These are sampled UI and source checks. SwiftUI behavior has no formal proof. Physical validation used Apple M5 and macOS 27.2. Other Macs and OS versions were not exercised in this patch campaign. The editor retains its 256 KiB syntax-coloring budget. No inference or rendering performance gain is claimed. Hosted CI results remain separate from local validation.

## Source and download

- Compiled source `bb6a3e4bf689e50996650d4168f01a922ce17290`.
- `MLXL3-Desktop-v2.0.1-b54-Apple-Silicon.dmg` · **252,898,604 bytes**.
- SHA-256 `e5ae59c3b70e84e3db74a3ec284d74a701217acd385370988188280d765cdf9d`.
- Bundled [engine 2.0.10](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.10).

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the design feedback, embedded-language report and release testing, and [Lucas Bauduin / HENK0O](https://github.com/HENK0O) for the existing appearance and reading controls. Thanks to [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3) and [SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, model format, conversion and math-rendering stack.
