# MLXL3 Desktop 2.0.0-beta.14

Desktop prerelease **v2.0.0-beta.14**, internal version **2.0.0 / build 45**.
The bundled MLX engine remains **2.0.5**; there is no new engine release.

## Changes

- Personalize the app with additional themes, preset or custom accent colors,
  message fonts, bubble styles, message spacing, reading width and sidebar
  width. Settings apply immediately, persist locally and can be reset.
- Hugging Face model cards render Markdown, HTML, tables and supported images
  instead of exposing frontmatter or raw styles. Card scripts are disabled;
  external links open in the browser.
- The model settings panel removes a duplicate MTP tuning action and uses a
  clearer label. Pinned revisions in model details have a readable label.

## Fixes

- The update notification opens the Updates page directly rather than the
  first page of Settings.
- A native table rendering test now compares settled frames rather than
  different moments of the streaming fade animation.

## Performance observations

The README includes measured MTP and DFlash2 results for Qwen3.6 35B-A3B and
Qwen3.8 27B on an Apple M5 with 24 GiB of unified memory. These are small local
samples with changing swap conditions, not a controlled or universal speedup
claim. See the [measurement record](../../measurements/v2-speculative-20261111/).

## Installation and compatibility

Requires Apple Silicon and **macOS 26.2 or later**. Enable the Desktop beta
channel in Settings or install the DMG manually. The application is ad-hoc
signed and not notarized. Models and conversations are not bundled.

## Validation and known limits

The model-card tests cover Markdown, HTML, pinned relative images, blocked
scripts and loopback requests, including the real Qwen3.6 card and its graph.
Model details were rendered at 480 and 840 points. The appearance tests cover
persisted settings, reset and the update shortcut. Desktop and Python checks
were completed in separate segments after the system Python lacked pytest.
No physical-screen interaction or Swift source-level formal proof is claimed.

Model cards larger than 256,000 bytes cannot be displayed in-app; images hosted
outside the permitted Hugging Face and GitHub domains are blocked. The previous
main-branch CI run included failed Rust proof jobs, including an out-of-memory
failure. Current GitHub run statuses and artifact identities are recorded with
the published release.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the requests and feedback;
[Marked](https://github.com/markedjs/marked) for Markdown rendering;
[Apple MLX](https://github.com/ml-explore/mlx),
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) and
[SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, format
and mathematical rendering components.
