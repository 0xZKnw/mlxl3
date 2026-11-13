# MLXL3 Desktop 2.0.0-beta.15

Desktop prerelease **v2.0.0-beta.15**, internal version **2.0.0 / build 46**.
The bundled MLX engine remains **2.0.5**; there is no new engine release.

## Fixes

- Release notes in Settings now render Markdown instead of showing raw markup.
  Headings, lists, emphasis and links use the same renderer as conversations.
- The model-card integration check now waits for complete server metadata and
  reports a startup failure clearly. A slow, partial startup is covered in CI.

## Installation and compatibility

Requires Apple Silicon and **macOS 26.2 or later**. Select the Desktop beta
channel in Settings or install the DMG manually. The app is ad-hoc signed and
not notarized. Models and conversations are not included in the download.

## Validation and known limits

The update-notes regression exercises release selection and the real Settings
page with a disposable update response. It checks the rendered content at two
window widths in light and dark themes; the download itself is simulated.
Desktop test-server startup is checked with normal, delayed and incomplete
metadata, as well as early-exit and timeout cases. Swift source-level formal
verification and a physical-screen interaction were not performed.

The previous main-branch CI had Kani timeouts, resource exhaustion and a verifier
compiler crash. See the current GitHub Actions runs for the exact commit's
status; none of these tool failures establishes a Rust counterexample or proof.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for reporting the release-notes
issue, and to [Apple MLX](https://github.com/ml-explore/mlx),
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) and
[SwiftMath](https://github.com/mgriebling/SwiftMath) for their contributions to
the runtime, model format and rendering stack.
