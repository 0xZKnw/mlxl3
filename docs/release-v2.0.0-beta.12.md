# MLXL3 Desktop 2.0.0-beta.12

Desktop prerelease **v2.0.0-beta.12**, internal version **2.0.0 / build 43**.
It keeps the MLX engine **2.0.5** and converter shipped with beta 11. This is a
Desktop update; no separate engine release is published.

## Changes

- Paste a copied image into the composer with **Command-V**. PNG, JPEG and WebP
  image data and copied image files use the existing attachment flow. TIFF data
  from the clipboard is converted to PNG before being attached. A loaded
  image-capable model is required; the existing image limits still apply.
- Open **Downloads** from the small button at the top left. The compact panel
  shows the model, progress, transfer speed and status, and offers pause or
  resume when available. It is also accessible from the model library and when
  the sidebar is hidden.

## Fixes

- Updated an installer test fixture to report the MLX backend required by the
  current runtime contract. This corrects the test, not the shipped engine.

## Installation and compatibility

Requires Apple Silicon and **macOS 26.2 or later**. Enable the Desktop beta
channel in Settings or install the DMG manually. The application is ad-hoc
signed and is not notarized. Models and conversations are not bundled.

## Validation and known limits

The Swift release build, image-import and history checks, download state tests,
installer checks and Desktop suite were run locally. The Desktop suite was
completed in two segments after fixing the pre-existing updater fixture. Image
checks cover copied data, TIFF conversion, validation, the simulated bridge,
conversation history and two rendered window widths; they are not a physical
model or general vision-quality evaluation. The download tracker includes 300
finite cases, and the panel was visually inspected in active, paused and failed
states. The system paste shortcut and popover were not manually exercised in a
running installed app. The full Python suite was interrupted after 198 passed
and 6 skips (PyTorch unavailable). No Swift source-level formal proof or new
performance claim is made. Source identity and artifact checksums accompany the
release assets.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the product direction and
feedback; [Apple MLX](https://github.com/ml-explore/mlx),
[LiquidAI](https://huggingface.co/LiquidAI),
[ExLlamaV3](https://github.com/turboderp-org/exllamav3),
[PonyExl3](https://github.com/beamivalice/PonyExl3) and
[SwiftMath](https://github.com/mgriebling/SwiftMath) for the foundations and
components retained in this release.
