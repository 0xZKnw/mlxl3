# MLXL3 Desktop 2.0.0-beta.17

Desktop prerelease **v2.0.0-beta.17**, internal version **2.0.0 / build 48**.
The bundled MLX engine remains **2.0.5**; there is no new engine release.

## Fixes

- Load the new MLXL3 icon directly at startup so the running app does not rely
  on a stale cached icon. The packaged icon also has a new resource name for
  macOS to distinguish it from the previous artwork.

## Installation and compatibility

Requires Apple Silicon and **macOS 26.2 or later**. Select the Desktop beta
channel in Settings or install the DMG manually. The app is ad-hoc signed and
not notarized. Models and conversations are not included in the download.

## Validation and known limits

The packaged app icon and startup assignment are tested with the real AppKit
image API, including a missing-resource fallback. The icon is checked across
light, dark and midnight blue themes. These are bounded tests, not a formal
proof; macOS controls its own icon cache and Finder behavior.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for reporting the stale icon,
and to [Apple MLX](https://github.com/ml-explore/mlx),
[ExLlamaV3](https://github.com/turboderp-org/exllamav3), and
[SwiftMath](https://github.com/mgriebling/SwiftMath) for their work on the
runtime, model format and rendering stack.
