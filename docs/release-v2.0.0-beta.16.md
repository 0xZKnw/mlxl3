# MLXL3 Desktop 2.0.0-beta.16

Desktop prerelease **v2.0.0-beta.16**, internal version **2.0.0 / build 47**.
The bundled MLX engine remains **2.0.5**; there is no new engine release.

## Changes

- MLXL3 has a new M3 logo. The Desktop icon uses a dark background and a large
  monogram for contrast in the Dock and Finder.
- The app interface uses a transparent version of the mark, with a dark tint
  in light themes. The website and Discord server also use the new identity.

## Installation and compatibility

Requires Apple Silicon and **macOS 26.2 or later**. Select the Desktop beta
channel in Settings or install the DMG manually. The app is ad-hoc signed and
not notarized. Models and conversations are not included in the download.

## Validation and known limits

The icon generation check compares the packaged ICNS artwork to its source.
The logo view is checked in light, dark and midnight blue themes, both with
source assets and with assets loaded from an app bundle. These are bounded
checks, not a formal proof; macOS may cache a previous icon after updating.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for providing the logo and
reporting the icon contrast issue, and to [Apple MLX](https://github.com/ml-explore/mlx),
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) and
[SwiftMath](https://github.com/mgriebling/SwiftMath) for their work on the
runtime, model format and rendering stack.
