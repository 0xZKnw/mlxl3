# MLXL3 Desktop 2.0.0-beta.20 · Polished controls

Desktop prerelease **v2.0.0-beta.20**, internal version **2.0.0 / build 51**.
This update refines Aurora's controls and settings. The DMG includes the published **MLX engine 2.0.9-beta.1**.

## Changes and fixes

- The model selector keeps a consistent capsule shape when hovered. Its name and status use the interface's neutral text colors.
- MTP, DFlash, response appearance and generation controls share soft, rounded hover feedback within the glass toolbar. Selected, pressed and disabled controls retain distinct states.
- New conversation is a full-width action in the sidebar. In Agent mode it creates a new task. Command-N remains available.
- The Generation inspector uses continuous sections for sampling, MTP and the other model settings. Its close button is compact and neutral.
- Settings groups use a subtle translucent surface that follows the selected theme.

## Installation and compatibility

Requires **Apple Silicon and macOS 26.2 or later**. Select the Desktop beta channel in Settings or open the DMG and drag MLXL3 Desktop to Applications. The application uses an ad-hoc signature and has no Apple notarization.

## Artifact and source

- Source commit `9cd98149c2685d99e88ae842d1462b463e30247a`
- File `MLXL3-Desktop-v2.0.0-beta.20-b51-Apple-Silicon.dmg`
- Size **249,238,127 bytes**
- SHA-256 `36e9fa49ffbf5779c1356043413f78a0e7b7f44917171bd28056682b97cbc82d`
- Bundled runtime [engine-v2.0.9-beta.1](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.9-beta.1). Its four components match the published manifest.
- The bundled converter retains its beta 19 build. All 107 recorded converter source hashes match this release's sources.

## Validation and limits

The complete Desktop end-to-end suite passed locally, including 2,324 Studio mode checks, 391 preference assertions, 24 MTP toolbar and panel cases, 2,240 responsiveness checks and 83 image-input checks. The Studios suites passed 39 Python tests, 58 native checks and 44 workspace checks.

New regressions exercise the actual new-conversation row at both edges, opening and closing the inspector while retaining the draft, and settings surfaces over distinct backgrounds in light and dark themes. The previous opaque settings component fails the new surface check. Native screenshots cover six hover targets, including enabled MTP and DFlash controls, with an amber accent and both appearance modes.

Swift 6 strict compilation and compiler SIL verification passed. These are compiler and sampled runtime checks, with no formal proof of SwiftUI behavior. Inference throughput and GPU power are outside this UI validation.

The mounted DMG passed the production updater's preflight checks. App identity, signature, arm64 executable, build number, bundled runtime hashes, icons and math fonts were checked. The bundled quantizer inspected a 16,384-parameter local fixture. Hosted workflow results are available from [GitHub Actions](https://github.com/0xZKnw/mlxl3/actions).

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the visual feedback and the original aurora reference in [Locked](https://github.com/0xZKnw/locked). Thanks to [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, model format and math-rendering stack.
