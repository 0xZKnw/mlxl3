# MLXL3 Desktop 2.0.0-beta.19 · Aurora

Desktop beta 19 (build 50) brings an animated aurora to the workspace, native Liquid Glass to the model controls, and a redesigned generation inspector. The DMG includes the previously published MLX engine 2.0.9-beta.1.

## Workspace and settings

- Five bands of light drift independently across the top of the window. When a response starts, the bands descend and accelerate to 4.4× their resting speed. They ease back over 2.6 seconds when it finishes. The reading area keeps the selected theme's background. Reduce Motion displays a still aurora.
- Chat and Agent open with the MLXL3 monogram, a serif title and a roomier composer. Start a new conversation from the history header or use Command-N.
- The Generation inspector shows context usage, the model's maximum and its full recommended token limit. Select the recommendation, then apply it and reload the model. The inspector floats over compact windows and sits alongside the conversation in wide windows.
- All seven Settings pages share the workspace navigation and theme. The appearance picker previews the aurora, sidebar and composer for each theme.

## Rendering cost

On an Apple M5 with 24 GiB of memory, on battery, the release design used **1.31% of one CPU core** at visible idle, **2.50%** during simulated generation, and **0.009%** with the window hidden. Mean process footprint was **56.7 MiB** at idle and **69.0 MiB** during simulated generation, respectively about 23 and 35 MiB above the previous interface.

The comparison used the production workspace in a 1024 × 640 pt Retina window, identical fixture content, and two seven-second samples per state and version. These measurements cover the interface. GPU power, battery life and inference throughput were outside this test. [Measurements and protocol](https://github.com/0xZKnw/mlxl3/blob/1766a04c89a308400f87b90711d43c298190f172/docs/measurements/desktop-aurora-20261113.json)

## Installation and compatibility

Requires **Apple Silicon and macOS 26.2 or later**. Select the Desktop beta channel in Settings or open the DMG and drag MLXL3 Desktop to Applications. The bundle uses an ad-hoc signature and has no Apple notarization. The [bundled engine release](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.9-beta.1) has its own update channel.

## Artifact

- Desktop source commit `1766a04c89a308400f87b90711d43c298190f172`
- `MLXL3-Desktop-v2.0.0-beta.19-b50-Apple-Silicon.dmg`, **250,033,181 bytes**
- SHA-256 `6511e05bc7c46197e40df77758d07d9af71681896c8df2bde8fcc61ad2d47e23`

## Verification

The local Desktop end-to-end suite passed, including conversation navigation, settings, streaming, model controls and workspace flows. Visual checks covered compact light and wide dark windows. Animation tests covered acceleration, settling and twelve rapid restarts.

The production updater mounted and validated the release DMG. On the mounted app we checked the signature, build identity, arm64 executable, all four published engine components, icons and math fonts. The bundled quantizer inspected a 16,384-parameter fixture. The bundled engine completed two real LFM3-1.7B EXL3 8 bpw chat turns on Apple Silicon and recalled the code given in the first turn.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3's visual direction and the original aurora motion in [Locked](https://github.com/0xZKnw/locked). Thanks to [Apple MLX](https://github.com/ml-explore/mlx) for the inference runtime, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the model format and [SwiftMath](https://github.com/mgriebling/SwiftMath) for math rendering.
