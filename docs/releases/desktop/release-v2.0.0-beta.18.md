# MLXL3 Desktop 2.0.0-beta.18

Desktop prerelease **v2.0.0-beta.18**, internal version **2.0.0 / build 49**.
The app bundles the published **MLX engine 2.0.6-beta.1**; this is a Desktop
release and does not publish a new engine.

## Changes

- Lab compares each model's latest decode speed and charts recorded generation
  history. Choose a model and metric for the history, including decode, prefill,
  latency, token counts and process memory where recorded. Measurements come
  from Chat and Agent; they are not controlled benchmarks.
- The server page adds context and port settings, local-network access and a
  copyable API test command. Local access remains the default; network access
  requires the generated API key and uses unencrypted HTTP.
- Model details keep the architecture and memory estimate in a compact,
  expandable panel, leaving more room for the model card.

## Installation and compatibility

Requires Apple Silicon and **macOS 26.2 or later**. Select the Desktop beta
channel in Settings or install the DMG manually. The application is ad-hoc
signed and not notarized. Models and conversations are not included in the DMG.

## Artifact and source

- Source commit: `50a51b26b2c281608080d0c0ce3ec82117d39d87`
- File: `MLXL3-Desktop-v2.0.0-beta.18-b49-Apple-Silicon.dmg`
- Size: **249,926,072 bytes**
- SHA-256: `b211307b7e6edaffcb76d399f762373ae1aa978888c72fb5f40f1129e0d6aee0`
- Bundled engine: `engine-v2.0.6-beta.1`; its component hashes match the
  published engine manifest.

## Validation and limits

The Desktop end-to-end suite passed locally, including 2,324 Studio mode
checks and model-card layout checks at 480 and 840 points. The DMG checksum,
mounted app signature, arm64 executable, app version, build, source identity,
engine component hashes, icon and quantizer were checked. Server lifecycle and
API authentication were exercised with a local test engine. The Lab preview
used synthetic data. Physical GPU inference and a source-level proof of these
Swift changes were not performed; these checks do not prove all behavior.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the design feedback and
feature requests, and to [Apple MLX](https://github.com/ml-explore/mlx),
[ExLlamaV3](https://github.com/turboderp-org/exllamav3), and
[SwiftMath](https://github.com/mgriebling/SwiftMath) for their work on the
runtime, model format and rendering stack.
