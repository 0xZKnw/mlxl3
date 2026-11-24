# MLXL3 Desktop 2.0.21

**Inkling 2 Mini can see: attach a picture to your message and it reads it.**

Desktop **2.0.21 / build 74** bundles MLXL3 engine **2.0.29**: the same sources as [`engine-v2.0.29`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.29).

## New

### Pictures for Inkling 2 Mini

- **Attach, paste or drop pictures** (PNG, JPEG, WebP) in a conversation with Inkling 2 Mini, as with LFM3-VL. It reads text in pictures and describes what they show.
- **Native resolution.** Pictures up to 1920 × 1080 are read as they are; larger ones are scaled down.
- **Works with DSpark** and with the recommended sampling settings.
- **The recommended Inkling 2 Mini download** now installs the Hugging Face revision that ships the vision encoder ([`deac6c50`](https://huggingface.co/0xzknw/Inkling-2-Mini-EXL3-3.33bpw/tree/deac6c50ece340a4096a0e26f86664ed50cf0274)). It adds one 0.64 GB file; the rest of the model is unchanged. An Inkling 2 Mini downloaded before keeps working, text only, and Desktop says so if a conversation has pictures.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The Windows app is unchanged since 2.0.16.

## Unchanged

- Models, conversations, settings, Agent mode and Agent projects.
- Everything else is as in [2.0.20](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.20).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Engine 2.0.29:** see its [release notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.29) for the comparison with the reference vision encoder, the end-to-end checks and the limits.
- **Desktop checks:** the image-input check (now with Inkling 2 Mini with and without its vision encoder) and the workspace check (291) passed with the macOS 26.5 SDK; restricting the image gate to LFM3-VL is caught. The full Desktop suite was not run again for this release.
- **This DMG:**
  - `hdiutil verify` valid, `codesign --deep --strict` passed;
  - version 2.0.21 / build 74, built with the macOS 26.5 SDK;
  - `build-info` has commit `6ee0c5001` with no tracked changes;
  - the bundled engine reports 2.0.29 (`engine-v2.0.29`), and its five runtime files are identical to the published engine archive.
- **Bundled engine in the bridge:** reads a picture with Inkling 2 Mini, with the request Desktop sends by default and with DSpark.
- **Not verified:** the app was not driven by hand with a picture; visual quality beyond two synthetic test pictures.

## Source and download

- Source: [`6ee0c5001`](https://github.com/0xZKnw/mlxl3/commit/6ee0c5001a2c76a93cb9821ac63eedb9378ac0d7) (tag `v2.0.21`).
- `MLXL3-Desktop-v2.0.21-b74-Apple-Silicon.dmg`: 253,640,688 bytes, SHA-256 `82318d0fdb9032e49657e43c3422a4c6bb270efd7e7b668103cd3731b9fad9a9`.
- `MLXL3-Desktop-v2.0.21-b74-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.29, the same files as [`engine-v2.0.29`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.29).

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for asking Inkling 2 Mini to see.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini and its vision encoder, and to [Hugging Face Transformers](https://github.com/huggingface/transformers) for the reference implementation used to check it.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
