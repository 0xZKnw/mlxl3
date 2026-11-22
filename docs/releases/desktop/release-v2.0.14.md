# MLXL3 Desktop 2.0.14

**Inkling 2 Mini downloads another 434 MB less: 13.10 GB of weights, with a quality difference within measurement noise.**

Desktop **2.0.14 / build 67** bundles MLXL3 engine **2.0.23**.

## New

### An even smaller Inkling 2 Mini

- **434 MB less.** The recommended Inkling 2 Mini now installs revision `dde31cd0` of [`0xzknw/Inkling-2-Mini-EXL3-3.33bpw`](https://huggingface.co/0xzknw/Inkling-2-Mini-EXL3-3.33bpw). Its token embedding table is stored in 8 bits.
  - Weights: 13.10 GB, against 13.54 GB in Desktop 2.0.13 and 14.75 GB in 2.0.12.
- **Quality.** Against an exact reference of the original model, on 6,144 held-out positions:
  - KL changes by +0.00019 nats on prose (95% interval −0.00023 to +0.00063);
  - it changes by +0.0006 overall (−0.0038 to +0.0050);
  - top-1 agreement is 94.68% and 94.76% overall.

  No difference is significant. See the [engine 2.0.23 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.23) for every domain.
- **What you need to do.** Download the model again from the recommended models.
  - The previous version stays on your disk until you remove it, and still works with this release.
  - Desktop 2.0.13 and 2.0.12 keep installing the versions they pin.
- **Engine 2.0.23**, bundled here, is required to load the new files.

## Unchanged

- Everything else is as in [2.0.13](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.13), including the DSpark toolbar control.
- Conversations, settings and models are kept.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Desktop checks** (`scripts/check-desktop.sh`, the full suite, Command Line Tools toolchain with the macOS 26.5 SDK): passed. The end-to-end checks took 438 s, and the layout check passed its 78 cases, including DSpark.
- **Engine and model**, from the [engine 2.0.23 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.23):
  - the new files give the same results bit for bit as the previous files with their embedding table rounded to 8 bits at load, the measured setting;
  - engine 2.0.22 refuses them, and engine 2.0.23 still loads the previous revisions.
- **Model on Hugging Face:** the 21 files of revision `dde31cd0` match the local files (SHA-256 of the weights; the card and configs are byte-identical).
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum;
  - after mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.14 / 67;
  - its `build-info.json` names source `b905ab641e75`, with no uncommitted changes, and engine 2.0.23;
  - the bundled engine reports 2.0.23. Its `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib` are byte-identical to the engine 2.0.23 archive. Its `mlxl3` was built from the same engine source; only the embedded source commit and signature differ, 197 bytes;
  - the bundled engine, run from the mounted DMG, passed `scripts/smoke-release.py` on the new files: two turns, the second recalls a code given in the first.
- **Not verified:**
  - A full download of the new revision through the app was not repeated.
  - The 8-bit embeddings change the outputs slightly. Their cost was measured on 12 rows (English, French, code and one agentic transcript), and no task benchmark was run.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. It contains none of these changes: the Windows app uses its own engine and does not run Inkling 2 Mini.

## Source and download

- Compiled source `b905ab641e75e1d389a979bd547b48cbfb42cd89`: `main` at `e052db54b` (engine 2.0.23) plus the revision pin and versions.
- `MLXL3-Desktop-v2.0.14-b67-Apple-Silicon.dmg` · **251,056,703 bytes**, SHA-256 `d56e87df1a85267dbaf6905b8546257208e671095187b67b02778f3d5c24145e`.
- `MLXL3-Desktop-v2.0.14-b67-Apple-Silicon.dmg.sha256` · 110 bytes, SHA-256 `e0024dde218a9ea4807bed6cea3bee00c5ab32c5068a40310a37b91baca2a21c`.
- Bundled engine 2.0.23 (`mlxl3` SHA-256 `20ea49982e07ac17717895943cb45c82c6ba76e213822db2633af2f2e4539f0e`, source `b905ab641e75`). The Swift app was built with the Command Line Tools toolchain and its macOS 26.5 SDK.
- Model: `0xzknw/Inkling-2-Mini-EXL3-3.33bpw` at revision `dde31cd0aea4ca0f61eaa5649559ef01e5baf25a`.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for asking to measure and then ship the 8-bit embeddings, and for the release decision.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini and its DSpark drafter.
- Thanks also to the projects behind the formats and runtime:
  - [ExLlamaV3](https://github.com/turboderp-org/exllamav3), for the EXL3 format and quantizer;
  - [Apple MLX](https://github.com/ml-explore/mlx), for the runtime and its affine quantization;
  - [PonyExl3](https://github.com/beamivalice/PonyExl3), for the Metal EXL3 port;
  - [SwiftMath](https://github.com/mgriebling/SwiftMath), for math rendering.
