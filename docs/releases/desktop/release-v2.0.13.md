# MLXL3 Desktop 2.0.13

**DSpark gets a toolbar control like MTP and DFlash2, and Inkling 2 Mini downloads 1.2 GB less with the same answers.**

Desktop **2.0.13 / build 66** bundles MLXL3 engine **2.0.22**.

## New

### A DSpark control like MTP and DFlash2

- **Toolbar.** Models with a DSpark drafter, such as Inkling 2 Mini, now show a **DSpark On/Off** control next to the model picker. It opens the DSpark settings, as the MTP and DFlash2 controls open theirs.
  - It replaces the MTP control, which Inkling 2 Mini showed greyed out because the model has no MTP head.
  - Other models keep the MTP control exactly as before.
- **Settings.** The toolbar popover and the inspector card now share one panel:
  - an **Enable DSpark** switch;
  - its state: Ready, Off, and during an answer Active or Ordinary decoding;
  - the acceptance of the latest answer that used DSpark, for example `63/100 proposals accepted (63%) · 1.63 tokens per block`;
  - a short explanation of the drafter.

  As with MTP and DFlash2, the switch is locked while an answer is being generated, and the choice is remembered.

### A smaller Inkling 2 Mini

- **1.2 GB less, same answers.** The recommended Inkling 2 Mini now installs revision `a5763a01` of [`0xzknw/Inkling-2-Mini-EXL3-3.33bpw`](https://huggingface.co/0xzknw/Inkling-2-Mini-EXL3-3.33bpw): 13.54 GB of weights instead of 14.75 GB.
  - It drops the vision encoder, which MLXL3 does not run.
  - It stores the output head and the DSpark drafter in the 8-bit form the engine already used at load.
  - Answers, DSpark acceptance and quality are identical to the previous files, bit for bit.
- **What you need to do.** Download the model again from the recommended models.
  - The previous version stays on your disk until you remove it, and still works with this release.
  - Desktop 2.0.12 keeps installing the previous version, which it pins.
- **Engine 2.0.22**, bundled here, is required to load the smaller files.

## Unchanged

- Everything else is as in [2.0.12](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.12).
- Conversations, settings and models are kept.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Desktop checks** (`scripts/check-desktop.sh`, the full suite, Command Line Tools toolchain with the macOS 26.5 SDK): passed in 502 s.
  - The MTP and DFlash2 layout check now covers DSpark in 78 cases, FR and EN, 200–320 pt:
    - which control the toolbar shows: DSpark for a model with a drafter, in place of a disabled MTP control only when the model has no MTP head, and MTP unchanged for other models;
    - the toolbar control's compact size;
    - the panel's height, 105–135 pt.
  - Forcing the MTP control to always show is caught by the check.
- **Engine and model**, from the [engine 2.0.22 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.22):
  - Against the previous files, the smaller Inkling 2 Mini gives the same logits, drafter proposals, DSpark acceptance, KL and greedy bridge answers, bit for bit.
  - Engine 2.0.21 refuses it, and engine 2.0.22 still loads the previous files.
- **Model on Hugging Face:** the 21 files of revision `a5763a01` match the local files (SHA-256 of the weights; the card and configs are byte-identical).
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum;
  - after mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.13 / 66;
  - its `build-info.json` names source `57699d7e5e7a`, with no uncommitted changes, and engine 2.0.22;
  - the bundled engine reports 2.0.22. Its `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib` are byte-identical to the engine 2.0.22 archive. Its `mlxl3` was built from the same engine source; only the embedded source commit and signature differ, 196 bytes;
  - the bundled engine, run from the mounted DMG, passed `scripts/smoke-release.py` on the smaller Inkling 2 Mini: two turns through the Desktop protocol, the second recalls a code given in the first.
- **Not verified:**
  - The new DSpark control was not used by hand in the shipped app; the layout and visibility were checked with the test harness.
  - A full download of the new model revision through the app was not repeated.
  - SwiftUI behaviour has no formal proof.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. It contains none of these changes: the Windows app uses its own engine and does not run Inkling 2 Mini.

## Source and download

- Compiled source `57699d7e5e7ae334cabb1c7cf94649bc97c8e857`: `main` at `114f4a0da` (engine 2.0.22) plus the Desktop changes.
- `MLXL3-Desktop-v2.0.13-b66-Apple-Silicon.dmg` · **250,548,087 bytes**, SHA-256 `17bf981990f7d27b2fc7b328297de6fedeeac71be5ec586ddcfa7ccb0cfccb3d`.
- `MLXL3-Desktop-v2.0.13-b66-Apple-Silicon.dmg.sha256` · 110 bytes, SHA-256 `6421fa99140d11b93f110cb80ffebfa5a4b8b0ffbdb3c59cf2f7bd18c2a38a21`.
- Bundled engine 2.0.22 (`mlxl3` SHA-256 `a05a102bfc0e7cfdf5ed3356681667a8b648935b25765184f95b180654fcfe4f`, source `57699d7e5e7a`). The Swift app was built with the Command Line Tools toolchain and its macOS 26.5 SDK.
- Model: `0xzknw/Inkling-2-Mini-EXL3-3.33bpw` at revision `a5763a01f6e34c84bdf1a5812c890ff3978d5d75`.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for pointing out that DSpark was not shown like DFlash2 in the app, and for the smaller Inkling 2 Mini and the release decision.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini and its DSpark drafter.
- Thanks also to the projects behind the formats and runtime:
  - [ExLlamaV3](https://github.com/turboderp-org/exllamav3), for the EXL3 format and quantizer;
  - [Apple MLX](https://github.com/ml-explore/mlx), for the runtime and its affine quantization;
  - [PonyExl3](https://github.com/beamivalice/PonyExl3), for the Metal EXL3 port;
  - [SwiftMath](https://github.com/mgriebling/SwiftMath), for math rendering.
