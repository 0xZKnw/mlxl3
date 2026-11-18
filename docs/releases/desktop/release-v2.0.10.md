# MLXL3 Desktop 2.0.10

**Fixes Tune with sampling and with near ties: in 2.0.9 it could fail with "Incomplete or cancelled test".**

Desktop **2.0.10 / build 63** bundles MLXL3 engine **2.0.19**, the same engine as 2.0.9 (the published binary, byte for byte).

## Fixes

### Tune keeps the result the engine measured

- **The bug.** In 2.0.9 the app recomputed the best mode itself and still required every mode to produce exactly the same tokens as the baseline. With engine 2.0.19 that cannot hold in two cases:
  - **sampled Tunes**, which measure with your sampling settings and draw different text on every run;
  - **Auto in greedy mode**, whose divergence on a tie between two equally likely tokens the engine verifies and accepts.

  The app then picked another mode than the engine and dropped the result with **"Incomplete or cancelled test. Setting preserved."**
- **Who was affected.** This hit DFlash2 Tune and MTP Tune, including the automatic MTP Tune, whenever a sampling preset was active.
- **The fix.** The app now accepts different tokens only for the modes the engine itself reports as sampled or as a verified near tie. Every other mode must still match the baseline, so results from older engines are checked exactly as before.

### Clearer DFlash2 panel after a failed Tune

- A Tune error now reads **"Active · last Tune needs attention"** while DFlash2 keeps running.
- **Retry activation** only appears when DFlash2 is not active. In 2.0.9 a failed Tune looked like a failed activation.

## Unchanged

- Everything else is as in [2.0.9](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.9): the DFlash2 panel, sampling with DFlash2 and Tune, and the engine.
- Conversations, settings, models and saved Tune results are kept. A Tune that failed in 2.0.9 left the previous setting in place; run **Compare speeds** again to save the new result.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Reproduced before the fix.** The new cases fail against the 2.0.9 rule ("Near tie rejected") and pass with the fix:
  - a near-tie row with its own token hashes is accepted;
  - an empty near-tie list is not;
  - a sampled Tune is accepted;
  - an unflagged divergent row and an ineligible sampled row are still refused.
- **Desktop checks** (`scripts/check-desktop.sh`, full suite, Xcode toolchain with the macOS 26.5 SDK): every check passed, including speculative sampling (10,009 assertions over 1,944 capability and intent states).
  - The fake engine now reports realistic token hashes for sampled and near-tie rows.
  - A sampled MTP Tune and a sampled DFlash2 Tune both complete without error and keep the engine's mode.
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum;
  - after mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.10 / 63;
  - its `build-info.json` names source `de4a5920c740`, with no uncommitted changes, and engine 2.0.19;
  - the bundled runtime folder is identical to 2.0.9's, and its `mlxl3` is **byte-identical** to the published engine 2.0.19 binary (SHA-256 `127c6691…783a`). No engine file changed since 2.0.19, so the app reuses that binary instead of rebuilding it.
- **Not verified:** the fixed Tune was not run by hand in the shipped app with a real model (it was checked with the fake engine above; the engine's Tune itself was checked at the bridge for 2.0.19). SwiftUI behavior has no formal proof.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. It contains none of these changes: the Windows app uses its own engine and has no DFlash2.

## Source and download

- Compiled source `de4a5920c740f0727976e7fb52de6e87b338f4f6`.
- `MLXL3-Desktop-v2.0.10-b63-Apple-Silicon.dmg` · **242,871,627 bytes**, SHA-256 `57da4f60af6384f13559e03d4d1c3571fc8bd88dc6a092855af5a3cbd148d0d9`.
- `MLXL3-Desktop-v2.0.10-b63-Apple-Silicon.dmg.sha256` · 110 bytes, SHA-256 `d4344a78cbbc873bf387d3e2244a86568c9981130339973c19c8df9ac31cbda9`.
- Bundled engine 2.0.19 (`mlxl3` SHA-256 `127c6691e727fd7e5379dca21dd42ec81f10dbf0bdbaa9d79ae6a9f16c2f783a`, source `2d74600037a8`). The Swift app was built with the Xcode toolchain and the Command Line Tools macOS 26.5 SDK.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for reporting the failed Tune with a screenshot, and for the release decision.
- Thanks to [incoai Splash](https://github.com/incoai/splash) for the DFlash2 draft.
- Thanks also to [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3) and [SwiftMath](https://github.com/mgriebling/SwiftMath), for the runtime, the EXL3 format, the Metal EXL3 port, and math rendering.
