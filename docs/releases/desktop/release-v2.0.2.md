# MLXL3 Desktop 2.0.2

**Automatic MTP tuning, a one-click unload button, and Flash Next on 16 GiB Macs.**

Desktop **2.0.2 / build 55**, bundling MLXL3 engine **2.0.11 beta 3**.

## New

- **Automatic MTP tuning.** When a model with an MTP head loads and no Tune result is stored for its configuration (model, head and engine), Desktop runs Tune once in the background, saves the fastest depth and applies it. Later loads with the same configuration reuse the result. A new engine or head tunes again.
  - The composer stays usable during the Tune. Sending a message pauses the Tune, sends, and the Tune resumes when the app is idle.
  - Unloading or switching models interrupts the Tune without an error. If the Tune fails, the current depth is kept and the same configuration is not retried in that session.
  - Manual Tune in the MTP settings works as before and overrides the automatic result.
  - To turn the automatic Tune off, run `defaults write io.mlxl3.desktop studio.mtpAutoTune -bool false` and restart the app.
- **Unload button next to the model selector.** The model selector and an eject button now share one capsule, like the MTP and DFlash2 controls.
  - The button appears when a model is loaded or loading. While loading, it cancels the load.
  - It frees the engine's memory and keeps the model selected, so choosing it again reloads it.
  - It is disabled while a reply, a manual Tune, document preparation, the local server or a mode switch is using the engine. An automatic Tune does not block it.

## Engine

The bundled engine is [2.0.11 beta 3](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.11-beta.3). Compared with the 2.0.10 engine in Desktop 2.0.1, it includes:

- the 2.0.11 beta 1 MTP head layout and layer pipeline;
- the beta 2 MTP defaults: post-norm drafting and confidence-based verify truncation. On Apple M5 with Qwen3.6-35B-A3B, beta 2 measured +18.1% (French prompt) and +11.8% (code prompt) MTP3 decode throughput against beta 1, with identical tokens. See the [beta 2 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.11-beta.2) for the protocol and limits;
- the beta 3 Flash Next fix: the memory reserve now scales with the Mac's RAM. A 16 GiB Mac needs 7.56 GiB of available memory instead of 8.56 GiB, and the refusal message states the shortfall and the `MLXL3_SSD_CACHE_GIB` workaround.

Each engine figure comes from its own release campaign. They are not added together, and no new measurement was made for this Desktop build.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- The user asked to skip the remaining tests for this release, so the Desktop check suite was stopped before its end. Before it stopped, `scripts/check-desktop.sh` passed 18 groups on the integrated source, including:
  - the new unload control check (40 assertions: state table, capsule layout and the real app and bridge lifecycle with a fixture engine);
  - the new automatic Tune check (35 checks: first Tune, reuse, manual override, key change, failure, chat preemption, unload from the toolbar during a Tune, model switch, explicit stop and opt-out);
  - model idle unload (120 assertions), Studio modes (2,324), composer updates, MTP and DFlash2 controls, Flash downloads, agent sandbox (25) and agent protocol (29).
- Not run for this release: the remaining check-desktop groups (preferences, response UI, syntax and streaming), the Studios and workspace suites, the updater preflight from 2.0.1 and native page captures.
- Earlier, on their own branches, the unload button passed the full `check-desktop.sh` except one response-UI pixel check that also fails intermittently on the previous source. The automatic Tune was checked with an exhaustive model of its decision ledger.
- The release build compiled Swift 6, signed the app, and ran the bundled engine. The DMG checks listed under Source and download were run.
- **Not verified:** a real click on the unload button with a model loaded on the GPU; the automatic Tune with a real model; Flash Next loading on the GPU and on an 8 or 16 GiB Mac; Macs other than Apple M5 and macOS 27.2. SwiftUI behavior has no formal proof. Hosted CI results are separate from local validation.

## Source and download

- Compiled source `22ac280263d7d6cbdd7c7f96bc4ca67b26d714e5`.
- `MLXL3-Desktop-v2.0.2-b55-Apple-Silicon.dmg` · **249,490,905 bytes**.
- SHA-256 `0f93bc5c59a9f32c815486c3ecaf5b39f7523860391d5a147176e7d80b7d8ce0`.
- `hdiutil verify` reported a valid checksum. The mounted app passed `codesign --verify --deep --strict` and is identical to the built app. The bundled engine is byte-identical to the [engine 2.0.11 beta 3](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.11-beta.3) archive.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the feature requests, the Flash Next report and the release decisions, and to the user who reported the Flash Next refusal. Thanks to [Apple MLX](https://github.com/ml-explore/mlx), [Qwen](https://huggingface.co/Qwen), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [MTPLX](https://github.com/youssofal/mtplx), [TensorFold](https://github.com/ashhart/TensorFold) and [SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, models, model format, MTP research and math rendering.
