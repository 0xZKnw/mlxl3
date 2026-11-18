# MLXL3 Engine 2.0.17

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release lets the Hugging Face downloader fetch **Liquid d1 decision models** in full, so Desktop 2.0.8 can offer them in its recommended models.

## Fixes

- **`hub download` keeps `calibration.json`.** d1 reads its answer temperatures from this file. Before, the downloader left it out of the files it fetches, and a downloaded d1 model then answered with neutral temperatures, without any error: the probabilities were not calibrated. The file is now taken from the variant's own folder (never from a parent folder, which may belong to another model).
- **`hub details` recognizes `lfm3_omni`.** d1-Omni checkpoints were shown as an unsupported architecture before download.
- `model-extra.safetensors` (d1-Omni's audio encoder and projector) was already fetched; a new test now covers it.

The checkpoint loader, decisions, chat, MTP and DFlash2 are unchanged from 2.0.16.

## New d1 checkpoints on Hugging Face

Our 8 bpw EXL3 conversions are now public, with the LFM Open License v1.0, a modification notice and the upstream model card:

- [`0xzknw/d1-2B-EXL3-8bpw`](https://huggingface.co/0xzknw/d1-2B-EXL3-8bpw), revision `0e99f8d8d54789a7dd022814e8844bdf14960d06`;
- [`0xzknw/d1-Omni-2B-EXL3-8bpw`](https://huggingface.co/0xzknw/d1-Omni-2B-EXL3-8bpw), revision `54ded3cfde24661248600f19e713e15f935144c9`.

Their weights are the exports validated in [engine 2.0.16](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.16), byte for byte.

```sh
mlxl3 hub download 0xzknw/d1-2B-EXL3-8bpw --revision 0e99f8d8d54789a7dd022814e8844bdf14960d06
```

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

Engine 2.0.16 runs these checkpoints too when their folder is complete; only its downloader missed `calibration.json`. A d1 folder downloaded with 2.0.16 can be fixed by copying `calibration.json` from the repository next to `config.json`.

## Verification and limits

- **Real downloads** with the binary built from this source, into an empty models folder and registry:
  - `hub details` lists one root variant for each repository: 2,726,317,096 bytes for d1-2B and 3,223,846,596 for d1-Omni, `architecture_supported: true` for `lfm3_vl` and `lfm3_omni`;
  - `hub download` installed both models; every downloaded file is byte-identical to the validated export (except `README.md`, the new model card), including `calibration.json` and d1-Omni's `model-extra.safetensors`;
  - `decide` gives byte-identical output on the local export and on the downloaded folder: d1-2B on a support ticket, d1-Omni on a recorded call (audio);
  - the same d1-2B folder without `calibration.json`, which is what 2.0.16 downloaded, gives different probabilities (billing 0.777 instead of 0.698).
- **New test** `decision_variants_keep_calibration_and_extra_weights_in_their_folder`. With the fix removed, it fails (calibration file missing); with the fix, it passes.
- **Chat unchanged:** greedy text identical between the published 2.0.16 binary and this one for LFM3-1.7B (French prompt) and LFM3-VL-2B with a picture. Speeds were single runs and are not compared.
- **Released archive, extracted into a new directory:**
  - its four files match the manifest: release tag `engine-v2.0.17`, version 2.0.17, minimum Desktop 1.4.0, minimum macOS 26.2.0;
  - the three Mach-O files pass strict signature and arm64 checks;
  - `mlx.metallib` is identical to MLX 0.32.2;
  - `runtime-info` reports version 2.0.17, MLX, chat and source `6f8e4b3d2ee6`.
- **Local checks on the release source:**
  - Rust formatting passed.
  - Strict Clippy (`-D warnings`, all targets) passed for `chat`, `mlx,chat` and `direct-metal,chat`.
  - CPU tests: 602 passed, 10 ignored.
  - MLX tests: 665 passed, 312 ignored, none failed. The ignored tests are physical-GPU or fixture tests.
- **Not verified:**
  - Kani was not run.
  - Direct-Metal tests were not rerun (only `hub.rs` changed since 2.0.16, and Clippy passed for that build).
  - Only an Apple M5 MacBook with 24 GB was used.
  - Hosted CI on this commit had not completed when this was written.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the request and the release decision. Thanks to [Liquid AI](https://huggingface.co/LiquidAI) for d1-2B, d1-Omni-2B and their calibration, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format and converter, [Hugging Face](https://huggingface.co) for hosting the checkpoints, and [Apple MLX](https://github.com/ml-explore/mlx) and [PonyExl3](https://github.com/beamivalice/PonyExl3) for the array runtime and the Metal EXL3 port.

## Source and assets

Engine source commit `6f8e4b3d2ee604c3cb2fffa60c65dba7c6ee07f1`: engine 2.0.16 (`40440b0bc`) plus the downloader fix, the Desktop 2.0.8 changes and the version bump, on [PR 180](https://github.com/0xZKnw/mlxl3/pull/180).

- `MLXL3-Engine-v2.0.17-arm64.tar.gz`: 68,865,911 bytes, SHA-256 `cb28265b1500bd2384ca99b70e759d718ca3ce5b8e6510d87f55a6c5d58f9249`.
- `MLXL3-Engine-v2.0.17-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `6e281e6babac5cb13bf07f8ea7285c591da5e6235b5eb0ef9dc5ddcce6910858`.
- `mlxl3`: 14,556,064 bytes, SHA-256 `7c6aefb85f45bd19a12ba12e1cb2f86636682f4da3a22148763bbf07239f9ec2`.
- Detailed record: `opti.md`, entry `RELEASE-ENGINE-2.0.17-DESKTOP-2.0.8`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
