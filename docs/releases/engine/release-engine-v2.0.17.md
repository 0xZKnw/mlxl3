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

VERIFICATION_PLACEHOLDER

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the request and the release decision. Thanks to [Liquid AI](https://huggingface.co/LiquidAI) for d1-2B, d1-Omni-2B and their calibration, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format and converter, [Hugging Face](https://huggingface.co) for hosting the checkpoints, and [Apple MLX](https://github.com/ml-explore/mlx) and [PonyExl3](https://github.com/beamivalice/PonyExl3) for the array runtime and the Metal EXL3 port.

## Source and assets

SOURCE_PLACEHOLDER
