# MLXL3 Engine 2.0.29

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release lets **Inkling 2 Mini see**: it reads pictures with the model's own vision encoder, in chat, through the bridge and with the DSpark drafter.

## New

### Image input for Inkling 2 Mini

- **The vision encoder runs in the engine.** It follows Hugging Face Transformers' `Inkling2VisionModel`: 20-pixel patches with a class token per picture, learned positions resampled to the picture, a 2D rotary embedding, 23 layers, then 2 × 2 patches merged and projected into one language-model token. Its 0.64 GB of BF16 weights load with the first picture.
- **Pictures at their native resolution.** One image token per 40 × 40 pixels, as the reference processor does, with transparent pixels composited on its grey checkerboard. A picture larger than 1920 × 1080 (1,296 tokens) is scaled down, keeping its aspect ratio; `MLXL3_INKLING_IMAGE_MAX_PIXELS` changes that cap. The conversation limits are unchanged: 8 pictures, 16.7 megapixels and 16,384 image tokens.
- **Same prompt as the reference.** Chat messages keep their text and image parts for Inkling's chat template, and the image placeholder is repeated once per image token, as in `Inkling2Processor`.
- **Works with DSpark and sampling.** The image features replace the placeholder embeddings during the prefill, including when a prefill chunk cuts through a picture.
- **The ready event reports `images_supported: true`** only for an Inkling 2 Mini checkpoint that ships its vision encoder: the Hugging Face revision [`deac6c50`](https://huggingface.co/0xzknw/Inkling-2-Mini-EXL3-3.33bpw/tree/deac6c50ece340a4096a0e26f86664ed50cf0274) or later. `MLXL3_INKLING_VISION=0` keeps the model text only.
- **Text-only prompts are unchanged** and keep the same code path, prompt cache and prompt lookup.

### FP32 activations for the encoder

On the M5 GPU, MLX 0.32.2 computes FP32 matrix products with reduced-precision kernels (7.7e-4 relative error on one product, against 5.7e-7 on the CPU). The encoder therefore keeps BF16 weights in memory but computes its activations in FP32: its features are **0.40 %** from the exact FP32 result, against 2.4 % with BF16 activations and 1.9 % for the reference model itself in BF16. Encoding a 1920 × 1080 picture takes about 1.5 s (1.24 s in BF16).

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- Desktop 2.0.21 bundles this engine and lets you attach pictures to Inkling 2 Mini. Older Desktop versions only accept pictures for LFM3-VL.
- Image input needs the Inkling 2 Mini revision that ships the vision encoder; earlier revisions keep working, text only.
- Requests behave as with 2.0.28. A conversation that contains a picture is not saved in the prompt cache or resumed from it, and its pictures are encoded again at each turn.

## Verification and limits

- **Reference:** Hugging Face Transformers 5.24.0.dev0 (`Inkling2ImageProcessor`, `Inkling2VisionModel` with the checkpoint's tensors) on the CPU, on a 210 × 130 test picture (24 image tokens, 52 prompt tokens):
  - pixel values identical to the bit (115,200 values) and prompt tokens identical;
  - with exact FP32 matrix products (`MLX_METAL_GPU_ARCH=applegpu_g16s`), the engine's features differ from the FP32 reference by **5e-6** (relative L2);
  - with this release's default (FP32 activations, M5 kernels): **0.40 %** from the FP32 reference.
  - The criterion fixed before the first measurement for a BF16 encoder (within 2 % of the BF16 reference) failed at 2.28 %; that is why the encoder computes in FP32.
- **Language model:** the image features move the logits by 8.2 % against plain placeholders; a prefill in 7-token chunks that cut through the picture gives 0.14 % difference and the same top token; text-only logits are identical to the bit before and after an image prompt; prompts whose placeholders do not match their pictures are refused.
- **End to end:**
  - `mlxl3-rs run --image`, greedy: the model reads "MLXL3 sees / 42 + 17 = ?" exactly and answers 59;
  - in the bridge, with DSpark, greedy: it describes the test picture correctly (the gradient's four corners, a red square on the left, a blue disc on the right);
  - in the bridge, with DSpark at temperature 1, two turns with the picture in the history: answer "59".
- **Released archive, extracted into a new directory:** its four files match the manifest (`engine-v2.0.29`, version 2.0.29, minimum Desktop 1.4.0, minimum macOS 26.2.0); the three Mach-O files pass strict signature and arm64 checks; `mlx.metallib` is identical to MLX 0.32.2's; `runtime-info` reports 2.0.29 and source `6ee0c5001a2c`.
- **Released archive in the bridge**, without any environment variable, with the Inkling 2 Mini revision `deac6c50`: the ready event reports `images_supported: true`; on the text picture, the request Desktop sends by default (temperature 1, prompt lookup) answers "Text: MLXL3 sees / 42 + 17 = ? — Answer: 59", and the same request with DSpark (133 blocks) gives the same reading and answer.
- **Tests:** CPU tests of the preprocessing, patch order, padding, checkerboard, caps, placeholders and rotary and position tables; GPU tests against the reference (ignored by default). Three mutations are caught: rows and columns swapped in the rotary embedding, `align_corners=True` position resampling, and the Desktop image gate restricted to LFM3-VL. Before this release: Rust `mlx,chat` 586 tests passed and 351 ignored in the library, 68 passed and 14 ignored in the binary, 0 failed; strict Clippy for `mlx,chat`, `chat` and `direct-metal,chat`; MLX source inventory of 343 files.
- **Not verified:** visual benchmarks (only two synthetic pictures were used); pictures larger than the default cap at native resolution; the test suites were not run again on the version commit.

## Source and download

- Compiled source `6ee0c5001a2c76a93cb9821ac63eedb9378ac0d7`: `main` at `d87a9ff8d` (Desktop 2.0.20) with the Inkling vision work (`65c766f67`) and the versions.
- `MLXL3-Engine-v2.0.29-arm64.tar.gz` · **69,170,592 bytes**, SHA-256 `7c6cec5f233f84f1335ceeeae5d29fc05bb1714bcce34840f32d22de288ce88b`.
- `MLXL3-Engine-v2.0.29-arm64.tar.gz.sha256` · 100 bytes.
- `mlxl3` · 15,243,920 bytes, SHA-256 `890a411277b2bf041442321809f6f08995e232ec3eadea0daeb321403e376a15`.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for asking Inkling 2 Mini to see.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini and its vision encoder.
- Thanks to [Hugging Face Transformers](https://github.com/huggingface/transformers) for the Inkling 2 reference implementation (`Inkling2VisionModel`, `Inkling2ImageProcessor`), used as the independent oracle.
- Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (EXL3 format), [PonyExl3](https://github.com/beamivalice/PonyExl3) (Metal EXL3 port), [minijinja](https://github.com/mitsuhiko/minijinja) (templates), [image](https://github.com/image-rs/image) (picture decoding).
