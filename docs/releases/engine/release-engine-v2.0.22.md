# MLXL3 Engine 2.0.22

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release loads the **slim Inkling 2 Mini checkpoint**: 1.2 GB less to download and store, with the same outputs.

- **Slim Inkling 2 Mini R3.** [`0xzknw/Inkling-2-Mini-EXL3-3.33bpw`](https://huggingface.co/0xzknw/Inkling-2-Mini-EXL3-3.33bpw) now stores two parts in the 8-bit form the engine already computed at load: the output head, and the DSpark drafter's projections. It also drops the vision encoder, which MLXL3 does not run. Weights go from **14.75 GB to 13.54 GB**.
- **Same outputs.** Against the previous files, decoding logits, the drafter's states and proposals, DSpark acceptance, KL against an exact reference and CLI text are **identical bit for bit**.
- **Engine 2.0.21 cannot load the new files.** The previous files stay downloadable at revision `0a4558409fc77d6491979505c0668e7469f7b3f7`, and 2.0.22 still loads them with identical outputs.

Other models are unchanged; see Verification.

## New in this release

### Stored 8-bit weights for Inkling 2

Until now the engine read the BF16 output head and drafter and made 8-bit copies of them at every load. Those copies are what generation uses by default. A checkpoint can now ship them directly:

| Part | Stored as | Loads like |
|---|---|---|
| Output head | MLX affine 8 bits, group 64, the 200,058 real vocabulary rows (`.weight` U32, `.scales` and `.biases` F16) | the 8-bit head copy, the default since 2.0.19 |
| DSpark drafter projections | MLX affine 8 bits, group 64 | `MLXL3_INKLING_DSPARK_BITS=8`, the default |
| DSpark Markov-head fold | FP16 `[200058, 256]` and its bias, precomputed from the BF16 head | the fold computed at load |

- The fold cannot be recomputed from an 8-bit head, so it ships with the drafter. Loading it also skips the 1.8 GB FP32 pass over the head that computed the fold at load. Load time was not measured.
- A request for another width on such a checkpoint is refused with an explicit error, rather than silently using the stored width:
  - `MLXL3_INKLING_HEAD_BITS=16`;
  - `MLXL3_INKLING_DSPARK_BITS=16`;
  - a draft-only head copy (`MLXL3_INKLING_DSPARK_HEAD_BITS` below 16).
- Checkpoints with dense weights (the previous R3 files, R2, R1) load exactly as before.

**Sizes** (bytes from the safetensors headers):

| | Previous R3 files | Slim R3 |
|---|---|---|
| Vision encoder | 0.637 GB | removed |
| Output head | 0.926 GB (BF16) | 0.490 GB |
| DSpark drafter (`mtp.safetensors`) | 0.636 GB | 0.493 GB |
| Experts, attention, embeddings, norms | same bytes | same bytes |
| **All weights** | **14.75 GB** | **13.54 GB** |

## Experimental (off by default)

- `MLXL3_INKLING_EMBED_BITS=8` rounds the embedding table to 8 bits at load. It is an evaluation option only, lossy, and its KL cost has not been measured yet.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- **The slim files need engine 2.0.22.** Engine 2.0.21 stops at load with "model.llm.unembed.weight: expected floating-point weights".
- **Desktop 2.0.12 keeps working.** It pins revision `0a4558409fc77d6491979505c0668e7469f7b3f7` of the model, which loads in 2.0.22 with identical outputs. A later Desktop release will move to the slim files.

## Verification and limits

- **Correctness of the slim files, before anything else** (Apple M5, MLX 0.32.2, default settings):
  - **Same logits.** Against the previous R3 files, the fingerprint of all logits over 128 greedy decoding steps is identical (`ec84d901be285027`), and so are the tokens. The published 2.0.21 engine gives the same fingerprint on the previous files.
  - **Same drafter.** A DSpark block's features, block states, base logits and 7 proposals are identical byte for byte. In DSpark generation at widths 2 and 3, acceptance, accepted positions and all 64 tokens are identical, and the tokens equal plain decoding.
  - **Same KL.** Against an exact local reference of the source model, the per-position KL on 1,024 positions of two held-out rows is identical.
  - **Exact copy.** 37,536 tensors were copied byte for byte and checked one by one by SHA-256 of their content; 422 vision tensors were removed and 28 replaced. The repack script has 8 tests, including two mutations it catches.
- **New loader:**
  - A stored affine copy reads back byte for byte what the load-time quantization produced, and gives the same matmul bits, at 4 and 8 bits.
  - Wrong dimensions, F32 scales, missing biases and a requested width other than the stored one are refused. Disabling the width check is caught by the test.
- **No change for other checkpoints.** The Inkling 2 fingerprint gate of 2.0.21 is identical with the 2.0.22 test binary:
  - R1 short and long prompts, with 8-bit and 16-bit heads: `4cc7ad95aff66139`, `20c59539832e972a`, `21751eb3a17fe320`, `73f8b3c2b5d2f92c`;
  - tiny-model oracle and rollback checks: `140afb3d9cef3e4a`, `af45501d5c09f625`.

  The rest of the code for other models is unchanged; only functions were added to the MLX bridge.
- **Released archive, extracted into a new directory:**
  - Its four files match the manifest (`engine-v2.0.22`, version 2.0.22, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files). The three Mach-O files pass strict signature and arm64 checks, and `mlx.metallib` is identical to MLX 0.32.2's.
  - `runtime-info` reports 2.0.22 and source `ca96f92f528a`.
  - Bridge, no environment variables, slim files, three prompts plus a second turn, all finished on the model's end token with thinking and answer separated:
    - greedy: DSpark accepted 63/100, 171/239 and 111/172 proposals;
    - temperature 1: DSpark accepted 70/108, 226/366 and 108/187 proposals;
    - `"dspark": false`: no DSpark statistics.

    `ready` reports `dspark_supported: true`. Greedy answers, thinking and the second-turn token hash are identical between the slim and the previous R3 files.
  - `scripts/smoke-release.py` on the slim files: two turns through the Desktop protocol, the second recalls a code given in the first. Engine 2.0.21 refuses the slim files with "model.llm.unembed.weight: expected floating-point weights".
- **Local checks on the release source:**
  - Rust formatting;
  - strict Clippy (`-D warnings`) for `chat`, `mlx,chat` and `direct-metal,chat`;
  - the MLX source inventory (341 files);
  - the complete Python suite (2,842 passed, 121 skipped).

  | Suite | Passed | Failed | Ignored |
  |---|---|---|---|
  | CPU | 600 | 0 | 10 |
  | All MLX targets | 672 | 0 | 344 |
  | Direct-Metal | 619 | 0 | 144 |

- **Not verified, and limits:**
  - Kani was not run on the new code.
  - The other models' greedy regression and the Desktop suite were not rerun: their code is unchanged.
  - No speed or load-time change is claimed. Timings were not measured for this release.
  - The KL cost of `MLXL3_INKLING_EMBED_BITS=8` was not measured: the run was interrupted.
  - Image input is still not supported for Inkling 2.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3 and for asking for a smaller Inkling 2 Mini download at the same quality.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its DSpark drafter.
- Thanks to [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format and the quantizer used for R3.
- Thanks to [Apple MLX](https://github.com/ml-explore/mlx) for the affine quantization, dequantization and quantized matmul used by the stored copies.
- Thanks also to the [unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) Inkling 2 Mini GGUF release, whose separate vision-projector and drafter files and size table were part of the comparison that led to this change.

## Source and assets

Engine source commit `ca96f92f528a50228339b73ad90563a98e14dfba`. It is `main` at `c9ebc069b` (engine 2.0.21, Desktop 2.0.12, updater fix) plus the stored-weight loader and the version change.

- `MLXL3-Engine-v2.0.22-arm64.tar.gz`: 69,058,643 bytes, SHA-256 `ba12fb08161d7f3ea0810c0601403da813a83d499bd827dc0322caa62fcd80c3`.
- `MLXL3-Engine-v2.0.22-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `ed6a2d45ef6d0bb5121961fa3ce69b631ace7cc8d8314ab7d12fdef690e81d6d`.
- `mlxl3`: 14,995,072 bytes, SHA-256 `78220a235af31f9392b10fa6e4d3f74f745c3331b42f503788d91c8cb4070168`.
- Model: `0xzknw/Inkling-2-Mini-EXL3-3.33bpw`, slim files at revision `a5763a01f6e34c84bdf1a5812c890ff3978d5d75`; the previous R3 files are revision `0a4558409fc77d6491979505c0668e7469f7b3f7`.
- Detailed trial record: `opti.md`, entry `SIZE-2026-11-21-INK2M-SLIM`, and `RELEASE-2026-11-22-ENGINE-2.0.22`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
