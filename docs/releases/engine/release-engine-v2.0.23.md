# MLXL3 Engine 2.0.23

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release loads an **even smaller Inkling 2 Mini checkpoint**, whose token embedding table is stored in 8 bits: 13.10 GB of weights, against 13.54 GB in 2.0.22 and 14.75 GB for the original R3 files.

- **8-bit embeddings.** [`0xzknw/Inkling-2-Mini-EXL3-3.33bpw`](https://huggingface.co/0xzknw/Inkling-2-Mini-EXL3-3.33bpw) now also stores its 0.93 GB embedding table as an MLX affine 8-bit copy. The engine dequantizes only the rows it looks up. This saves 434 MB.
- **Measured cost: within noise.** Against an exact reference of the source model, on 6,144 held-out positions, KL changes by +0.00019 nats on prose (95% interval −0.00023 to +0.00063) and +0.0006 overall (−0.0038 to +0.0050).
- **Engines 2.0.22 and older cannot load the new files.** The previous files stay downloadable, and 2.0.23 still loads them with unchanged outputs.

Other models are unchanged; see Verification.

## New in this release

### Stored 8-bit embedding table for Inkling 2

| Part | Stored as | Looked up as |
|---|---|---|
| Token embedding table, `[201024, 2304]` | MLX affine 8 bits, group 64 (`.weight` U32, `.scales` and `.biases` F16) | the requested rows of the three arrays, dequantized to FP16, then the usual FP32 norm |

- The lookup gives, element for element, the table rounded to 8 bits at load. `MLXL3_INKLING_EMBED_BITS=8` produces that table, and it is the setting whose quality was measured.
- A requested width other than the stored one (`MLXL3_INKLING_EMBED_BITS=16`) is refused with an explicit error.
- Checkpoints with a dense embedding table load exactly as before. `MLXL3_INKLING_EMBED_BITS=8` on such a checkpoint remains an evaluation option.

**Quality** (Apple M5, exact local reference of the source model with a BF16 head, 12 held-out rows of 2,048 tokens, every 4th position):

| Rows | Positions | KL, BF16 embeddings | KL, 8-bit embeddings | Difference [95% CI] |
|---|---:|---:|---:|---|
| Prose (English, French, code) | 4,608 | 0.01374 | 0.01393 | +0.00019 [−0.00023, +0.00063] |
| Agentic chat transcript | 1,536 | 0.0996 | 0.1014 | +0.0018 [−0.0154, +0.0187] |
| All | 6,144 | 0.0352 | 0.0358 | +0.0006 [−0.0038, +0.0050] |

- Top-1 agreement is 94.68% and 94.76% overall.
- The criterion fixed before the measurement (prose difference at most +0.0007 nats, upper bound at most +0.0014) was met.
- With the BF16 table, the run reproduced the earlier R3 measurement bit for bit.

**Sizes** (weights in the safetensors files):

| | Original R3 | Slim (2.0.22) | **Slim with 8-bit embeddings** |
|---|---|---|---|
| Embedding table | 0.926 GB | 0.926 GB | 0.492 GB |
| **All weights** | **14.75 GB** | **13.54 GB** | **13.10 GB** |

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- **The new files need engine 2.0.23.** Engines 2.0.22 and older stop at load with "expected floating-point weights".
- Previous revisions of the model:
  - `a5763a01f6e34c84bdf1a5812c890ff3978d5d75`: slim, BF16 embeddings, engine 2.0.22;
  - `0a4558409fc77d6491979505c0668e7469f7b3f7`: R3, engine 2.0.21.

  Both still load in 2.0.23.
- Desktop 2.0.13 and 2.0.12 pin their revisions and keep working. Desktop 2.0.14 moves to the new files.

## Verification and limits

- **Correctness of the new files, before anything else** (Apple M5, MLX 0.32.2, default settings). The new files give **the same results bit for bit** as the previous slim files with their embedding table rounded to 8 bits at load (`MLXL3_INKLING_EMBED_BITS=8`, the measured setting):
  - fingerprint of all logits over 128 greedy decoding steps (`d582053e10aac428`), the same as R3 with the same setting;
  - a DSpark block's drafter features, block states, base logits and proposals, byte for byte;
  - DSpark generation at widths 2 and 3, apart from timings;
  - KL on 1,024 positions of two held-out rows.

  The previous slim files keep their fingerprint (`ec84d901be285027`).
- **Exact copy.** Only `model.llm.embed.weight` was replaced. The 2,226 other tensors of its file were copied byte for byte and checked by SHA-256 of their content, and the other files are identical. The repack script has 5 tests, including one mutation it catches.
- **New loader.** Stored rows equal the rounded table bit for bit. Another width and a wrong vocabulary size are refused; disabling the width check is caught by the test.
- **No change for other checkpoints.** The Inkling 2 fingerprint gate of 2.0.21 and 2.0.22 is identical:
  - R1 short and long prompts, 8-bit and 16-bit heads: `4cc7ad95aff66139`, `20c59539832e972a`, `21751eb3a17fe320`, `73f8b3c2b5d2f92c`;
  - tiny-model oracle and rollback checks: `140afb3d9cef3e4a`, `af45501d5c09f625`.

  Only Inkling 2 code changed.
- **Released archive, extracted into a new directory:**
  - Its four files match the manifest (`engine-v2.0.23`, version 2.0.23, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files). The three Mach-O files pass strict signature and arm64 checks, and `mlx.metallib` is identical to MLX 0.32.2's.
  - `runtime-info` reports 2.0.23 and source `8c792b13921b`.
  - Bridge, new files, no environment variables: greedy and `"dspark": false` requests passed, and `scripts/smoke-release.py` passed. At temperature 1 the three answers passed, but the second-turn check of the test harness came back empty. That check allows 120 tokens, which the model's sampled reasoning used up.
    - The same happens with the previous slim files in this engine.
    - With a 600-token budget, the second turn answers correctly.
    - Sampling differs from run to run, and the 120-token check passed on another sampled run with the measured setting.
  - Engine 2.0.22 refuses the new files with "model.llm.embed.weight: expected floating-point weights".
- **Local checks on the release source:**
  - Rust formatting;
  - strict Clippy (`-D warnings`) for `chat`, `mlx,chat` and `direct-metal,chat`;
  - the MLX source inventory (341 files);
  - the complete Python suite (2,842 passed, 121 skipped).

  | Suite | Passed | Failed | Ignored |
  |---|---|---|---|
  | CPU | 600 | 0 | 10 |
  | All MLX targets | 673 | 0 | 345 |
  | Direct-Metal | 619 | 0 | 144 |

- **Not verified, and limits:**
  - Kani was not run on the new code.
  - The 8-bit embeddings change the outputs slightly. The cost was measured on 12 rows of English, French, code and one agentic transcript, and no task benchmark was run.
  - The other models' greedy regression and the Desktop suite were not rerun: their code is unchanged.
  - No speed or load-time change is claimed.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for asking to measure the 8-bit embeddings and then to ship them.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its DSpark drafter.
- Thanks to [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format and the quantizer used for R3.
- Thanks to [Apple MLX](https://github.com/ml-explore/mlx) for the affine quantization and dequantization used by the stored table.

## Source and assets

Engine source commit `8c792b13921b5d869d517de39ec6bd0a90fbcfbc`. It is `main` at `a3f5bd30b` (engine 2.0.22, Desktop 2.0.13), plus the journal entry of the embedding measurement (`26e14317c`) and the stored-embedding loader with the version change.

- `MLXL3-Engine-v2.0.23-arm64.tar.gz`: 69,058,312 bytes, SHA-256 `6d3ad2f5490fc6e752155a5da0e5222ade2ba36e5ddf617faef10ee2e6e02075`.
- `MLXL3-Engine-v2.0.23-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `a7cf8702d451b2b1e5fd60b61412b25fd650571522a5527dd4070333d4e15190`.
- `mlxl3`: 14,994,784 bytes, SHA-256 `fc99c0405c5cdd0ffecbd23d95186a06518b414e04d7bd27e85fa5a85d33453f`.
- Model: `0xzknw/Inkling-2-Mini-EXL3-3.33bpw`, new files at revision `dde31cd0aea4ca0f61eaa5649559ef01e5baf25a`.
- Detailed trial records: `opti.md`, entries `SIZE-2026-11-21-INK2M-SLIM` (part B, the measurement), `SIZE-2026-11-22-INK2M-EMBED8` and `RELEASE-2026-11-22-ENGINE-2.0.23`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
