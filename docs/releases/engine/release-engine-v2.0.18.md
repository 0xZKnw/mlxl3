# MLXL3 Engine 2.0.18

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release makes **Qwen3.8-27B** faster again on Apple M5, in two places:

- **DFlash2 verifies a tree of candidates instead of a single chain.** Greedy DFlash2 decoding is 21–25% faster on prose and 5% faster on code. The generated text matched the chain on every measured prompt.
- **Prefill decodes each weight tile once per 256-token chunk instead of four times.** Prompt processing is 46% faster on a 1,097-token document, and time to first token is halved on a short prompt. Outputs are bit-identical.

DFlash2 is also no longer offered for **Qwen3.6-35B-A3B**, where MTP is faster. Everything from engine 2.0.17 (d1 decision models) is included. Weights, quantization formats and the bridge protocol are unchanged.

## New in this release

### DFlash2 tree verification (Qwen3.8-27B, Apple M5)

In 2.0.15, each DFlash2 block on the 27B proposed one chain of 7 tokens, the path picked by the draft's selector. The selector already scores 16 candidates per position, plus the transition between every pair of candidates, so other likely continuations were computed but never verified.

- **How it works.**
  - From the selector's scores, the engine builds a best-first tree of 11 draft nodes. Each child is scored by `log_softmax((unary + 0.6 × edge) / 1.0)`, cumulated along its path, and the highest-scoring nodes are kept. This is TensorFold's scheme.
  - The target verifies the whole tree in one 12-row pass:
    - projections, norms and the MLP run per row, as in 2.0.15;
    - attention uses per-depth positions and an ancestor mask;
    - gated DeltaNet runs once per root-to-leaf path with the existing kernels.
  - The engine keeps the longest path that follows the target's greedy choices, compacts its attention K/V and replays its DeltaNet state.
- **Gain.** On our prompts, one block now yields 4.6 tokens on prose instead of 3.5, for a verification only slightly more expensive.
- **Quality.**
  - A path verified alone through the new code is bit-identical to the 2.0.15 chain verification.
  - Inside a full tree, logits differ from that path by a few F16 ULP; in our oracle, one argmax changed in 30 rows and none after a commit. The cause is attention summing over more (masked) keys, which reorders floating-point sums.
  - The generated text was identical to the chain on both benchmark prompts in every run. As with the TensorOps mode of 2.0.15, this is not guaranteed in general: a long greedy generation can diverge after a near-tie.
- **When it is used.**
  - The wide TensorOps mode of 2.0.15: Apple M5, Qwen3.8-27B, DFlash2 mode 1.
  - Greedy requests only; Desktop runs DFlash2 greedy.
  - Full blocks only: sampled requests, context copies and the last tokens of a reply keep the chain.
- **Settings.** On by default. `MLXL3_DFLASH_TREE=0` restores the chain, and `MLXL3_DFLASH_TREE_NODES` sets 7 to 15 draft nodes (default 11, the fastest in our tests).

### Faster prefill (EXL3 projections on Apple M5)

Prefill projections run on the M5 TensorOps unit in 256-token chunks. In 2.0.15 each threadgroup handled one 64-row block, so every 16 × 32 weight tile was decoded four times per chunk.

- **The fix.**
  - A threadgroup now covers four row blocks and decodes each tile once into threadgroup memory.
  - Each SIMD group then runs exactly the same TensorOps operations as before, in the same order, on its own block.
- **Quality.** Outputs are bit-identical to 2.0.15: 358 million values compared across all projection shapes from 24 to 512 rows, and text parity at the bridge.
- **Scope.** The kernel serves all EXL3 projections of 24 rows or more on M5: Qwen3.8-27B in full, and the dense parts of Qwen3.6-35B-A3B.
- **Setting.** `MLXL3_DENSE_PREFILL_SHARED=0` restores the 2.0.15 kernel. `2` and `8` choose other group sizes, which were not faster.

### DFlash2 removed from Qwen3.6-35B-A3B

On the 35B, MTP depth 1 is faster than DFlash2 (about 81 tok/s against 62–70 tok/s on our M5).

- The bridge no longer announces DFlash2 for this model: `dflash_supported` and DFlash2 Tune are false.
- A DFlash2 request is refused with "Unsupported DFlash2 target". Desktop then shows DFlash2 as unavailable for this model.
- MTP and autoregressive decoding are unchanged; greedy token hashes are identical to 2.0.14.
- `MLXL3_DFLASH_35B=1` re-enables DFlash2 on the 35B.

## Experimental (off by default)

These prefill variants are all bit-identical to the default kernel, and none was faster in our tests:

- `MLXL3_DENSE_PREFILL_WINDOW=1`: window decoding, with the weight tile read straight from threadgroup memory.
- `MLXL3_DENSE_PREFILL_STEP_TILES=2|4|8`: 2, 4 or 8 depth tiles per barrier.
- `MLXL3_DENSE_PREFILL_FUSED=1`: fused input and output transforms.
- `MLXL3_DENSE_PREFILL_DEPTH=32`: 32-deep operations.
- `MLXL3_DENSE_PREFILL_SHARED=16`: 16 row blocks per group.

## Performance

All measurements use an Apple M5 MacBook with 24 GB and a 10-core GPU, on AC power.

- **Setup.** The bridge ran with a 4096-token context and the prompt cache off. Decode used a French explanation and a Rust function, 128 generated tokens, one fresh bridge per pass and a 32-token warm-up.
- **Order.** Each comparison was run as an ABBA series, and the tree comparison also in reversed order. The two arms of each series used the same binary, with only one environment variable changed.
- **Not additive.** Tables from different series do not add up.

**DFlash2 mode 1, greedy, chain (2.0.15 behaviour) against tree (2.0.18 default):**

| Prompt | Chain | Tree (11 nodes) | Change |
|---|---|---|---|
| French prose, series 1 | 20.48 tok/s | 24.81 tok/s | +21.1% |
| French prose, series 2 (reversed order) | 20.83 tok/s | 26.01 tok/s | +24.9% |
| Rust code, series 1 | 19.20 tok/s | 20.13 tok/s | +4.9% |
| Rust code, series 2 (reversed order) | 19.27 tok/s | 20.31 tok/s | +5.4% |

- Drift within an arm was at most 9.5% in series 1 and 4.0% in series 2.
- A 128-token reply took 28 blocks instead of 37 on prose, and 36 instead of 40 on code.
- Tree size, in tok/s: 7 nodes 24.34 / 18.98, **11 nodes 25.57 / 19.79**, 15 nodes 23.22 / 18.88.

**Prefill, 2.0.15 kernel (`MLXL3_DENSE_PREFILL_SHARED=0`) against 2.0.18:**

| Model and prompt | 2.0.15 | 2.0.18 | Time to first token |
|---|---|---|---|
| Qwen3.8-27B, 1,097-token document | 127.4 tok/s | 185.8 tok/s (+46%) | 8.94 → 6.13 s |
| Qwen3.8-27B, 2,825-token document | 133.8 tok/s | 189.6 tok/s (+42%) | 21.42 → 15.12 s |
| Qwen3.8-27B, short chat prompt (≈70 tokens) | 66.9 tok/s | 128.7 tok/s (+92%) | 1.04 → 0.54 s |
| Qwen3.6-35B-A3B, 1,097-token document | 751.0 tok/s | 868.2 tok/s (+16%) | 1.46 → 1.26 s |

- The 2,825-token and 35B rows were measured with 2-block groups; the 2.0.18 default is 4. A 4-against-8 comparison measured 190.8 tok/s on the 2,825-token document.
- Decode speed is unchanged by the prefill kernel.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The tree verification and the TensorOps prefill kernel need an Apple M5 with macOS 26.3 or later; other Macs keep their 2.0.17 paths. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

No Desktop update is required. Desktop already follows the engine's `dflash_supported` flag, so it hides DFlash2 for Qwen3.6-35B-A3B with this engine.

## Verification and limits

- **GPU oracles on a physical Apple M5:**
  - **Prefill kernel:** 358,123,008 values bit-identical to the 2.0.15 kernel, covering 8 projections including the head, 24 to 512 rows, and groups of 2, 4 and 8. The experimental variants are also bit-identical.
  - **Tree verification, on 12-row trees at 299 and 640 tokens of context:**
    - every path verified alone through the new code equals the 2.0.15 chain verification bit for bit;
    - inside the tree, the maximum log-probability difference was 0.047, with 1 argmax change in 30 rows;
    - committing an accepted path (prefix, non-prefix, full path or root only) and running one more step gave a maximum difference of 0.047 and no argmax change against the same sequence verified as a chain.
- **Bridge, released binary extracted from the archive, no environment variables.** Prompts were a French explanation and a Rust function, 128 tokens, one model at a time.
  - **Qwen3.8-27B greedy:** one token hash per prompt across autoregressive decoding, MTP depth 1, MTP depth 3 and DFlash2 mode 1, identical to engine 2.0.15. DFlash2 ran as a tree (28 and 36 blocks).
  - **Qwen3.8-27B with the Desktop preset** (sampled, so DFlash2 ran as a chain): every generation completed.
  - **Prefill against the published 2.0.15 binary,** same Qwen3.8-27B model (ABBA, text parity required):
    - parity held;
    - 1,097-token document: 125.2 → 177.5 tok/s (+41.7%), time to first token 9.10 → 6.42 s;
    - short prompt: 65.3 → 126.2 tok/s (+93.4%).
  - **Qwen3.6-35B-A3B:**
    - greedy token hashes identical to engine 2.0.14 in autoregressive decoding and MTP depth 1;
    - `ready` reports `dflash_supported: false`, and a DFlash2 request returns "Unsupported DFlash2 target";
    - with `MLXL3_DFLASH_35B=1`, DFlash2 is announced and runs.
  - **LFM3-1.7B:** greedy token hashes identical to engine 2.0.14.
- **Released archive, extracted into a new directory:**
  - its four files matched the manifest: release tag `engine-v2.0.18`, version 2.0.18, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files;
  - the three Mach-O files passed strict signature and arm64 checks;
  - `mlx.metallib` is identical to MLX 0.32.2;
  - `runtime-info` reports version 2.0.18, MLX, chat and source `1eff4caffa43`.
- **Local checks on the release source:**
  - Rust formatting passed.
  - Strict Clippy (`-D warnings`) passed for `chat`, `mlx,chat` and `direct-metal,chat`.
  - CPU tests: 607 passed, 10 ignored.
  - All MLX test targets: 671 passed, 327 ignored.
  - Direct-Metal library and binary: 566 passed, 141 ignored.
  - MLX source inventory check passed (339 files).
  - The ignored tests are physical-GPU or fixture tests; the relevant GPU tests above ran separately.
- **Not verified:**
  - Kani was not run on the new code.
  - **Tree verification is not bit-exact against the chain.** A long greedy generation can differ after a near-tie.
  - Long contexts were not measured with trees: the attention mask grows with the context.
  - Only an Apple M5 MacBook with 24 GB was used; M1–M4 Macs were not tested (they keep their 2.0.17 paths).
  - Sampled DFlash2 does not use trees.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the request and the release decision. Thanks to [TensorFold](https://github.com/ashhart/TensorFold) for the best-first selector tree and its scoring weights, [Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B and Qwen3.6, [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime, custom Metal kernels and the TensorOps API, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format, [PonyExl3](https://github.com/beamivalice/PonyExl3) for the Metal EXL3 port, [DFlash](https://github.com/z-lab/dflash) and [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) for the block speculative decoding research behind DFlash2, and [IncoAI Splash](https://github.com/incoai/splash) for the DFlash2 draft and its selector.

## Source and assets

Engine source commit `1eff4caffa435575031c723b641d64ce15a1c8d8`. It is `main` at `935a2964e` (engine 2.0.17) plus commit `d8519570c` (tree verification, prefill kernel, 35B DFlash2 removal), merge `f22dd942b` and `1eff4caff` (version 2.0.18).

- `MLXL3-Engine-v2.0.18-arm64.tar.gz`: 68,903,141 bytes, SHA-256 `ed577f3f1db4a7838d33640e66c74f1a41c5a78bac9206cfbddd5b2f6c587a88`.
- `MLXL3-Engine-v2.0.18-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `08ccf3056ca4e178a2a9c0a828dec8af905c0951e1a83396b84197f66b166a94`.
- `mlxl3`: 14,633,024 bytes, SHA-256 `449ae1ca71748504e12cf4cb59164de9507ffee447006483b0fd7c5f524478a3`.
- Detailed trial records: `opti.md` (entries `Q27P-2026-11-18-01` to `12`, `Q27D-2026-11-18-34` to `37`, `PRODUCT-2026-11-18-35B-NO-DFLASH2` and `RELEASE-ENGINE-2.0.18`), `docs/measurements/q27-prefill-20261118/` and `docs/measurements/q27-decode-20261117/`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
