# MLXL3 Engine 2.0.15

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release makes speculative decoding fast on **Qwen3.8-27B** (dense, EXL3). On an Apple M5, DFlash2 now decodes at about **20 tok/s in greedy mode**, against 9.9 tok/s for plain autoregressive decoding. Sampled DFlash2, which 2.0.14 refused, now works with the same exact sampling law as autoregressive decoding.

- On every Mac, the exact verification path of the 27B decodes each weight once per verification pass, keeps attention K/V in place and speeds up the 27B DFlash2 draft. These changes are bit-identical.
- On Apple M5 only, verification passes of three or more rows on the 27B now run on the GPU's TensorOps unit. **This path is not bit-exact**: the measured quality difference is a mean KL of 1.7e-4 and a perplexity change of +0.04% (details below). Set `MLXL3_DENSE_TENSOR_ROWS=0` to keep the exact kernels.
- Sampled DFlash2 is available for every Qwen DFlash2 model (Qwen3.8-27B and Qwen3.6-35B-A3B).

Autoregressive decoding and MTP depth 1 stay exact on every model. Greedy outputs of the other models are unchanged. Weights, quantization formats and the bridge protocol are unchanged.

## New in this release

### Exact verification for Qwen3.8-27B (all Macs)

In 2.0.14 a DFlash2 or MTP verification pass of three or more rows cost three to seven autoregressive steps on the 27B, so MTP depth 2–3 and every DFlash2 mode were slower than plain decoding.

- **Shared-decode verification.** Verification blocks of 3 to 8 rows now decode each EXL3 weight once and apply it to every row, with each row keeping the accumulation order of the one-row kernel. Outputs are bit-identical to 2.0.14. Set `MLXL3_DENSE_SHARED_VERIFY=0` to disable it.
- **Narrower adaptive DFlash2 blocks.** On the 27B exact path, DFlash2 mode 1 proposes at most three tokens per block, which made it faster than mode 2.
- **Attention K/V in place.** The dense 27B writes new attention keys and values into a preallocated buffer instead of concatenating. Outputs are bit-identical; an autoregressive step at a 16k-token context is 11% faster. Set `MLXL3_QWEN_KV_INPLACE=0` to disable it.

### Sampled DFlash2

- 2.0.14 rejected DFlash2 for sampled requests. DFlash2 now accepts each deterministic proposal with the target model's probability and draws the correction token from the residual distribution, the same exact rule MTP has used since 2.0.12. Each token is drawn from the same distribution as autoregressive decoding.
- It applies to Qwen3.8-27B and Qwen3.6-35B-A3B with their DFlash2 drafts, and covers Desktop's default preset (temperature 0.2, top-k 80, repetition penalty 1.05). Flash Next still requires greedy decoding for DFlash2.

### TensorOps verification on Apple M5 (Qwen3.8-27B, not bit-exact)

- **What it does.** Verification passes of three or more rows run the 27B EXL3 projections on the M5 TensorOps unit (Metal Performance Primitives `matmul2d`). The new kernel puts the weights in the left operand, decodes each trellis window once and keeps FP32 partial sums. The whole block then goes through each layer at once: norms, residuals and gated DeltaNet inputs are batched across the rows.
- **Why it is faster.** On the M5 the unit's cost is almost the same for 1 and 16 rows. A pass of 8 rows costs about 1.5 autoregressive steps, against about 3.5 on the exact path.
- **Wider DFlash2 blocks.** Because the verification cost hardly depends on the number of rows, DFlash2 mode 1 proposes seven tokens per block in this mode, and context copies can propose up to seven tokens.
- **Quality.** We compared teacher-forced logits on 3,072 positions of French prose and Rust code against the exact path:
  - mean KL divergence: 1.67e-4 (maximum 0.016);
  - argmax changes: 3 of 3,072 positions;
  - perplexity: 8.0275 → 8.0305 (+0.04%) on French, 2.12844 → 2.12832 on code.
  Greedy token hashes of the two benchmark prompts matched the exact path in every bridge run. This is not guaranteed in general: a long greedy generation can diverge from the exact path after a near-tie.
- **When it is used.** Apple M5 with macOS 26.3 or later, on the Qwen3.8-27B dense projections, for verification blocks of 3 rows or more. Autoregressive decoding and MTP depth 1 (one or two rows) keep the exact kernels, as do other Macs and other models.
- **Settings.** On by default in the engine binary.
  - `MLXL3_DENSE_TENSOR_ROWS=0` restores the exact kernels; an integer from 2 to 16 sets the row threshold (default 3).
  - `MLXL3_DENSE_TENSOR_BLOCK=0` verifies row by row instead of as one block. It produced the same values in our test and is slightly slower.
- **Long context.** At a 16k-token context, a three-row verification pass takes 177 ms instead of 214 ms on the exact path.

### Faster 27B DFlash2 draft

- The Q4 projections of the 27B draft now give each 128-column tile its own threadgroup. Their threadgroup counts had been tuned for the 2.5× narrower 35B draft.
- The draft network takes 14.5 ms per block instead of 18.9 ms, with bit-identical draft outputs on all 272 measured blocks.
- Set `MLXL3_DFLASH_Q4_GROUPS=tuned` to return to the previous counts.

### Context copies on by default for the 27B

- When the context already contains the continuation of the anchor token, a DFlash2 block proposes the copied tokens and skips the draft network.
- On by default for Qwen3.8-27B (`MLXL3_DFLASH_CONTEXT_COPY=0` disables it). Other models keep it off (`=1` enables it).
- On a copy prompt, decoding was 9.2% faster, with identical acceptance counts. Code editing (+0.7%) and French prose (+0.5%) did not change measurably.

### Fused TensorOps prologue and epilogue

- In TensorOps mode, the input preparation of each projection now runs as one Metal dispatch instead of about five MLX operations: F16 conversion, input signs, Hadamard-128, lane order and padding. The output finish also runs as one dispatch: split sum, F16, Hadamard-128 and output signs.
- Outputs are bit-identical to the MLX operations (20.4 million values compared, 0 differences).
- A synchronized verification pass is 2.3% faster. No decode-speed change was measurable at the bridge (−1.2% to −0.4% in an ABBA on AC power, within drift).

## Experimental (off by default)

- `MLXL3_QMV_FUSED_GLUE=1` applies the same fused prologue and epilogue to the exact QMV paths (autoregressive decoding, MTP, exact verification) of every model.
  - Outputs are bit-identical (15.1 million values compared on the 27B, 0 differences).
  - Bridge speed did not change measurably, so it stays off.
- TensorOps kernel tuning knobs, none faster than the defaults in our tests:
  - `MLXL3_TENSOR_LANE_SHAPE` = `32x16`, `64x16`, `32x64` or `64x32`;
  - `MLXL3_TENSOR_LANE_TOKENS=8`;
  - `MLXL3_TENSOR_LANE_BUFFERS=2` (bit-identical but 14–29% slower);
  - `MLXL3_TENSOR_LANE_SPLITS=target,limit`.
- `MLXL3_DFLASH_EDGE_SCALE` sets the DFlash2 selector's transition weight (default 0.25, range 0 to 4). Other values were not faster in our tests.

## Performance

All measurements use an Apple M5 MacBook with 24 GB and a 10-core GPU.

- **Setup.** The bridge ran with a 4096-token context and the prompt cache off, generating 128 tokens on two prompts: a French explanation and a Rust function. Each pass used a fresh bridge with a 32-token warm-up.
- **Order.** Arms alternated A, B, B, A, and both arms used the same binary with only environment variables changed.
- **Not additive.** Tables from different series do not add up.

**Exact path versus the 2.0.15 TensorOps default, greedy** (AC power; the wider draft is active in both arms):

| Mode | Exact path, fr / code | 2.0.15 default, fr / code |
|---|---|---|
| Autoregressive | 9.95 / 9.94 tok/s | 9.94 / 9.90 tok/s |
| MTP depth 1 | 14.56 / 12.52 tok/s | 14.49 / 12.55 tok/s |
| MTP depth 3 | 14.16 / 12.10 tok/s | 16.36 / 13.16 tok/s |
| DFlash2 mode 1 | 14.01 / 13.26 tok/s | **20.29 / 18.14 tok/s** |

- Greedy hashes matched across both arms and all modes. Drift within an arm was at most 2.5%.
- A second ABBA of the final default against `MLXL3_DENSE_TENSOR_ROWS=0` (AC power) gave DFlash2 mode 1 13.32 → 19.75 tok/s (+48%) on French and 12.66 → 18.53 tok/s (+46%) on code. Autoregressive decoding and MTP depth 1 were unchanged within drift.

**Exact-path changes alone, greedy** (on battery; 2.0.14 paths versus shared verification, K/V in place and the capped mode 1, before the TensorOps mode and the wider draft):

| Mode | 2.0.14 paths, fr / code | Exact paths, fr / code |
|---|---|---|
| Autoregressive | 9.71 / 9.71 tok/s | 9.68 / 9.59 tok/s |
| MTP depth 1 | 14.14 / 12.01 tok/s | 14.07 / 12.21 tok/s |
| MTP depth 3 | 9.90 / 9.41 tok/s | 13.88 / 11.84 tok/s |
| DFlash2 mode 1 | 6.73 / 6.52 tok/s | 13.43 / 12.69 tok/s |

**Desktop default preset** (temperature 0.2, top-k 80, repetition 1.05; AC power). These runs use a clock seed, so their texts differ between passes, and drift reached 10%:

- DFlash2 mode 1: 13.79 → 17.87 tok/s on French and 12.97 → 20.44 tok/s on code, exact path versus TensorOps default.
- Autoregressive decoding (9.8 tok/s) and MTP depth 1 (12.5 to 13.6 tok/s) were unchanged within drift.
- 2.0.14 could not run sampled DFlash2.

**Compared with 2.0.14 on this Mac** (a separate single-pass run on 17 November, AC power, indicative only): 2.0.14's fastest mode on the 27B was MTP depth 1 at 13.25 / 11.20 tok/s, and its DFlash2 mode 1 ran at 6.46 / 5.83 tok/s. In 2.0.15, DFlash2 mode 1 reaches 20.3 / 18.1 tok/s.

Prefill and the other models' speed are unchanged.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The TensorOps mode needs an Apple M5 and macOS 26.3 or later; other Macs use the exact path. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

No Desktop update is required. Desktop 2.0.6 and earlier versions work with this engine unchanged.

## Verification and limits

- **GPU oracles on a physical Apple M5**, each run alone:
  - Shared-decode verification on the 27B, for 1 to 8 rows: all 248,320 logits per row, the DFlash captures and the 128 state arrays bit-identical to row-by-row verification and to token-by-token autoregressive replay, including 28 partial commits.
  - Attention K/V in place: logits and the 128 state arrays bit-identical to concatenation in 45 comparisons at 256, 300 and 4,096 tokens of context.
  - Wider draft: draft outputs bit-identical on 272 blocks.
  - Fused TensorOps prologue and epilogue: 20,411,392 values bit-identical to the MLX operations (9 projections, 3 to 16 rows, F16 and BF16 inputs).
  - Opt-in exact-path fusion: 15,096,832 values bit-identical (8 projections, 1 to 8 rows).
  - TensorOps quality: the teacher-forced KL, argmax and perplexity comparison above, on the block-verification version of the kernel. Later kernel edits (one layout formula checked at run time, fused prologue and epilogue) keep the same operations, and greedy hashes did not change. The quality test itself was not rerun on the final binary.
- **Sampled DFlash2 law** (`bridge_dflash_sampling_matches_autoregressive_policy`), on the 27B and, on the release source, on Qwen3.6-35B-A3B:
  - greedy DFlash2 modes 1 to 3 give the same IDs as autoregressive decoding;
  - a deterministic penalized policy (temperature 0, repetition 1.1, presence 0.5) reproduces autoregressive decoding token by token;
  - the Desktop and Qwen presets complete reproducibly for a given seed. On the 35B, 60 to 74 drafted tokens were accepted per 96-token run.
- **Bridge, released binary extracted from the archive, no environment variables** (fr and code prompts, 128 tokens, one model at a time):
  - Qwen3.8-27B greedy: one token hash per prompt across autoregressive decoding, MTP depth 1, MTP depth 3 and DFlash2 mode 1, identical to every earlier run of this campaign.
  - Qwen3.8-27B with the Desktop preset (autoregressive decoding and DFlash2): both completed.
  - Qwen3.6-35B-A3B (autoregressive decoding, MTP depth 1 and DFlash2 mode 1) and LFM3-1.7B (autoregressive decoding): greedy token hashes **identical to engine 2.0.14**.
  - An ABBA of 2.0.14 against 2.0.15 on 35B DFlash2 showed no regression: −2.0% on French and +1.9% on code, within drift.
- **Released archive, extracted into a new directory:**
  - Its four files matched the manifest: release tag `engine-v2.0.15`, version 2.0.15, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files.
  - The three Mach-O files passed strict signature and arm64 checks.
  - `mlx.metallib` is identical to MLX 0.32.2.
  - `runtime-info` reports version 2.0.15, MLX, chat and source `fbbf217d28e3`.
- **Local checks on the release source:**
  - Rust formatting passed.
  - Strict Clippy (`-D warnings`) passed for `chat`, `mlx,chat` and `direct-metal,chat`. This also fixes the CPU-only Clippy failure of the previous push of [#179](https://github.com/0xZKnw/mlxl3/pull/179).
  - CPU tests: 582 passed, 10 ignored.
  - All MLX test targets: 643 passed, 312 ignored.
  - Direct-Metal library and binary: 541 passed, 141 ignored.
  - MLX source inventory check: passed (333 files).
  - The ignored tests are physical-GPU or fixture tests; the relevant GPU tests above ran separately.
- **Not verified:**
  - Kani was not run on the new code, so bounds and arithmetic of the new paths have no machine-checked proof.
  - **The TensorOps mode is not bit-exact.** A long greedy generation can differ from the exact path after a near-tie.
  - Only an Apple M5 MacBook with 24 GB was used. The exact path was run on that M5 with TensorOps disabled; M1–M4 Macs, other memory sizes and macOS 26.2 were not tested.
  - Contexts longer than 16k tokens were not measured.
  - The speed of sampled DFlash2 on Qwen3.6-35B-A3B was not measured; only its sampling law was tested.
  - DFlash2 Tune still evaluates greedy decoding only.

### Known issues

None new in this release.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the request and the release decision. Thanks to [Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B and Qwen3.6, [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime, custom Metal kernels and the TensorOps API, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format, [PonyExl3](https://github.com/beamivalice/PonyExl3) for the Metal EXL3 port, [DFlash](https://github.com/z-lab/dflash) and [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) for the block speculative decoding research behind DFlash2, [IncoAI Splash](https://github.com/incoai/splash) for the DFlash2 draft and its TensorOps Q4 kernel, [MTPLX](https://github.com/youssofal/mtplx) for the MTP speculative decoding approach, and [TensorFold](https://github.com/ashhart/TensorFold) for the compact draft vocabulary.

## Source and assets

Engine source commit `fbbf217d28e3874b0cc2165866ca592910e802e5`. It is `main` at `1f6a9ee6b` plus commit `8f2b47a88` (exact verification boosts, [#179](https://github.com/0xZKnw/mlxl3/pull/179)), `9244c80e3` (TensorOps verification and the other changes above) and `fbbf217d2` (version 2.0.15).

- `MLXL3-Engine-v2.0.15-arm64.tar.gz`: 68,241,980 bytes, SHA-256 `3482a3f848295eaf792f823eb6cea7f2a9045559bb0e73a14ba948b7965a38c0`.
- `MLXL3-Engine-v2.0.15-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `de69ff3cab48272e4a6749e1e6499f8cceb3b9bbaef87cdd06a505af9a202ec8`.
- `mlxl3`: 13,059,504 bytes, SHA-256 `cb06816b959076c979ad337b9434c7deadfacbedf1796c83e5c53e58fea4a2e6`.
- Detailed trial records: `opti.md` (entries `Q27D-2026-11-17-00` to `Q27D-2026-11-18-33` and `RELEASE-ENGINE-2.0.15`) and `docs/measurements/q27-decode-20261117/`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
