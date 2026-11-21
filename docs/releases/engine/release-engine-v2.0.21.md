# MLXL3 Engine 2.0.21

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release runs the new **Inkling 2 Mini R3** quantization, makes **DSpark adapt to the text** so it no longer slows long sampled answers down, and stops the engine's memory from growing during long generations.

- **New Inkling 2 Mini quantization (R3).** [`0xzknw/Inkling-2-Mini-EXL3-3.33bpw`](https://huggingface.co/0xzknw/Inkling-2-Mini-EXL3-3.33bpw) now holds a version with exactly the same size (3.33 bits per expert weight, 14.8 GB) and **17% lower KL divergence**. It gives 4 bits to the down projection of every expert and 3 bits to gate and up. Expert banks can now use a separate bit width for down projections; engine 2.0.20 cannot load this version.
- **Adaptive DSpark.** The drafter's block width now follows the acceptance measured during the answer. On long sampled answers DSpark was about 25% slower than plain decoding; it is now close to plain decoding there, and still faster on greedy and short answers.
- **Memory.** The MLX buffer cache is limited to 1 GiB while Inkling 2 is loaded. In a 1,200-token answer the engine's memory grew by 2.3–3.6 GB; it now grows by about 0.4 GB, or 1.5 GB with DSpark.
- **DSpark controls.** `ready` announces `dspark_supported`, a request can turn DSpark off with `"dspark": false`, and `complete.stats` reports DSpark timings and widths in `dspark_profile`.

Other models are unchanged; see Verification.

## New in this release

### Mixed-width expert banks and Inkling 2 Mini R3

EXL3 expert banks store gate and up in one trellis, so they share a bit width, but down is a separate trellis. The engine now reads one width for gate/up and one for down:

- a bank whose gate and up widths differ, or whose down widths differ between experts, is still refused, with an explicit error, before any tensor is read;
- single-width checkpoints run the same kernels with the same arguments as before.

**Why R3.** A study of the previous version (R2) against an exact local reference of the source model showed that, per byte, a bit spent on the down projection lowers the KL about 3 times more than on gate/up, and that R2 had put most of its 4-bit layers where they helped least. R3 spends the same bits differently: 4 bits for down in all 24 layers, 3 bits for gate and up, instead of 4 bits for all projections of 8 layers.

**Quality**, measured on an H100 against the BF16 model on the same 24,564 held-out positions for all versions:

| | R2 (previous) | **R3** |
|---|---|---|
| KL divergence | 0.0336 | **0.0280** (−17%) |
| Top-1 agreement | 94.22% | **94.66%** |
| English / French / code | 0.0117 / 0.0247 / 0.0133 | 0.0106 / 0.0202 / 0.0125 |
| Chat transcript | 0.0845 | 0.0689 |

An independent check in this engine, against an exact reference of the source model computed locally, gives KL 0.0137 for R3 against 0.0156 for R2 on 4,608 prose positions (−12%).

### Adaptive DSpark

DSpark verifies a block of the drafter's proposals in one pass. Each extra verified row costs about 0.2 of a decoding step on Inkling 2 Mini, because every row reads its own experts.

- **The problem.** On long answers sampled at temperature 1, only about 35% of first proposals were accepted. A block of 3 then delivered fewer tokens than its cost, and DSpark ran at about 13.8 tok/s against 18.4 for plain decoding.
- **The change.** Before each block the engine picks the width (0 to 3) that delivers the most tokens per unit of cost, from the per-position acceptance observed so far in the answer. Width 0 is a plain decoding step. Every 8 blocks it tries the full width again, so it notices when the text becomes easier.
- **Exactness.** The width does not change the acceptance rule, so sampled tokens still follow the model's sampling law exactly. A width-0 step gives the plain decoding step's logits bit for bit.
- **Reproducibility.** The choice depends only on accepted counts, never on timings, so a greedy answer is reproducible.
- **Setting.** `MLXL3_INKLING_DSPARK_ADAPTIVE=0` restores the fixed width of 3; `MLXL3_INKLING_DSPARK=0` turns DSpark off.

### Memory

Without a limit, MLX kept freed buffers in its cache while Inkling 2 generated, and the engine's footprint grew by 3.4–3.9 GB over a long answer. The engine now sets a 1 GiB cache limit when Inkling 2 is loaded. `MLXL3_ALLOCATOR_CACHE_MIB` still overrides it.

### Fixes

- **Speculative sampling with a drafted distribution was quadratic in the vocabulary.** When a token drawn from the draft's distribution was rejected, building the residual distribution looked up the draft probability of every token by scanning the whole distribution. With a 200,000-token vocabulary that took minutes per rejection. It is now linear. This affected only the opt-in sampled drafts (`MLXL3_MTP_SAMPLED_DRAFT=1` and the new `MLXL3_INKLING_DSPARK_SAMPLED=1`).

## Experimental (off by default)

- `MLXL3_INKLING_DSPARK_SAMPLED=1`: draws DSpark proposals from the drafter's distribution on the GPU (Gumbel-max) and verifies them with min(1, p/q), instead of proposing the drafter's most likely token. Measured on long sampled answers: acceptance 19.8% against 18.8%, same speed, so it stays off.

## Performance

Apple M5 MacBook, 24 GB, 10-core GPU, AC power; Inkling 2 Mini R3 in the bridge, 8192-token context. Sampled runs write different text each time; single runs or pairs. Tables from different series do not add up.

**Long answer sampled at temperature 1** (French essay, 1,200 tokens, then a short summary turn; alternating runs):

| Mode | Essay | Summary |
|---|---|---|
| Plain decoding | 17.95 / 18.43 / 18.36 tok/s | 17.49 / 18.10 / 18.11 |
| DSpark, fixed width (2.0.20 behaviour) | 13.77 / 13.88 | 17.34 / 14.12 |
| **DSpark, adaptive (2.0.21 default)** | **16.72 / 19.61 / 17.45** | **17.74 / 17.90 / 16.71** |

- On this text only about 35% of first proposals are accepted; the adaptive width mostly uses widths 0 and 1 there.
- Against the fixed width it is about 25% faster. Against plain decoding it ranges from equal to about 6% slower, inside the spread between runs.

**Short greedy answers** (French, Python, English prompts; identical text in every run):

| Mode | French | Python | English |
|---|---|---|---|
| DSpark, fixed width | 26.5 / 26.3 tok/s | 30.7 / 27.9 | 25.1 / 22.5 |
| DSpark, adaptive | 26.5 / 25.6 | 26.9 / 28.8 | 26.2 / 27.4 |

On average the adaptive width is level with the fixed one (26.9 against 26.5 tok/s). Plain greedy decoding of Inkling 2 Mini runs at about 19.6 tok/s.

**Memory during a 1,200-token answer** (process footprint at the start and end of the answer):

| | Before (2.0.20) | 2.0.21 |
|---|---|---|
| Plain decoding | 16.05 → 18.36 GB | 15.12 → 15.55 GB |
| DSpark | 14.73 → 18.37 GB | 14.72 → 16.24 GB |

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- **R3 needs engine 2.0.21.** Engine 2.0.20 refuses to load it ("Inkling layer 0: mixed expert bit widths"). The previous version stays downloadable at revision `d1248f8abc322dbaf82cd571f6e7eaca9f7d6dee` and still loads in 2.0.21.
- Desktop 2.0.11 keeps downloading the previous version, which it pins. Desktop 2.0.12 installs R3 and shows a DSpark switch.

## Verification and limits

- **Correctness, before any speed measurement:**
  - **Mixed-width banks.** A synthetic bank with 3-bit gate/up and 4-bit down matches an independent computation (each expert's projections through the ordinary EXL3 linear layer, activation in FP32) within 2·10⁻³ relative, on the one-row, few-row and 72-row (TensorOps) paths, with SiLU and Inkling's soft-capped SiLU. Reading down with the gate/up width is caught. Six admission cases (mismatched widths, invalid widths) are refused before any tensor is read.
  - **A real mixed-width Inkling checkpoint** (R2 with 3-bit gate/up from the 3.0 bpw conversion in 8 layers) gives the same KL through the fused path as through a per-expert reference path: 0.0177 against 0.0177, paired difference +0.0001 [−0.0005, +0.0007] on 4,608 positions.
  - **R3 against the CUDA oracle of the same weights:** on the 1,200-token row the median relative RMSE of centered logits is 2.1% and KL(oracle ‖ engine) is 0.0009 nats/token; argmax agreement 96–100%.
  - **Adaptive width.** A width-0 step gives the plain decoding step's logits bit for bit over 4 consecutive tokens, and the same drafter features as the verification path. CPU tests cover the width choice (high acceptance: wide blocks; the measured free-text acceptance: width 1; repeated rejections: plain steps), the probes, and the reproducibility of the choice.
  - **Sampled drafts (opt-in).** The GPU Gumbel draw follows softmax(logits / T): 40,000 draws at three temperatures, chi-square 5.6–12.2 against a bound of 25.7; a wrong noise is caught (chi-square 2,001). A rejection over a 200,000-token vocabulary now takes 0.15 s; the old lookup did not finish in 90 s.
- **No change for other models.** Logits fingerprints of the Inkling 2 decode path (short and long prompts, 8-bit and 16-bit head, and the tiny-model oracle and rollback checks) are identical to the 2.0.20 baseline. With the released binary against the published 2.0.20 binary, greedy bridge token hashes, token counts, stop reasons and text are identical for Qwen3.6-35B-A3B, Qwen3.8 Flash Next, d1-2B and Inkling 2 Mini R2 (with the fixed DSpark width, since the adaptive width can change where greedy ties break).
- **Bridge, released binary extracted from the archive, no environment variables, R3:** greedy and temperature-1 requests and a second turn all completed on the model's end token with thinking and answer separated; a request with `"dspark": false` runs without DSpark statistics; `ready` reports `dspark_supported: true`. Engine 2.0.20 refuses R3 with "Inkling layer 0: mixed expert bit widths".
- **Released archive, extracted into a new directory:** its four files match the manifest (`engine-v2.0.21`, version 2.0.21, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files); the three Mach-O files pass strict signature and arm64 checks; `mlx.metallib` is identical to MLX 0.32.2's; `runtime-info` reports 2.0.21 and source `c1f371b93ef1`, with no uncommitted changes.
- **Local checks on the release source:** Rust formatting and strict Clippy (`-D warnings`) for `chat`, `mlx,chat` and `direct-metal,chat`; the MLX source inventory (341 files); the complete Python suite (2,842 passed, 121 skipped).

  | Suite | Passed | Failed | Ignored |
  |---|---|---|---|
  | CPU | 616 | 0 | 10 |
  | All MLX targets | 686 | 0 | 343 |
  | Direct-Metal library and binary | 635 | 0 | 144 |

- **Not verified, and limits:**
  - Kani was not run on the new code.
  - Speeds come from one Apple M5 MacBook with 24 GB, single runs or pairs; sampled runs write different text each time.
  - On long sampled answers the adaptive width can still be a few percent slower than plain decoding.
  - The cost model of the adaptive width was measured on Inkling 2 Mini; it is not tuned for other Macs.
  - The prompt cache is still not reused between turns for Inkling 2.
  - The Gemma (GELU) expert path with mixed widths was not exercised: no such checkpoint exists.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the reports from the app (DSpark not visible, slow decoding, growing memory), the R3 quantization run and the release decision.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its DSpark drafter.
- Thanks to [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format and the quantizer (1.6.7) used for R3, and to [Modal](https://modal.com) for the H100 run.
- Thanks also to:
  - [vcruz305](https://huggingface.co/vcruz305), whose SAGE 2.92 bpw quantization (down above gate/up) was part of the study;
  - the [AtomicChat](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF), [ggml-org](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF), [mudler](https://huggingface.co/mudler/Inkling-2-Mini-APEX-GGUF) and [unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) quantizations, whose published recipes were compared;
  - [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime, random numbers and custom Metal kernels;
  - [PonyExl3](https://github.com/beamivalice/PonyExl3) for the Metal EXL3 port.

## Source and assets

Engine source commit `c1f371b93ef19432b71438d3fd374ca82929a092`. It is `main` at `a6babef2a` (engine 2.0.20, Desktop 2.0.11) plus commit `384808813` (mixed-width expert banks) and `c1f371b93` (adaptive DSpark, DSpark controls, memory cap, sampling fix, versions).

- `MLXL3-Engine-v2.0.21-arm64.tar.gz`: 69,053,963 bytes, SHA-256 `112aca8dd8fd765225bb9a43459b0cd4f8b6fe12acf7c489c98299a68f576fd0`.
- `MLXL3-Engine-v2.0.21-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `053111ec0336a557d85b16df0fd850a93d6cc1311e5c356fbf2b7779fd4a95a0`.
- `mlxl3`: 14,977,552 bytes, SHA-256 `3ed1d4e79d6abbf2700f7bf7b6b64eaf63d22e272ee1cc774b793550e0084386`.
- Model: `0xzknw/Inkling-2-Mini-EXL3-3.33bpw` at revision `0a4558409fc77d6491979505c0668e7469f7b3f7` (R3); the previous version is revision `d1248f8abc322dbaf82cd571f6e7eaca9f7d6dee`.
- Detailed trial records: `opti.md` (entries `DIAG-2026-11-21-INK2M-LOCALREF`, `QUANT-2026-11-21-INK2M-EXL3-R3-PREP`, `DIAG-2026-11-21-INK2M-APP` with DSPARK-INK2M-P3/P4/P4b/P4c and APP-F1/F2/F2b, and `RELEASE-2026-11-21-ENGINE-2.0.21-DESKTOP-2.0.12`).
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
