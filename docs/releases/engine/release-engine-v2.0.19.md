# MLXL3 Engine 2.0.19

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release speeds up **Qwen3.8-27B** DFlash2 with sampling on Apple M5, and fixes DFlash2 Tune:

- **Sampled DFlash2 blocks are verified as trees too.** In 2.0.18 only greedy decoding used the tree. With sampling, DFlash2 is now 12–13% faster than the sampled chain, and the text still follows the model's sampling law exactly.
- **A small, bounded relaxation of sampled verification**, inspired by RFlash, is on by default. It adds about 4% more accepted tokens per block, for a measured divergence of 0.0045 nat per token. Greedy decoding is untouched.
- **Tune no longer rejects a mode for a tie.** It accepts a divergence where the baseline itself hesitated between the two tokens. On our Mac, Tune now keeps the tree mode (Auto) instead of falling back to the chain.
- **Tune can measure with sampling settings.** The engine also tells Desktop that DFlash2 supports sampling.

Greedy outputs and speeds are unchanged from 2.0.18. Everything from 2.0.18 is included. Weights, quantization formats and the bridge protocol are unchanged; the new request and event fields are optional.

## New in this release

### DFlash2 tree verification under sampling (Qwen3.8-27B, Apple M5)

Engine 2.0.18 verified a tree of 11 draft candidates in one pass, but only for greedy requests. Sampled requests (temperature above 0, or a penalty) still verified one chain.

- **How it works.** The tree is the same as in greedy mode. At each node, the children are distinct candidates, tried in order:
  - a child is accepted with its probability under the remaining target mass;
  - a rejected child leaves the law, which is renormalized;
  - if every child is rejected, the token is drawn from what remains.

  The token produced follows the target sampling law exactly, including temperature, top-k, top-p and penalties, each node using the history of its own path. With one child per node this is exactly the 2.0.18 chain verification, with the same random draws.
- **Gain**, Desktop's Qwen3.8 "Thinking" preset (temperature 1.0, top-k 20, top-p 0.95), two ABBA series, 12 generations per arm and prompt:

  | Prompt | Sampled chain | Sampled tree | Change |
  |---|---|---|---|
  | French prose | 17.56 tok/s | 19.62 tok/s | +11.7% |
  | Rust code | 16.93 tok/s | 19.19 tok/s | +13.3% |

  Accepted tokens per block rose from 1.94 to 2.48 (prose) and 1.79 to 2.46 (code); a block costs 6–10% more. Sampled speeds vary a lot between generations, because each one writes different text: single generations ranged from 12.5 to 23.3 tok/s.
- **Settings.** On by default. `MLXL3_DFLASH_TREE_SAMPLED=0` keeps the chain for sampled requests; `MLXL3_DFLASH_TREE=0` disables trees in both modes. The tree keeps 11 nodes: 7 and 15 nodes were slower with sampling too.

### Relaxed sampled verification (near-lossless, on by default)

This is inspired by RFlash ([Splash PR #776](https://github.com/incoai/splash/pull/776)), adapted to our deterministic draft candidates and to trees.

- **How it works.** At each sampled tree node, the target law p is inclined toward the node's draft candidates: p̂ ∝ p · e^β on the candidates. β is chosen so that the divergence KL(p̂ ‖ p) equals a budget κ, 0.01 nat by default. Probability ratios among the candidates, and among the other tokens, are kept. The exact verification above then samples from p̂.
- **What it changes.** Sampled text follows p̂ instead of p, so this mode is **not lossless**. Greedy decoding is not affected: its laws are points and are never inclined.
- **Quality**, measured on 1,881 generated tokens with the Thinking preset:
  - mean divergence of **0.0045 nat per emitted token**, below the budget, because nodes whose candidates already hold most or none of the mass are barely inclined;
  - the expected perplexity of the sampled text under the exact sampling law is ×0.973. The text moves slightly toward more likely tokens.
- **Gain.** The exact expected acceptance was computed on 276 real sampled blocks, with and without inclination, on the same target laws: **+4.4% tokens per block** at κ 0.01. The inclination costs microseconds per node, and block time was unchanged in the speed ABBA (177.0 against 176.8 ms on prose). The speed ABBA itself could not resolve a 4% change: sampled runs drifted by up to 19%.

  | κ (nat per token) | Expected tokens per block | Mean KL per token |
  |---|---|---|
  | 0 (exact) | 3.454 | 0 |
  | 0.005 | +3.1% | 0.0021 |
  | **0.01 (default)** | **+4.4%** | **0.0043** |
  | 0.02 | +6.1% | 0.0085 |
  | 0.05 | +9.6% | 0.0206 |
- **Setting.** `MLXL3_DFLASH_RELAX_KL` sets κ from 0 to 1. `0` keeps the exact verification.

### Tune

- **Near ties.**
  - **Before.** In 2.0.18, DFlash2 Tune required every mode's output to be identical to the autoregressive baseline. The tree changed one token at an exact tie on one Tune prompt, so Tune marked Auto "Output differs from baseline" and kept "7 tokens", the chain without a tree.
  - **Now.** At the first divergence, Tune re-reads the baseline's log-probabilities outside the timed runs. If the gap between the baseline token and the mode's token is at most 0.10 nat, the divergence counts as a near tie and is reported in the row's `near_ties`.
  - **Still refused.** A wider gap, or an output that stops on one side only, still makes the mode ineligible.
  - **Same rule for the warm-up** and for MTP Tune.
  - **Result.** On the 27B, Tune now keeps Auto (23.60 tok/s, a near tie at a gap of 0.000) over 7 tokens (22.69 tok/s).
- **Sampling.** `tune_mtp` and `tune_dflash` read the request's sampling fields (temperature, top-k, top-p, repetition and presence penalties).
  - **Sampled measurement.** With a non-greedy sampler, the measured runs sample from a fixed seed per prompt, the same seed for every mode. The warm-up stays greedy and keeps the correctness check. Sampled outputs are not compared token by token.
  - **Reporting.** Rows and the final event carry `sampled`.
  - **Default.** Requests without these fields stay greedy, as before.
- **Capability.** `ready` now announces `dflash_sampling_supported` for Qwen targets with DFlash2.

## Experimental (off by default)

- `MLXL3_DFLASH_TREE_BATCHED_GDN=1`: runs the gated DeltaNet of every tree path in one batched call. It is bit-identical to the default: 26.7 million compared values, all equal. It saves only 0.6–2.8 ms per verification in a micro-benchmark.
- `MLXL3_GDN_PATH_LOOP=1` (with the previous option): replays every path from one state load. It is also bit-identical, with no gain.
- Diagnostics: `MLXL3_DFLASH_PROFILE` (block anatomy), `MLXL3_DFLASH_RELAX_LOG` (divergence and perplexity terms) and `MLXL3_DFLASH_RELAX_CURVE` (expected acceptance for several budgets).

## Performance

- **Hardware.** All measurements use an Apple M5 MacBook with 24 GB and a 10-core GPU, on AC power.
- **Setup.** The bridge ran with a 4096-token context and the prompt cache off. Prompts were a French explanation and a Rust function, 128 generated tokens, one fresh bridge per pass.
- **Not additive.** Tables from different series do not add up.

**Where the time goes in a greedy tree block** (66 blocks, 175 ms each):

| Phase | Median |
|---|---|
| Draft and lattice readback | 22.4 ms (13%) |
| Tree verification | 151.9 ms (87%) |
| Tree building and commit | 0.6 ms |

Verification rows are almost free (8 → 15 rows: +6 ms), while each extra tree path costs 1.2–2.9 ms of gated DeltaNet work. Bigger trees, tree shapes with fewer paths, and the two batched variants above did not beat the 11-node tree.

**Greedy, unchanged from 2.0.18.** With the released binary, DFlash2 mode 1 runs at 24.85 tok/s on prose and 18.97 tok/s on code, with token hashes identical to 2.0.18.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The tree verification and the TensorOps kernels need an Apple M5 with macOS 26.3 or later; other Macs keep their earlier paths. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- **Older Desktop versions** keep DFlash2 greedy, as they did before. They ignore the new `dflash_sampling_supported` flag and send greedy Tune requests. They still benefit from the Tune near-tie fix.
- **Desktop 2.0.9** lets DFlash2 and Tune follow the sampling settings.

## Verification and limits

- **Correctness, before any speed measurement:**
  - **Exact sampled tree, unit tests.** Fixed-seed Monte Carlo runs check that the produced token follows the target law within 4 standard deviations: 400,000 draws per case over 9 laws and candidate sets, and the joint law of the first two tokens over 300,000 blocks.
  - **One candidate per node** gives the chain's verdicts and random state bit for bit over 10,000 draws; point laws give the greedy path.
  - **Mutation check.** Computing the remaining mass without removing rejected candidates makes both law tests fail.
  - **Inclination.** The budget is met to 1e-9; ratios inside and outside the candidate set are kept to 1e-12; the produced token follows p̂ over 400,000 draws. The exact expected acceptance matches the mean of 300,000 sampled blocks.
  - **Tune rule.** Identical outputs, gaps of 0, 0.1 and −0.04 accepted; 0.1001, 0.5 and −0.2 refused; one-sided end of output refused; non-finite gaps rejected. Removing the absolute value makes the test fail.
  - **Quality of the target rows of a tree** (teacher forcing, 1,536 positions per text, F32 log-softmax), against the exact path:
    - French: perplexity 8.0294 exact, 8.0308 for the TensorOps chain, 8.0311 for the tree; 2 and 3 argmax changes;
    - code: 2.1295, 2.1292 and 2.1289; 1 and 2 argmax changes.
  - **Batched GDN option.** 26,711,040 values were compared against path-by-path verification, with 0 differences. Removing the per-path offset makes this check fail.
- **Bridge, released binary extracted from the archive, no environment variables** (French explanation and Rust function, 128 tokens, one model at a time):
  - **Qwen3.8-27B greedy.** One token hash per prompt across autoregressive decoding, MTP depth 1, MTP depth 3 and DFlash2 mode 1, identical to engine 2.0.18. DFlash2 runs at 25.48 / 19.88 tok/s (single pass).
  - **Qwen3.8-27B with the Thinking preset.** Every generation completed (DFlash2 17.89 / 18.51 tok/s, single sampled pass).
  - **DFlash2 Tune, greedy.** Baseline 9.76 tok/s, Auto 23.44 (a near tie at a gap of 0.000), 2 tokens 15.54, 7 tokens 22.68; Auto is kept.
  - **DFlash2 Tune, Thinking preset.** Four rows flagged sampled: baseline 9.87, Auto 20.21, 2 tokens 13.80, 7 tokens 19.72; Auto is kept.
  - **Capabilities.** `ready` reports `dflash_sampling_supported: true` for Qwen3.8-27B, and DFlash2 runs. For Qwen3.6-35B-A3B it reports `dflash_supported: false` and a DFlash2 request returns "Unsupported DFlash2 target".
  - **Qwen3.6-35B-A3B.** Greedy token hashes identical to engine 2.0.14 in autoregressive decoding and MTP depth 1.
  - **LFM3-1.7B.** Greedy token hashes identical to engine 2.0.14.
- **Released archive, extracted into a new directory:**
  - its four files match the manifest: release tag `engine-v2.0.19`, version 2.0.19, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files;
  - the three Mach-O files pass strict signature and arm64 checks, and `mlx.metallib` is identical to MLX 0.32.2's;
  - `runtime-info` reports version 2.0.19, MLX, chat and source `2d74600037a8`, with no uncommitted changes.
- **Local checks on the release source:**
  - Rust formatting and the strict Clippy builds (`-D warnings`) for `chat`, `mlx,chat` and `direct-metal,chat` passed.
  - Tests:

    | Suite | Passed | Failed | Ignored |
    |---|---|---|---|
    | CPU | 613 | 0 | 10 |
    | All MLX targets | 679 | 0 | 330 |
    | Direct-Metal library and binary | 572 | 0 | 141 |

  - MLX source inventory check passed (339 files). The ignored tests need a physical GPU or a model; the relevant GPU tests above ran separately.
- **Not verified:**
  - Kani was not run on the new code.
  - **The relaxed sampled verification is not lossless.** Its gain (+4.4%) is an exact expectation on recorded blocks; the speed ABBA could not resolve it.
  - Sampled speeds were measured with the Thinking preset only, not with Instruct (temperature 0.7, presence penalty 1.5).
  - Long contexts with trees were not measured.
  - Only an Apple M5 MacBook with 24 GB was used.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the requests (sampled trees, near-tie Tune, sampled Tune, the relaxation budget) and the release decision.
- Thanks to [incoai Splash](https://github.com/incoai/splash) for:
  - the DFlash2 draft and its selector;
  - RFlash ([PR #776](https://github.com/incoai/splash/pull/776)), whose relaxed verification inspired the inclination;
  - its tree gated DeltaNet design, reviewed for this release.
- Thanks also to:
  - [TensorFold](https://github.com/ashhart/TensorFold) for the best-first selector tree;
  - [uzu](https://github.com/trymirai/uzu) for its tree-verification and tree-budget work, reviewed for this release;
  - [DFlash](https://github.com/z-lab/dflash) for the block speculative decoding research behind DFlash2;
  - [Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B;
  - [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime and custom Metal kernels;
  - [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format;
  - [PonyExl3](https://github.com/beamivalice/PonyExl3) for the Metal EXL3 port.

## Source and assets

Engine source commit `2d74600037a860a2f08f4a8f3ec4470686e31ae8`. It is `main` at `530fa61b5` (engine 2.0.18) plus commit `3bcd614d3` (sampled trees, relaxation, Tune, Desktop) and `2d7460003` (versions 2.0.19 / Desktop 2.0.9).

- `MLXL3-Engine-v2.0.19-arm64.tar.gz`: 68,924,781 bytes, SHA-256 `07ec288a16d9aaa7e7ca85744d46419f8afc2e784397f3752b22136034c6161c`.
- `MLXL3-Engine-v2.0.19-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `1f4a6e15d459bd782b99ee0c3ef03c9a555f96a16686977ad2362828c9dcc431`.
- `mlxl3`: 14,690,656 bytes, SHA-256 `127c6691e727fd7e5379dca21dd42ec81f10dbf0bdbaa9d79ae6a9f16c2f783a`.
- Detailed trial records: `opti.md` (entries `Q27D-2026-11-18-38` to `48`, `PRODUCT-2026-11-18-TUNE-NEAR-TIE`, `PRODUCT-2026-11-18-DFLASH-SAMPLING`, `RESEARCH-2026-11-18-Q27-UPSTREAM-R5` and `RELEASE-ENGINE-2.0.19-DESKTOP-2.0.9`) and `docs/measurements/q27-decode-20261117/`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
