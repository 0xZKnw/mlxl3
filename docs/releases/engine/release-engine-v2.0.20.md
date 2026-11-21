# MLXL3 Engine 2.0.20

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release adds a new model family: **Inkling 2 Mini** from Thinking Machines, a 26B looped mixture-of-experts model, with its **DSpark drafter enabled by default**.

- **New architecture `inkling2_loop`.** The engine loads EXL3 quantizations of Inkling 2 Mini and runs them in `mlxl3 run` and in the Desktop bridge.
- **Recommended quantization:** [`0xzknw/Inkling-2-Mini-EXL3-3.33bpw`](https://huggingface.co/0xzknw/Inkling-2-Mini-EXL3-3.33bpw), 14.8 GB, experts at 3.33 bits per weight on average.
- **DSpark speculative decoding** uses the drafter shipped with the model. It keeps the sampling law exact and is 10–25% faster than plain decoding with sampling on an Apple M5.
- **Thinking and answer are streamed separately**, and turning thinking off sets the model's `reasoning_effort` to 0.

Every other model is unchanged: greedy token hashes are identical to engine 2.0.19 on the models checked below. Quantization formats and the bridge protocol are unchanged; the new statistics fields are optional.

## New in this release

### Inkling 2 Mini

Inkling 2 Mini has 24 layers, of which layers 6–17 are run three times, so one token costs 48 layer visits. Each layer routes to 8 of 128 experts with a sigmoid router. The engine implements:

- the looped schedule with separate caches and states for each visit, and the cross-pass message gates;
- attention with relative bias, sink logits, L2-clipped queries and keys, and depth attention over earlier visits;
- static and dynamic short convolutions, the dynamic ones in one Metal kernel;
- the soft-capped SwiGLU experts in the existing fused EXL3 expert kernel.

**What is quantized.** In the recommended repository only the routed experts are EXL3 codes: 4 bits in layers 0, 2, 4, 7, 8, 13, 16 and 17, and 3 bits in the other 16 layers. Attention, routers, norms, embeddings, the output head, the vision encoder and the drafter are the source tensors.

**Quality of the 3.33 bpw quantization**, measured on an H100 against the BF16 model on 24,564 positions (English, French, code and chat):

| | KL divergence | Top-1 agreement |
|---|---|---|
| 3.00 bpw (uniform) | 0.0381 nat | 93.84% |
| **3.33 bpw (recommended)** | **0.0336 nat** | **94.22%** |

**Output head in 8 bits by default.** At load time the engine quantizes the output head to 8 bits (MLX affine, groups of 64). This is **not lossless**: it adds a KL of 0.00001 nat and keeps 99.67% of top-1 choices against the 16-bit head, and saves about 3.6 ms per token. `MLXL3_INKLING_HEAD_BITS=16` restores the 16-bit head.

### DSpark speculative decoding

The model repository ships a small drafter (`mtp.safetensors`). The engine loads it when the file is present and uses it in the bridge:

- the drafter proposes up to 3 tokens, which the model verifies in one pass;
- **greedy requests** accept a proposal only when it is the model's own greedy token;
- **sampled requests** use rejection sampling with a point proposal: the token is accepted with its probability under the model's law, otherwise a token is drawn from the remaining mass. Each delivered token follows the ordinary sampling law, penalties included.

The drafter's layers run in 8 bits and its proposal head reuses the model's head. `MLXL3_INKLING_DSPARK=0` turns DSpark off.

`complete.stats` reports `dspark_proposed_tokens`, `dspark_accepted_tokens` and `dspark_blocks`.

### Chat format

- The model's control tokens (`<|message_model|>`, `<|content_thinking|>`, `<|content_text|>`, `<|end_message|>`) are never shown. Thinking is streamed in the `thinking` phase and the reply in the answer phase.
- With thinking disabled, the chat template receives `reasoning_effort = 0`.
- `hub download` fetches `mtp.safetensors` with the model, and `hub details` reports the architecture as supported.

## Performance

All measurements use an Apple M5 MacBook with 24 GB and a 10-core GPU, on AC power, with the 3.33 bpw quantization, an 8192-token context and one fresh bridge per pass. Tables from different series do not add up.

**Bridge, three prompts (French, code, English):**

| Mode | Plain decoding | DSpark |
|---|---|---|
| Greedy | 19.6–19.7 tok/s | 23.4–25.9 tok/s |
| Sampling, temperature 1.0 | 17.0–18.1 tok/s | 18.1–22.5 tok/s |

- With sampling, DSpark gained 10–25% over plain decoding outside the first request; plain decoding varied by ±0.5 tok/s.
- Accepted proposals: 242 of 402 and 115 of 228 in greedy mode; 119 of 303 to 264 of 507 with sampling.
- First request after loading: 3.4–3.6 s to the first token with DSpark, about 2 s more than plain decoding.

**Lossless work on the decode path** (bit-identical logits, checked by a fingerprint of the logits):

- fewer small kernels per token (merged short-convolution kernel, depth attention on stacked states, a leaner key/value layout): 0.6% faster on a short context and 5.6% faster on a long one;
- four threads per group in the expert kernels for Inkling's shapes.

**Where the time goes** in one decoded token (47.8 ms, plain decoding, before the 8-bit head): dense attention projections 17.1 ms, experts 11.6 ms, output head 3.5 ms, gates 2.3 ms, the rest of attention 2.2 ms, routers 1.9 ms, depth attention 1.4 ms, short convolutions 0.9 ms.

## Settings

- `MLXL3_INKLING_DSPARK=0`: plain decoding.
- `MLXL3_INKLING_HEAD_BITS=16`: 16-bit output head.
- `MLXL3_INKLING_DENSE_BITS=8`: also runs the dense attention projections in 8 bits. It is about 36% faster but **off by default**: it adds a KL of 0.0069 nat against the 16-bit engine.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- **Memory.** The 3.33 bpw model keeps about 15 GB resident, more with the drafter. It was run on a 24 GB Mac only; 16 GB Macs are not expected to hold it.
- **Desktop 2.0.11** is the first version that lists the model among the recommended models and selects its sampling preset. Older Desktop versions running this engine were not tested with it.

## Verification and limits

- **Correctness, before any speed measurement:**
  - **Against an independent oracle.** The engine's logits were compared with those of the quantized model and of the BF16 model computed on an H100 with Transformers, on fixed token rows; the test `matches_oracle_logits` passed for both quantizations.
  - **Drafter.** The proposals matched an independent implementation on 7 of 7 cases, again after the start-up fix below.
  - **Lossless changes.** Each one kept the logits fingerprint bit for bit, and the release branch reproduces the fingerprint of the validated local engine.
  - **Start-up fix.** The first DSpark request took 19–43 s to its first token because loading the drafter left a 1.8 GB copy in the MLX cache. The copy is now built in blocks and the cache is cleared; the first token arrives in 3.4–3.6 s.
- **Bridge, released binary extracted from the archive, no environment variables:**
  - **Inkling 2 Mini 3.33 bpw, greedy**, on the directory downloaded by `hub download` (three prompts: French, Python, English, then a second turn of the first conversation): every request ended on the model's end token, with thinking and answer separated and no control token in the text. DSpark decoded at 19.0 tok/s on the first request and 28.6 / 25.6 tok/s on the next two (single passes), accepting 64 of 120, 242 of 402 and 115 of 228 proposals, the same counts as the branch binary before the release commit.
  - **Same prompts with temperature 1.0** (the Desktop preset): all complete, DSpark 25.9 / 25.9 / 24.7 tok/s, 62 of 108, 244 of 441 and 138 of 273 proposals accepted. Sampled speeds vary from one generation to the next; these are single passes.
- **Other models, greedy, released binary against the published 2.0.19 binary** (two prompts, token hash, token counts, stop reason and text compared):
  - Qwen3.6-35B-A3B 2.49 bpw, Qwen3.8 Flash Next and d1-2B 8 bpw: **identical**.
  - This comparison ran with the binary built from the release branch before the release commit; the archive is built from the same sources.
- **Hub.** `hub download` of the recommended repository at its pinned revision into an empty directory fetched 15 files (14,790,034,171 bytes), all byte-identical to the published ones, `mtp.safetensors` included. The bridge check above ran on that directory.
- **Released archive, extracted into a new directory:**
  - its four files match the manifest: release tag `engine-v2.0.20`, version 2.0.20, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files;
  - the three Mach-O files pass strict signature and arm64 checks, and `mlx.metallib` is identical to MLX 0.32.2's;
  - `runtime-info` reports version 2.0.20, MLX, chat and source `8a67ba993c59`, with no uncommitted changes.
- **Local checks on the release source:**
  - Rust formatting and the strict Clippy builds (`-D warnings`) for `chat`, `mlx,chat` and `direct-metal,chat` passed.
  - Tests:

    | Suite | Passed | Failed | Ignored |
    |---|---|---|---|
    | CPU | 683 | 0 | 339 |
    | All MLX targets | 682 | 0 | 339 |
    | Direct-Metal library and binary | 701 | 0 | 469 |

  - MLX source inventory check passed (341 files). The ignored tests need a physical GPU or a model; the relevant GPU tests above ran separately.
- **Not verified, and limits:**
  - Kani was not run on the new code.
  - **No image input** for Inkling 2 Mini: the vision encoder is in the repository but the engine does not run it.
  - **No tool calls** (MCP) are offered to Inkling 2 Mini.
  - The prompt cache is not reused between requests for this model: in the second turn above, all 64 prompt tokens were evaluated again.
  - `mlxl3 run` decodes greedily without DSpark; DSpark runs in the bridge.
  - Greedy DSpark output can differ from plain greedy output where two tokens are nearly tied, because rows verified in a batch are not bit-identical to rows computed one at a time. On three prompts, one output was identical and two diverged after 1,016 and 233 characters.
  - The exactness of sampled DSpark relies on the acceptance rule already tested for MTP and DFlash2; no separate statistical test was run for DSpark.
  - The DSpark width is fixed at 3; the speeds come from three prompts and single passes on one Apple M5 MacBook with 24 GB.
  - A third-party 2.92 bpw quantization with per-expert and half-integer bit widths cannot be loaded by this engine.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the requests (Inkling 2 Mini kernels, the 3.33 bpw quantization, DSpark, the 8-bit head decision) and the release decision.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its DSpark drafter.
- Thanks to [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format and the quantizer (1.6.7) used for the recommended repository.
- Thanks also to:
  - [Hugging Face Transformers](https://github.com/huggingface/transformers) (PR #51050) and [mlx-lm](https://github.com/ml-explore/mlx-lm) (PR #2086), read as reference implementations of the architecture;
  - [exo](https://github.com/exo-explore/exo), whose handling of the Inkling family was reviewed;
  - [vcruz305](https://huggingface.co/vcruz305) for the SAGE 2.92 bpw quantization, studied for its bit allocation;
  - [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime and custom Metal kernels;
  - [PonyExl3](https://github.com/beamivalice/PonyExl3) for the Metal EXL3 port.

## Source and assets

Engine source commit `8a67ba993c59bee266fc1e78f891e64c955aa281`. It is `main` at `1d0a519fc` (engine 2.0.19, Desktop 2.0.10) plus the Inkling 2 Mini commit.

- `MLXL3-Engine-v2.0.20-arm64.tar.gz`: 69,046,054 bytes, SHA-256 `2d56c6f3e34ec3ad4d5eec48c04625181944dc2aed235e5b3e6bbcbec3c0d452`.
- `MLXL3-Engine-v2.0.20-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `a0ca4c36bedf27557962596d897dd6733eec51e04b69345bc19f15f7cbe75a77`.
- `mlxl3`: 14,973,456 bytes, SHA-256 `6d77721e393636f2f1a787e93d4743c100219e247a27b6eeef574c14da152a2a`.
- Model: `0xzknw/Inkling-2-Mini-EXL3-3.33bpw` at revision `d1248f8abc322dbaf82cd571f6e7eaca9f7d6dee`, built from `thinkingmachines/Inkling-2-Mini` at `ed40e7da625f594fc9c4990f5f66be24873c22d7`.
- Detailed trial records: `opti.md` (entries `QUANT-2026-11-21-INK2M-EXL3-3`, `QUANT-2026-11-21-INK2M-EXL3-R2`, `INTEGRATION-2026-11-21-INK2M-MLXL3`, `SPEED-2026-11-21-INK2M-P1` to `P5`, `DSPARK-INK2M-P0` to `P2` and `RELEASE-2026-11-21-ENGINE-2.0.20-DESKTOP-2.0.11`) and `docs/measurements/inkling2-*-20261121/`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
