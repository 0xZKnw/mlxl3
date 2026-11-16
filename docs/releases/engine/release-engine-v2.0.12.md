# MLXL3 Engine 2.0.12

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. **MTP speculative decoding is no longer limited to greedy decoding**: it now works with temperature, top-k, top-p, repetition and presence penalties, and keeps the exact output distribution of ordinary sampling. Greedy requests take the same path as in 2.0.11, bit for bit. Weights, quantization, kernels and vocabulary are unchanged.

## New in this release: MTP with sampling

- **Exact speculative sampling.** The MTP head still proposes its drafts deterministically, as before. The target model now verifies each draft against the full sampling policy of the request:
  - the draft token is accepted with the probability the target gives it;
  - otherwise the engine draws a replacement from the target distribution with that token removed and renormalized;
  - when every draft is accepted, the bonus token is drawn from the last verified row.
  
  This is standard rejection sampling with a deterministic draft. The tokens produced follow the same law as ordinary token-by-token sampling with the same settings, whatever the draft depth.
- **Penalties follow the accepted tokens.** The repetition penalty (last 20 tokens) and the presence penalty are recomputed for each verified position from the history including the earlier accepted drafts. A penalized but deterministic policy (temperature 0 with penalties) therefore gives exactly the same text with MTP as without it.
- **New request fields.** The bridge request accepts `top_p` (default `1`, range above 0 up to 1) and `presence_penalty` (default `0`, range −2 to 2). Older clients that do not send them get the previous behavior.
- **Capability reporting.**
  - The `ready` event reports `mtp_sampling_supported: true` on the MLX engine.
  - The `generation_mode` event reports `mtp_sampling`: `true` when MTP runs with a non-greedy policy.
  - Desktop 2.0.5 uses these fields to unlock the sampling controls while MTP is on.
- **Covered models:** Qwen3.6-35B-A3B and Qwen3.8-27B with their MTP heads, and Qwen3.8 Flash Next through the same verification code (see the limits below).

## Unchanged

- Greedy requests (temperature 0 or top-k 1, no penalties) keep the previous greedy verification path, including EVICT truncation.
- DFlash2, the direct-Metal engine and the MTP auto-tune stay greedy. The direct-Metal engine reports `mtp_sampling_supported: false`, so a client keeps MTP greedy there.
- The OpenAI-compatible server still rejects `top_p` other than 1.
- No kernel, weight format or memory layout changed.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

Desktop versions before 2.0.5 keep working with this engine. They do not send the new fields and still lock sampling to greedy while MTP is on.

## Verification and limits

- **Correctness of the sampling law (CPU tests):**
  - A golden test replays the previous bridge sampler (copied from the 2.0.11 source) on 7 policies × 24 tokens with fixed seeds: identical tokens and identical final random state.
  - An exhaustive oracle on a 400 × 400 uniform grid of random draws checks that verification gives exactly the target distribution, for 5 policies.
  - A chained test checks depths 1 to 3 against sequential target sampling, plus boundary and invalid-parameter cases.
  - Four deliberate bugs (mutations) were each caught by these tests: accepting drafts with twice their probability, a replacement draw that could return the rejected token, top-p ignored, and the presence penalty ignored for the draft prefix.
- **GPU parity (physical Apple GPU, one model at a time):** with a penalized deterministic policy, MTP at depths 1, 2 and 3 produced exactly the same raw text and token hash as plain decoding over 96 tokens, on:
  - Qwen3.6-35B-A3B EXL3 2.49 bpw with `Qwen3.6-35B-A3B-MTP-4bit`;
  - Qwen3.8-27B EXL3 with `Qwen3.8-27B-MTP-4bit`.
  
  With the Qwen "instruct" policy (temperature 0.7, top-k 20, top-p 0.8, presence 1.5), output was reproducible at a fixed seed and drafts were accepted. Over 96 tokens, at depths 1/2/3, Qwen3.6 accepted 44/50, 56/64 and 64/77 drafts, and Qwen3.8-27B accepted 36/58, 51/63 and 50/85. These acceptance counts are indicative only.
- **Released archive:** the archive below was extracted into a new directory.
  - Its files matched the manifest: release tag `engine-v2.0.12`, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files.
  - The three Mach-O files passed strict signature and arm64 checks.
  - `runtime-info` reports version 2.0.12, MLX, chat and source `3075413a5bc4`.
  - Through the Desktop NDJSON protocol, with Qwen3.6-35B-A3B and its MTP head at depth 3 (768 tokens each):
    - the `ready` event reported `mtp_sampling_supported: true`;
    - a greedy request ran MTP with `mtp_sampling: false` and accepted 475 of 573 drafts;
    - the Qwen "instruct" preset ran MTP with `mtp_sampling: true` and accepted 440 of 623 drafts;
    - the engine shut down cleanly.
  
  The first attempt of this check failed because the check script read only the visible answer while the model was still thinking. The engine was not at fault, and that attempt is kept in the evidence.
- **Local checks on the release source:**
  - Rust formatting passed.
  - Strict Clippy passed for `chat`, `mlx,chat` and `direct-metal,chat`.
  - CPU library: 454 tests passed, 10 ignored; binary 38 passed.
  - MLX library: 483 passed, 260 ignored; MLX binary: 51 passed, 9 ignored. The ignored tests are physical-GPU or fixture tests; the relevant GPU tests above ran separately.
  - Direct-Metal library: 467 passed, 140 ignored; binary: 39 passed, 1 ignored.
- **Not verified:**
  - No speed measurement: this is a feature release. Sampled MTP accepts fewer drafts than greedy MTP, so its speed-up over plain sampling is expected to be smaller than greedy MTP's, but it was not measured.
  - Qwen3.8 Flash Next with sampled MTP was not run on a GPU (no local Flash model). It is covered only by compilation, Clippy and the shared CPU verification code.
  - The sampling law is checked by sampled and grid tests, not by a formal proof. Kani was not run.
  - M1–M4, 8/16 GiB Macs and macOS 26.2 were not tested.
- **Hosted CI (added after publication)** on the release commit `a08acf560`:
  - Passed: Flash profile compatibility, EXL3 conversion contracts, native coverage inventory, `mlx-package`, `native (ubuntu-latest)`, the Kani allocator and speculative shards, and the Metal and loader contracts.
  - Failed or cancelled, exactly as on the previous `main` commit `9eabe8eda`:
    - `desktop-and-protocol` and `desktop-preferences`: the hosted Swift compiler times out type-checking `ChatComponents.swift:390`, a file this release does not touch;
    - the Kani `core`, `routes`, `flash` and main shards: an internal Kani compiler panic plus harness timeouts in files this release does not touch;
    - `native (macos-26)` and `guard-proofs`: cancelled, as on every recent `main` commit.
  - These failures were not introduced by this release, but they mean hosted CI is **not green**. The local checks above are the release verification.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the feature request and the release decision. Thanks to [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime and Metal tensor operations, [Qwen](https://huggingface.co/Qwen) for the Qwen3.6 and Qwen3.8 models and their MTP layers, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format, [PonyExl3](https://github.com/beamivalice/PonyExl3) for the Metal EXL3 port, and [MTPLX](https://github.com/youssofal/mtplx) and [TensorFold](https://github.com/ashhart/TensorFold) for the speculative decoding work this engine builds on.

## Source and assets

Engine source commit `3075413a5bc41a6e10a9015c23f329dc4ee953e1`. It is engine 2.0.11 (source `5a46dfe68`) plus a test-only lint fix (`120710bb5`), commit `36bf45037` (sampled MTP and recommended catalog settings) and commit `3075413a5` (version numbers). The other commits since 2.0.11 change only Desktop and documentation.

- `MLXL3-Engine-v2.0.12-arm64.tar.gz`: 68,130,516 bytes, SHA-256 `247fe7c3e0d5d5d5b5639a4de4116e261dece7573a8c87161257491fe2d6db14`.
- `MLXL3-Engine-v2.0.12-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `f2527aa6c991c64426e2762797f7745914c625dc3d41eb437445aa7dd152b55f`.
- `mlxl3`: 12,805,008 bytes, SHA-256 `3574c82df210bee55a6d8448a8b389de85de67ecb9fed79b52bf15b32e828adf`.
- Detailed trial records: `opti.md` (entry `MTP-SAMPLING-20261116`) and `docs/measurements/mtp-sampling-20261116/`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
