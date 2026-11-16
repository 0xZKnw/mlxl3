# MLXL3 Engine 2.0.13

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. **Greedy decoding of LFM3 is about 12% faster**, and the output does not change: every emitted token and every cache state is bit-identical to engine 2.0.12. The speed-up comes from a certified greedy head, which is on by default. This release also adds an experimental prompt lookup mode for LFM3. Weights, quantization formats and the other models are unchanged.

## New in this release: certified greedy head for LFM3

LFM3-1.7B reads about 2 GB of weights per generated token. About a quarter of that is the 524 MB F16 output head (128,000 × 2,048). The EXL3 layers already run close to the memory bandwidth of the machine, so the head was the remaining lossless lever.

- **How it works.**
  - When a model is first used, the engine keeps an int8 copy of the head (one scale per row, 263 MB) and a rigorous error bound for each row, computed in f64 on the CPU (about 0.15 s).
  - At each greedy step, a Metal kernel scores all rows from the int8 copy. With the bound, it gets an interval guaranteed to contain each row's exact logit.
  - Only the rows that can still win are recomputed, with exactly the same arithmetic as the normal head. In practice this is usually a single row.
  - The token is accepted only when its exact logit beats every other row by more than 2⁻⁵, a margin chosen so the result cannot change after the final F16 log-softmax.
  - Otherwise, for example on near-ties or non-finite values, the step runs the normal head. This happened on about 0.6% of steps in our tests.
- **Lossless by construction.** The int8 copy only rules rows out; it never picks the token. Emitted IDs, text and caches are the same as without the feature.
- **When it is used:**
  - LFM3 models with a dense F16 head;
  - greedy requests only (temperature 0 or top-k 1, repetition penalty 1, presence penalty 0);
  - no MTP head, no draft model and no prompt lookup.
  
  Every other request takes the 2.0.12 path unchanged, including sampled requests (temperature above 0 with top-k above 1) such as Desktop's default chat temperature.
- **Setting.** The feature is on by default. Set `MLXL3_LFM3_CERT_HEAD=0` in the engine environment to disable it.
- **Reporting.** Bridge completions on this path add `lfm3_cert_tokens` (steps certified by the int8 head) and `lfm3_cert_fallbacks` (steps that used the full head) to their stats.

## Experimental: prompt lookup for LFM3

- With `MLXL3_LFM3_LOOKUP=1`, greedy LFM3 requests propose continuations copied from the prompt and verify them exactly with the target model.
  - Verification uses a multi-row head that reads the weights once, plus batched layer steps. Both are bit-identical to token-by-token decoding.
  - Accepted proposals are committed, rejected ones are discarded, and output is unchanged.
- `MLXL3_LFM3_LOOKUP=lines` accepts proposals only when the copied context contains a newline.
- This mode is **off by default**. It only helps when the answer copies the prompt: on the M5 measurements, copying a paragraph was 79% faster and editing code 27% faster. Other prompts were neutral.
- It is mutually exclusive with the certified head. When it is on, the certified head is not used.
- Stats: `lfm3_lookup_blocks`, `lfm3_lookup_proposed_tokens`, `lfm3_lookup_accepted_tokens`.

## Performance

Measured on an Apple M5 MacBook (24 GB, on AC power), LFM3-1.7B EXL3 8 bpw. We used the bridge with greedy decoding (top-k 1, no penalties), a 4096-token context, the prompt cache off and 256 generated tokens.
- Each case alternated off/on/on/off with 2 repetitions and a warm-up.
- Token hashes were identical across all passes.

| Prompt | Off (tok/s) | On (tok/s) | Decode | Total time |
|---|---:|---:|---:|---:|
| Copy a paragraph | 50.32 | 57.07 | +13.4% | −11.6% |
| Edit code | 50.33 | 56.82 | +12.9% | −11.2% |
| Write code | 50.52 | 56.76 | +12.4% | −10.9% |
| French explanation | 50.41 | 56.59 | +12.3% | −10.8% |
| Long prompt (2,248 tokens) | 48.61 | 54.10 | +11.3% | −7.9% |

- In an isolated microbenchmark, the head plus token selection went from 4.52 ms to 2.37 ms per step. That is the bandwidth limit for reading 263 MB.
- These ABBA numbers were measured on the feature commit before the rebase onto 2.0.12. The rebase did not change the certified head code.
- A single non-alternated pass of the released binary gave +9% to +13% (copy 45.7 → 49.7, write code 45.8 → 51.9 tok/s, under heavier system load). It is indicative only.
- Prompt processing (prefill) is unchanged. Memory use grows by 263 MB of GPU memory while an LFM3 model is loaded.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

No Desktop update is required. Desktop 2.0.5 and earlier versions work with this engine unchanged.

## Verification and limits

- **GPU correctness on a physical Apple GPU** (LFM3-1.7B EXL3 8 bpw, run alone, on the release source):
  - The exact recomputation kernel matched the production head bit for bit on all 128,000 logits at every tested step.
  - Over 120 real decode steps plus 6 synthetic inputs:
    - the bound was never violated, and the largest observed error was 12% of the bound;
    - 125 of 126 steps were certified, with no wrong token;
    - a zero input (all-tie) and a NaN input both fell back.
  - A deliberate bug (bounds divided by 100) was detected: 8.7 million bound violations.
  - On a synthetic head, duplicate and one-ulp rows fell back, and a clear winner was certified with the matrix-product argmax.
  - Decoding with the certified head gave the same token IDs and bit-identical caches (all 35 layer states) as normal decoding, for budgets 1, 2, 7 and 48 on three fixtures.
  - The prompt lookup oracles (multi-row verification and sessions, against token-by-token decoding) also passed.
- **Released archive, extracted into a new directory:**
  - Its four files matched the manifest: release tag `engine-v2.0.13`, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files.
  - The three Mach-O files passed strict signature and arm64 checks.
  - `mlx.metallib` is identical to MLX 0.32.2.
  - `runtime-info` reports version 2.0.13, MLX, chat and source `c5626a134537`.
- **Bridge parity of the released binary.** We compared default settings with `MLXL3_LFM3_CERT_HEAD=0` on 17 requests: copy, edit, code, French, a long prompt, end-of-sequence and budgets of 1 to 768 tokens.
  - Token hashes, text, cache context, stop reasons and token counts were identical.
  - Of 2,509 decode steps, 2,494 were certified and 15 used the full head.
  - The outputs with the feature off are identical to those of the pre-rebase source.
- **Presence penalty.** A temperature-0 request with presence penalty 0.5 did not use the certified head (no `lfm3_cert_*` stats), and a plain greedy request did. Before release, the LFM3 gates were moved to the same greedy rule as the sampler, so the presence penalty introduced in 2.0.12 cannot route a sampled request to the greedy-only paths.
- **Local checks on the release source:**
  - Rust formatting passed.
  - Strict Clippy passed for `chat`, `mlx,chat` and `direct-metal,chat`.
  - CPU library: 441 passed, 9 ignored; binary 38 passed.
  - MLX library: 485 passed, 269 ignored; MLX binary: 54 passed, 10 ignored.
  - Direct-Metal library: 467 passed, 140 ignored; binary: 39 passed, 1 ignored.
  - The ignored tests are physical-GPU or fixture tests; the relevant GPU tests above ran separately.
- **Not verified:**
  - The error bound and the 2⁻⁵ margin are proved by hand in the module documentation and checked by an independent f64 CPU oracle and the GPU oracles. They have no machine-checked proof, and Kani was not run.
  - The GPU oracle fixtures use token-ID sequences, not natural text. Natural text is covered by the 17 bridge requests.
  - Only LFM3-1.7B EXL3 8 bpw was tested. Other LFM3 checkpoints with a dense F16 head use the same path but were not run.
  - LFM3-VL image requests use the same path but were not run with images.
  - M1–M4, 8/16 GiB Macs and macOS 26.2 were not tested.
- **Hosted CI (added after publication)** on `main` commit `a1b73a9dc`. This commit is the release source plus a regenerated MLX source inventory and documentation; it changes no engine code.
  - Passed: Flash profile compatibility, EXL3 conversion contracts (`conversion`, `packing-proof`), native coverage inventory, `mlx-package`, `native (ubuntu-latest)`, the Kani allocator and speculative shards, and the Metal and loader contracts.
  - On the first push (`48d6a0888`), the native coverage inventory check failed because the reviewed inventory had not been updated for the new certified head and prompt lookup files. `a1b73a9dc` fixes this. The released archive is unaffected, because that check only reads sources.
  - Failed or cancelled, exactly as on the `main` commit before this release (`a08acf560`):
    - `desktop-and-protocol` and `desktop-preferences`: the hosted Swift compiler times out type-checking `ChatComponents.swift:390`, a file this release does not touch. The Python suite inside that job passed: 2,776 passed, 169 skipped;
    - the Kani `core`, `routes`, `flash` and main shards: an internal Kani compiler panic plus harness failures and timeouts in files this release does not touch;
    - `native (macos-26)` and `guard-proofs`: cancelled, as on every recent `main` commit.
  - These failures were not introduced by this release, but they mean hosted CI is **not green**. The local checks above are the release verification. The full local Python suite on the same tree gave 2,839 passed and 106 skipped.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the request and the release decision. Thanks to [Liquid AI](https://huggingface.co/LiquidAI) for the LFM models, [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime and custom Metal kernels, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format, [PonyExl3](https://github.com/beamivalice/PonyExl3) for the Metal EXL3 port, and [prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) for the idea behind the experimental lookup mode.

## Source and assets

Engine source commit `c5626a1345376d4977861329d7b7a6e80ff9ab22`. It is engine 2.0.12 (`3075413a5`, documented in `a08acf560`) plus commit `646727b8e` (certified head and prompt lookup) and commit `c5626a134` (default on, greedy gates, version 2.0.13).

- `MLXL3-Engine-v2.0.13-arm64.tar.gz`: 68,166,189 bytes, SHA-256 `f3ff86d7599e0f63ea8c68e6528d8f7b815a36bd1f5ac3c6b8aa9150e96abf31`.
- `MLXL3-Engine-v2.0.13-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `804a29284d5333804e8f69e2bd94541abfb7574e7d8c6cb3b283bc1abd7617d2`.
- `mlxl3`: 12,863,424 bytes, SHA-256 `644c2e082c1def333c91100410cb3598d2ab3f1bc4737573b18f66cd5a888e00`.
- Detailed trial records: `opti.md` (entries `LFM3-CERT-HEAD-20261116`, `LFM3-PLD-20261116`, `DIAG-20261116-M5-BW`) and `docs/measurements/lfm3-cert-head-20261116/`, `docs/measurements/lfm3-pld-20261116/`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
