# MLXL3 Engine 2.0.11 beta 2

Independent engine prerelease for Apple Silicon. It builds on 2.0.11 beta 1 and changes how the MTP head drafts and how much of each draft is verified. Weights, quantization, vocabulary and greedy outputs are unchanged: MTP verification stays greedy-exact, so every depth produces the same tokens as AR.

## Changes and defaults

- **Post-norm MTP wiring, on by default.** The MTP head is now seeded with the trunk's final-norm hidden state. Each draft level is then chained from the head's normalized residual, which is the input of its `lm_head`. On Qwen3.6-35B-A3B this accepts more draft tokens per cycle at MTP2 and MTP3. Set `MLXL3_MTP_POST_NORM=0` before loading the model to select the previous pre-norm graph.
- **Confidence-based verify truncation (EVICT), on by default.** After drafting the full depth, the engine verifies only the draft prefix that maximizes expected accepted tokens per unit of cost. The decision uses the top-1 draft log-probabilities, read in the same device transfer as the draft ids. It uses exact integer arithmetic, ties keep the shorter prefix, and a non-finite confidence fails the cycle instead of guessing. At MTP1 it never changes the width. Set `MLXL3_MTP_EVICT=0` to always verify the full draft.
- **Strict settings.** Each variable accepts unset, `0` or `1`. Any other value is rejected when the MTP head loads.
- **Tune profiles.** The Tune key now records both settings (`mtppostnorm-v1`, `mtpevict-v1`). Profiles saved by beta 1 are not reused, so Tune measures again on first use.

The EVICT survival table and costs were calibrated on Qwen3.6-35B-A3B EXL3 with the post-norm head on Apple M5. With other MTP models or devices, or with `MLXL3_MTP_POST_NORM=0`, outputs remain exact, but the truncation policy is not calibrated.

## Performance

Packaged 2.0.11 beta 1 versus packaged 2.0.11 beta 2 on Apple M5 with 24 GiB and macOS 27.2. Workload: Qwen3.6-35B-A3B EXL3 2.49 bpw with the same 4-bit MTP head, greedy, context 4096, prompt cache off. The French and code prompts are 27 and 43 tokens. Each engine ran twice per prompt in BAAB order. Each pass had one 128-token warmup and two measured 128-token replies. Rates are median decode throughput after the first token. Every campaign produced identical tokens, text and history in both engines.

| Mode | French, beta 1 → beta 2 | Code, beta 1 → beta 2 |
|---|---:|---:|
| AR (control) | 58.85 → 58.94 tok/s (+0.15%) | 58.95 → 58.97 (+0.05%) |
| MTP1 | 79.59 → 81.67 (+2.62%) | 76.72 → 78.19 (+1.91%) |
| MTP2 | 73.96 → 76.11 (+2.91%) | 73.41 → 79.31 (+8.04%) |
| MTP3 | 75.90 → 89.63 (+18.08%) | 72.99 → 81.59 (+11.78%) |

- **MTP3** is the clearest result. Every beta 2 pass was faster than every beta 1 pass. An earlier campaign of the same protocol, discarded because another build was running, measured +15.57% and +9.35%.
- **MTP2:** the code prompt gained +8.04%, and +7.59% in the earlier campaign. On the French prompt, one beta 2 pass ran at 68.00 tok/s against 84.23 for the other, so the median shows +2.91%. The earlier campaign measured +12.56% with stable passes, so the French MTP2 gain lies somewhere in that range.
- **MTP1 and AR:** the passes of the two engines overlap, so no change is established.
- **Where the gain comes from:** acceptance per cycle is almost unchanged, but EVICT verifies 18–32% fewer draft tokens at MTP2 and MTP3. For MTP3 on the French prompt, beta 1 accepted 81 of 138 proposed tokens in 46 cycles. Beta 2 accepted 82 of 94 in 45.
- **Prefill and latency:** prefill and time to first token moved between −3.1% and +3.8%, with no consistent sign. At MTP2 and MTP3, end-to-end latency fell by 0.3–14.1%.
- **Tune:** the bridge qualification ran Tune once, while another process was compiling. It selected MTP3 (AR 48.20, MTP1 69.14, MTP2 73.13, MTP3 79.08 tok/s). This run is indicative only.

These figures combine both changes against beta 1, from one order on AC power. The isolated trials of each change used other baselines and must not be added to them.

## Installation and compatibility

Install the archive through the Desktop engine updater on the beta channel, or extract its files together. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later.

The bundled MLX 0.32.2 libraries and Metal library are byte-identical to beta 1 and 2.0.10. The archive is ad-hoc signed. Only the MTP path changes. AR, prefill, DFlash2 and the CLI forward path are unchanged.

## Verification and limits

- **Archive:** the archive was extracted into a new directory. Its files matched the manifest, and the three Mach-O files passed strict signature and arm64 checks. `runtime-info` reported version 2.0.11, MLX, chat and source `3fe7b08972f5`.
- **Desktop updater:** the real `EngineRuntimeStore` installed the archive into a disposable store, relocated it, ran it and passed the fallback checks. A temporary test driver supplied the beta tag carried by the artifact.
- **CLI and bridge:**
  - the extracted CLI matched four saved 248,320-logit vectors bit for bit;
  - bridge AR/MTP1 token hashes matched the reference for budgets 1, 2, 3, 17, 64 and 128.
- **MTP and DFlash2:**
  - the depth check covered AR against MTP1/2/3, budgets, cancellation, recovery, prefix reuse and Tune, and passed;
  - its 35 completions and the tuner token hashes matched the beta 1 release build;
  - the default Tune key contains `mtppostnorm-v1=1:mtpevict-v1=1`;
  - DFlash2 produced 33 identical completions;
  - two real text turns through the Desktop protocol recalled the supplied project code.
- **Defaults, on the release source:**
  - with both variables unset, the MTP cycle diagnostic matched the earlier explicit `POST_NORM=1, EVICT=1` run exactly. Across 4 prompts × 3 depths, that covers tokens, cycles, accept histograms, draft levels and all 1,366 truncation decisions;
  - an independent float64 recomputation of the policy agreed with every recorded decision, and device top-1 bits equalled the host values;
  - with both set to `0`, the diagnostic matched the previous pre-norm default exactly;
  - with `EVICT=0` alone, it matched the post-norm trial exactly;
  - the session, lookup and cache tests kept exact states with the new defaults;
  - real loads rejected `MLXL3_MTP_EVICT=2`, `MLXL3_MTP_EVICT=true` and `MLXL3_MTP_POST_NORM=2`.
- **Earlier trials:**
  - post-norm requalified the compact draft vocabulary at MTP2 and MTP3;
  - a deliberate EVICT policy mutation that ignored confidence was detected.
- **Local test suites:**
  - these suites ran on `f568dd08`; the next commit changes only the MLX inventory JSON;
  - Rust formatting and strict Clippy passed for `mlx,chat` (all targets), no default features, `chat`, `hybrid-dispatch` and `direct-metal,chat`;
  - the MLX/chat library suite passed 490 tests with 251 ignored (474 in the library binary, the rest in single-test child processes), and the binary 51 with 9 ignored;
  - the CPU suites passed 448 (library), 554 (`chat`) and 556 (`hybrid-dispatch`) tests, including child processes, each with 9–10 ignored;
  - the reviewed MLX source inventory was updated for the new MTP sites and matched 322 files;
  - Python checks passed: `compare_native` (71), the Kani manifest (5) and ruff;
  - the Linux-only Kani CI tests (53) were skipped on macOS and are not counted as run.
- **Kani:**
  - Kani 0.68 verified both setting parsers over symbolic strings of up to four bytes;
  - the confidence band function was verified total and monotone over all finite and non-finite `f32`;
  - the width policy was verified to keep at least one and at most `width` draft tokens for every band combination up to width 4, and to reject invalid input;
  - no harness had a failed check, and all 20 cover goals were reached;
  - these bounded CPU proofs do not cover Metal, MLX, the allocator or model quality.
- **Not verified:**
  - hosted CI for the release commits is reported separately;
  - not measured in this campaign: Apple M1–M4, macOS 26.2, MTP models other than the 35B, and quality beyond token parity;
  - the archive comparison used one order (BAAB) on AC power.

The attached validation report, `release-engine-v2.0.11-beta.2-validation.md`, lists components, commands and raw results. It is also in the repository under [docs/releases/engine](https://github.com/0xZKnw/mlxl3/blob/main/docs/releases/engine/release-engine-v2.0.11-beta.2-validation.md).

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the MTP research request, the trial decisions and the target hardware.

Thanks also to the upstream projects:

- [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime;
- [Qwen](https://huggingface.co/Qwen) for the model and its MTP head;
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [PonyExl3](https://github.com/beamivalice/PonyExl3) for EXL3;
- [MTPLX](https://github.com/youssofal/mtplx) and [TensorFold](https://github.com/ashhart/TensorFold) for the MTP path this release builds on;
- the upstream speculative decoding engines surveyed in [issue 174](https://github.com/0xZKnw/mlxl3/issues/174), whose draft-wiring and verify-width designs motivated these trials;
- [DFlash](https://github.com/z-lab/dflash) and [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) for the block speculative decoding research behind DFlash2.

## Source and assets

Engine source commit `3fe7b08972f59c51969962aaa536b48ea6401e15`. It is beta 1 (`c42e4c95`) plus the post-norm wiring `cbf0bebc`, a benchmark option `24d6ee42`, EVICT with the new defaults `f568dd08`, and the MLX inventory update `3fe7b089`.

- `MLXL3-Engine-v2.0.11-beta.2-arm64.tar.gz`: 68,100,591 bytes, SHA-256 `143a5bfc3eb042132b7a77c550f8a0ec7ab7a82c19be6d99ca5df7f7b580a553`.
- `MLXL3-Engine-v2.0.11-beta.2-arm64.tar.gz.sha256`: 107 bytes, SHA-256 `896547d7fe11bf649963953038e3d0c2bbc2fe73e88d5673b009d50e426e27b3`.
- `release-engine-v2.0.11-beta.2-validation.md`: the validation report.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
