# MLXL3 Engine 2.0.11 beta 1

Independent engine prerelease for Apple Silicon. It packages the Qwen3.6-35B-A3B work merged since engine 2.0.10 and a new single-pass target head for MTP verification. Weights, quantization, vocabulary and greedy outputs are unchanged.

## Changes and defaults

- **Single-pass target head for verification widths 3 and 4.** On Apple M5, the Qwen3.6-35B-A3B EXL3 target head (K6, MCG, 2048 → 248,320) verifies three or four rows in one tile pass. That pass reads a lane-ordered copy of the hidden input. The permutation is prepared once when the model loads. The multiply-add order is unchanged, so logits are bit-identical to the previous path. MTP2 and MTP3 verify at these widths. Set `MLXL3_Q36_HEAD_LANE=0` before loading the model to select the previous layout. The engine accepts the variable unset, `0` or `1`, and refuses to load the model with any other value.
- **Progressive layer submission on by default.** Qwen generation now submits GPU work after layers 0, 3, 7, …, 35 instead of once per step. This covers AR, prefill, MTP, DFlash2 capture and layer-major verification, on every geometry. Set `MLXL3_QWEN_PIPELINE=0` to disable it. `MLXL3_QWEN_PIPELINE_STRIDE` (1–64, default 4) remains a diagnostic setting.
- **Four-row 35B verification and segmented prefill on M5.** Four-row QKV+Z projections read a lane-ordered input written by a fused kernel. The segmented prefill plans expert routes in parallel.
- **Smaller final prefill layer.** In the last ordinary text layer of the 35B prefill, the routed experts compute only the row that later stages consume.
- **Tune profiles.** The Tune key now records the layer pipeline setting and the head layout. Profiles saved by earlier engines are not reused, so Tune measures again on first use.

## Performance

Packaged 2.0.10 versus packaged 2.0.11 beta 1 on Apple M5 with 24 GiB and macOS 27.2. Workload: Qwen3.6-35B-A3B EXL3 2.49 bpw with the same 4-bit MTP head, greedy, context 4096, prompt cache off. The French and code prompts are 27 and 43 tokens. Each engine ran twice per prompt in BAAB order. Each pass had one 128-token warmup and two measured 128-token replies. Rates are median decode throughput after the first token. Every campaign produced identical tokens, text and history in both engines.

| Mode | French, 2.0.10 → 2.0.11 beta 1 | Code, 2.0.10 → 2.0.11 beta 1 |
|---|---:|---:|
| AR | 51.26 → 53.91 tok/s (+5.18%) | 50.16 → 53.91 tok/s (+7.46%) |
| MTP1 | 72.10 → 72.56 (+0.64%) | 70.77 → 71.25 (+0.67%) |
| MTP2 | 64.95 → 69.32 (+6.73%) | 64.20 → 69.43 (+8.14%) |
| MTP3 | 65.96 → 71.10 (+7.79%) | 63.62 → 68.63 (+7.88%) |

- **MTP1:** the passes of the two engines overlap, so no MTP1 gain is established. MTP1 is still the fastest fixed depth on these two prompts.
- **Tune:** the bridge qualification workload is a single indicative pass. There, Tune selected MTP2: AR 54.44, MTP1 76.45, MTP2 80.12, MTP3 79.74 tok/s.
- **Prefill and latency:** on these short prompts, prefill and time to first token moved between −4.8% and +5.1%, with no consistent sign. End-to-end latency fell by 0.4–6.9%.
- **Long prompts:** the prefill changes target long prompts and were measured on their own on 1,097 tokens. The parallel route plan gained +3.65% and +4.02% in the two orders. The smaller final layer gained +1.07% and +1.11%.

The new head layout was also measured alone. The comparison used the same binary with `MLXL3_Q36_HEAD_LANE=0` and `1`, in both orders, on the same prompts:

- MTP2: +2.95% to +8.06%.
- MTP3: +2.93% to +4.82%.
- MTP1 control, which never uses the new path: −0.58% and +0.46%.

These isolated results and the archive comparison have different baselines, so they must not be added together.

## Installation and compatibility

Install the archive through the Desktop engine updater on the beta channel, or extract its files together. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later.

The bundled MLX 0.32.2 libraries and Metal library are byte-identical to engine 2.0.10. The archive is ad-hoc signed.

The new head layout, the four-row QKV+Z path and the final-layer prefill change apply only to the 35B geometry on Apple M5. Other models and devices keep their previous kernels, but progressive layer submission applies to every Qwen geometry.

## Verification and limits

- **Archive:** the archive was extracted into a new directory. Its files matched the manifest, and the three Mach-O files passed strict signature and arm64 checks. `runtime-info` reported version 2.0.11, MLX, chat and source `a7241d8ba758`.
- **Desktop updater:** the real `EngineRuntimeStore` installed the archive into a disposable store, relocated it, ran it and passed the fallback checks. A temporary test driver supplied the beta tag carried by the artifact.
- **CLI and bridge:** the extracted CLI matched four saved 248,320-logit vectors bit for bit. Bridge AR/MTP1 token hashes matched the reference for budgets 1, 2, 3, 17, 64 and 128.
- **MTP and DFlash2:** the depth check covered AR against MTP1/2/3, budgets, cancellation, recovery, prefix reuse and Tune. It passed, and its 35 completions matched the release build of the head layout commit. DFlash2 produced 33 identical completions. Two real text turns through the Desktop protocol recalled the supplied project code.
- **Head layout:** before packaging, the layout was checked against the serial path:
  - 18/18 head components were exact;
  - 24/24 production verifications were exact for logits, MTP hidden state and all 80 cache/state arrays, across three prompts, widths 1–4 and both settings;
  - two deliberate mutations were detected;
  - the CLI, bridge, MTP depth and DFlash2 results were identical with the setting on and off.
- **Local test suites:**
  - Rust formatting and strict Clippy passed for `mlx,chat` (all targets), `chat`, `direct-metal,chat` and `hybrid-dispatch`.
  - The MLX/chat library suite passed 467 tests with 250 ignored.
  - The CPU suites passed 548 (`chat`) and 550 (`hybrid-dispatch`) tests, each with 10 ignored.
  - The MLX source inventory matched.
- **Kani:** Kani 0.68 verified the head-layout setting parser over symbolic strings of up to four bytes. It also verified the layout admission and grid arithmetic. Neither harness had a failed check, and all 15 cover goals were reached. These bounded CPU proofs do not cover Metal, MLX, the allocator or model quality.
- **Not verified:** hosted CI for the release commits is reported separately. Apple M1–M4, macOS 26.2, models other than the 35B for the new paths, and long-prompt prefill of the packaged archive were not measured in this campaign. The archive comparison used one order (BAAB) on AC power.

The attached validation report, `release-engine-v2.0.11-beta.1-validation.md`, lists components, commands and raw results. It is also in the repository under [docs/releases/engine](https://github.com/0xZKnw/mlxl3/blob/main/docs/releases/engine/release-engine-v2.0.11-beta.1-validation.md).

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the 35B profiling, the optimization requests and the target hardware.

Thanks also to the upstream projects:

- [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime;
- [Qwen](https://huggingface.co/Qwen) for the model;
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [PonyExl3](https://github.com/beamivalice/PonyExl3) for EXL3;
- [MTPLX](https://github.com/youssofal/mtplx) and [TensorFold](https://github.com/ashhart/TensorFold) for the MTP path this release builds on;
- [DFlash](https://github.com/z-lab/dflash) and [dflash-mlx](https://github.com/bstnxbt/dflash-mlx) for the block speculative decoding research behind DFlash2.

## Source and assets

Engine source commit `a7241d8ba758da79ba2e2aff72a9f6993584b22d`.

- `MLXL3-Engine-v2.0.11-beta.1-arm64.tar.gz`: 68,097,098 bytes, SHA-256 `dfd6e4470d3e4a581c2f7b2206f235a0004911f1367a0b93f698bdabcf81106e`.
- `MLXL3-Engine-v2.0.11-beta.1-arm64.tar.gz.sha256`: 107 bytes, SHA-256 `f13be1d1365c1cf39e63676807b945ef8a53f515be0465d9820b4c2806e14c35`.
- `release-engine-v2.0.11-beta.1-validation.md`: the validation report.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
