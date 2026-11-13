# MLXL3 Engine 2.0.9 beta 1

## Changes and defaults

The engine replays the original single-row GDN operation graph on eligible Qwen3.6-35B-A3B models on Apple M5 with MLX 0.32.2. Set `MLXL3_Q36_AR_REPLAY=0` before loading the model to select the ordinary path.

Small routed SwiGLU expert batches now prepare activations in the layout already consumed by the mapped Metal projection. This path applies to one through four rows, width 2048, intermediate width 512, 256 experts, top-8 routing and EXL3 MCG K2/K3. It covers AR and MTP verification at depths 1, 2 and 3. The target vocabulary, weights, quantization and rounding rules are preserved. MTP starts at depth 1. Tune can select a different depth for the model, runtime and hardware. The compact draft projection introduced in 2.0.8 remains automatic.

## Fixes

The engine preserves the bit pattern when loading mixed signed and unsigned 16-bit expert trellises. It validates bank dimensions and formats before dispatch. The diagnostic codec rejects empty, ragged and out-of-range routes and can process a subsequent valid request.

## Performance

On an Apple M5 with 24 GiB, a preliminary bridge comparison isolated expert preparation on the 2.49 bpw 35B checkpoint with the same 4-bit MTP head. The two MTP2 orders measured +5.45% and +4.04% on the French prompt and +3.99% and +4.33% on the code prompt. Each pass warmed up for 128 tokens and measured two 128-token replies, greedy, context 4096, with prompt reuse off. The comparison used a local baseline with AR replay. The companion validation report records the separate comparison of the packaged engines and variations between passes.

The packaged 2.0.9 versus published 2.0.8 comparison used the same two prompts and BAAB order. Rates are median post-first-token decode throughput. The ranges across prompts and between passes matter, especially for MTP3.

| Mode | French, 2.0.8 → 2.0.9 | Code, 2.0.8 → 2.0.9 |
|---|---:|---:|
| AR | 47.98 → 51.10 tok/s (+6.51%) | 45.77 → 50.03 (+9.30%) |
| MTP1 | 78.17 → 80.34 (+2.78%) | 76.40 → 78.56 (+2.84%) |
| MTP2 | 64.51 → 67.89 (+5.25%) | 65.72 → 68.39 (+4.06%) |
| MTP3 | 60.20 → 63.30 (+5.15%) | 61.56 → 61.53 (−0.05%) |

The isolation run for the experts alone also repeated MTP3 in the opposite order (+2.41% French, +1.97% code). Those comparisons have different baselines. The prefill and latency measurements appear in the validation attachment.

## Installation and compatibility

Install the full Apple Silicon archive through the Desktop engine updater or extract its contents together on macOS 26.2 or newer. This independent engine prerelease uses the engine update channel. Its ad-hoc signatures can be checked locally. Hardware performance qualification covers Apple M5.

## Verification and limits

The release qualification checks component outputs, full target logits and cache/state snapshots against saved references, real bridge requests and explicit depths. It includes cancellation and recovery. The Kani checks cover CPU admission conditions. Physical tests sample specific inputs. Hardware DRAM and ALU utilization were unavailable. The experimental K4 and K6 reader rewrites are excluded from this release.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the 35B profiling and target hardware, [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime, [Qwen](https://huggingface.co/Qwen) for the model, and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for EXL3. The existing MTP path builds on contributions from [MTPLX](https://github.com/youssofal/mtplx) and [TensorFold](https://github.com/ashhart/TensorFold).


## Source and assets

Engine source commit `e7923197cbafc4f63688a8d5cb4a2ba455c4e6e9`.

- `MLXL3-Engine-v2.0.9-beta.1-arm64.tar.gz`, 68,086,449 bytes. SHA-256 `7d8c6eb1f5c116d16783f443f499aad41b24f2b683fe34859006ac3075b56c6c`.
- `MLXL3-Engine-v2.0.9-beta.1-arm64.tar.gz.sha256`, 106 bytes. SHA-256 `2dd77da03ca887aeed42a4b6d744a6d9b35dd043a37e32840e60e200b9e2b284`.
- The validation attachment lists component sizes, hashes and test scope. The [35B profile](https://github.com/0xZKnw/mlxl3/issues/172) includes observed GPU activity, prefill, decode and host waits.
