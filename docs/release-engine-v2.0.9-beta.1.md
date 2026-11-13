# MLXL3 Engine 2.0.9 beta 1

## Changes and defaults

The engine replays the original single-row GDN operation graph on eligible Qwen3.6-35B-A3B models on Apple M5 with MLX 0.32.2. Set `MLXL3_Q36_AR_REPLAY=0` before loading the model to select the ordinary path.

Small routed SwiGLU expert batches now prepare activations in the layout already consumed by the mapped Metal projection. This path applies to one through four rows, width 2048, intermediate width 512, 256 experts, top-8 routing and EXL3 MCG K2/K3. It covers AR and MTP verification at depths 1, 2 and 3. The target vocabulary, weights, quantization and rounding rules are preserved. MTP starts at depth 1. Tune can select a different depth for the model, runtime and hardware. The compact draft projection introduced in 2.0.8 remains automatic.

## Fixes

The engine preserves the bit pattern when loading mixed signed and unsigned 16-bit expert trellises. It validates bank dimensions and formats before dispatch. The diagnostic codec rejects empty, ragged and out-of-range routes and can process a subsequent valid request.

## Performance

On an Apple M5 with 24 GiB, a preliminary bridge comparison isolated expert preparation on the 2.49 bpw 35B checkpoint with the same 4-bit MTP head. The two MTP2 orders measured +5.45% and +4.04% on the French prompt and +3.99% and +4.33% on the code prompt. Each pass warmed up for 128 tokens and measured two 128-token replies, greedy, context 4096, with prompt reuse off. The comparison used a local baseline with AR replay. The companion validation report records the separate comparison of the packaged engines and variations between passes.

## Installation and compatibility

Install the full Apple Silicon archive through the Desktop engine updater or extract its contents together on macOS 26.2 or newer. This independent engine prerelease uses the engine update channel. Its ad-hoc signatures can be checked locally. Hardware performance qualification covers Apple M5.

## Verification and limits

The release qualification checks component outputs, full target logits and cache/state snapshots against saved references, real bridge requests and explicit depths. It includes cancellation and recovery. The Kani checks cover CPU admission conditions. Physical tests sample specific inputs. Hardware DRAM and ALU utilization were unavailable. The experimental K4 and K6 reader rewrites are excluded from this release.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the 35B profiling and target hardware, [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime, [Qwen](https://huggingface.co/Qwen) for the model, and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for EXL3. The existing MTP path builds on contributions from [MTPLX](https://github.com/youssofal/mtplx) and [TensorFold](https://github.com/ashhart/TensorFold).
