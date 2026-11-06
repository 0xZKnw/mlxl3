# MLXL3 Engine 2.0.4-beta.1

Independent engine prerelease **engine-v2.0.4-beta.1**, runtime version **2.0.4**. Runtime source: **ead60d0323e99bc023c82485dddc89d9903e4ead**.

## Fixes and defaults

LFM3-1.7B and LFM3-VL-2B now select a **512 MiB default MLX allocator cache** before loading weights. This prevents unused decode buffers from accumulating throughout a response. It does not cap live model tensors or the KV cache. The existing `MLXL3_ALLOCATOR_CACHE_MIB` override remains supported; other model defaults and numerical operations are unchanged.

Native Qwen 3.8 Flash Next continues to use the separate Metal helper without MLX. Other models, including LFM3, retain their existing MLX paths and the previously released kernels.

## Memory validation

Apple M5 / 24 GiB, MLX 0.32.2, LFM3-1.7B EXL3 8bpw. Synthetic 6,835-token prompt, context 8,192, greedy decoding, MTP/DFlash/prompt reuse off, one model process at a time.

- Published 2.0.3 reached **6.04 GB** process footprint after 256 generated tokens, including 3.25 GB of unused MLX cache. An explicit 512 MiB control used **3.35 GB**, with identical generated tokens and text.
- The longer unrestricted baseline crossed the **10 GiB safety guard** and was cancelled and reaped. The reported 15 GB peak was deliberately not recreated.
- The actual packaged 2.0.4 engine, without an allocator override, completed **1,024 generated tokens / 7,859 context tokens**, peaking at **3.38 GB** process footprint. Its tokens and text match the published-engine 512 MiB control exactly. Live MLX memory remained about 2.62 GB and unused cache about 0.54 GB.

These are memory diagnostics on this model and shape, not a universal RAM limit or a speed benchmark. A cache cap can change allocation overhead. No new throughput gain is claimed.

## Installation and compatibility

Apple Silicon, macOS **26.2+**. Use the Desktop engine updater or extract the engine archive as a complete directory. Desktop beta 9 also bundles this exact runtime, and automatically uses it instead of an older external 2.0.3 engine. This engine release is published with `--latest=false` to preserve the Desktop update channel.

The archive contains the signed hybrid dispatcher with embedded MLX/native helpers, MLX libraries, Metal library, manifest and upstream notices. Signatures are ad hoc; binaries are not notarized.

## Checks and limits

The packaged bridge completed the long-response regression with bounded waits and joined processes. Runtime/helper identities, manifest hashes and arm64 signatures were inspected. Three focused allocator regressions pass for text/VL initialization and preservation of live arrays, including explicit 0/1 MiB overrides. One real-model test passes on text and VL: 128,000 finite logits, bit-exact logit/state reset replay and actual fused dispatch. The previous unbounded default failed the cache-bound regression.

Kani 0.68.0 was attempted on the actual allocator API and hit an aarch64 compiler internal error; **no allocator, MLX or Metal property was formally proved**. The full test suite was omitted at the user's request. No installed app or engine was replaced by this task.

## Artifacts

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| MLXL3-Engine-v2.0.4-beta.1-arm64.tar.gz | 73029188 | `f3a71e68139fdc9072b991582434148ca2a6c74ee25fb04f2118c5968ef74e9c` |
| MLXL3-Engine-v2.0.4-beta.1-arm64.tar.gz.sha256 | 106 | `1d8f56b366075109385bb1bd24db09d69e1e94144aaf556d3965e9a91c8e62e8` |
| MLXL3-Engine-v2.0.4-beta.1.json | 17741 | `03a954daf6342fdfd9d2d378e9dfe2a4aedf61985e86db94c72538801f8fa677` |
| MLXL3-Engine-v2.0.4-beta.1-validation.json | 4809 | `057716d33d861b06b1a5624bff3da616b04194e969b031fd9aee563ecbb0663d` |

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the reproducible RAM report and this fix; [Apple MLX](https://github.com/ml-explore/mlx) for allocator controls; [LiquidAI](https://huggingface.co/LiquidAI) and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the retained LFM3 reference semantics; and [PonyExl3](https://github.com/beamivalice/PonyExl3) / [MTPLX](https://github.com/youssofal/mtplx) for retained quantization and MTP foundations. Existing upstream notices remain included.
