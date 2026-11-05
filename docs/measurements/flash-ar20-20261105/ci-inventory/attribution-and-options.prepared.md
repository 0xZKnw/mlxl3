## Experimental AR20 options and upstream credits

All twelve options below default to OFF. Use only literal `0`/`1`; the M5/shape/capability guards and existing numerical fallbacks still apply. These are source-backed native adaptations, not twelve independently established speedups. The current AR speed selection excludes B01 after its short slowdown and B12 after its worse decode-miss replay. Root's final fresh-control MAIN32 report determines the remaining performance conclusions; no default is changed by component/model parity alone.

| ID | Environment option | Mechanism and origin |
| --- | --- | --- |
| B01 | `MLXL3_NATIVE_GDN_HEAD_GATES` | Per-head whole-GDN gates; oMLX |
| B02 | `MLXL3_NATIVE_QSA_CHAINS` | Smaller split-chain QSA CTAs and paired queries; oMLX |
| B03 | `MLXL3_NATIVE_QSA_PREFETCH` | K/V software pipelining preserving the canonical online order; oMLX |
| B04 | `MLXL3_NATIVE_GDN_RESIDENT_BUNDLE` | Resident qkv/z co-dispatch, adapted to separate EXL3 inputs/outputs; oMLX |
| B05 | `MLXL3_NATIVE_MUL1_SWAR` | Packed byte-pair codeword sums; TurboEXL3 |
| B06 | `MLXL3_NATIVE_TRELLIS_READER8` | Eight-codeword lane reader; TurboEXL3 |
| B07 | `MLXL3_NATIVE_QMV_K_PRELOAD` | Two-K-tile preload preserving accumulator order; TurboEXL3 |
| B08 | `MLXL3_NATIVE_H128_SIMD` | Register/SIMD H128 preserving native rounding; TurboEXL3 |
| B09 | `MLXL3_NATIVE_MOE_FINISH_MIX` | Down-finish/H128/SVH and canonical ten-route mix; TurboEXL3 mechanism with native reduction |
| B10 | `MLXL3_NATIVE_GPU_SPIN_WAIT` | Bounded50µs/4096-poll wait before blocking; Strata |
| B11 | `MLXL3_NATIVE_EXPERT_READAHEAD_OFF` | `F_RDAHEAD=0` on independently opened expert descriptions; Strata |
| B12 | `MLXL3_NATIVE_CACHE_SLRU` | Per-layer50/50 SLRU at the unchanged 53 slots; llama.cpp |

B01 remains available for analysis but is excluded from the current speed composition. B12 is also excluded for AR: on the existing full-route replay its decode misses increased 1.7674% MAIN / 1.4473% HOLD while prefill misses decreased 1.8394% / 2.5362%. These are cache-simulation differences, not tok/s. Forced LRU/prefetch overrides SLRU. The old ALL12 AR10 options are a separate baseline and are not counted as new AR20 work. No MTP/DFlash/P04/HC L3–L5 or cancelled encoder work is included.

The physical components, independent arithmetic oracles and cross-process full model/CLI gates are documented under [components](../components-01/report.json), [quality](../quality/report.json), [matrix](../matrix.md) and [SLRU review](../slru-audit/README.md). CPU proofs are bounded source properties; they do not prove Metal, OS advice/reads or concurrent execution. B12 Kani timed out without a verdict, the global Kani inventory compiler ICE remains unverified, and Python CrossHair was unavailable. Long-context/other models/other GPU families and sustained throughput are not validated by these short M5 checks. This is local/PR work, with no installed-app replacement or release.

Thanks to [davidtai/TurboEXL3](https://github.com/davidtai/turboexl3/tree/c3bb0e260091709f8ad64c84d1fd02808b12dfd1) for the implemented EXL3 reader, packed decoder, preload, SIMD transform and finish/mix mechanisms; [jundot/oMLX](https://github.com/jundot/omlx/tree/2165e75726ba83d9651dbaced61c97728da77bba) for GDN/QSA scheduling examples; [Niko1221/Strata](https://github.com/Niko1221/Strata/tree/ffa80d73133ea5c548aefe57889a16e8085a938f) for host-wait and file-advice examples; and [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/2bc46baa9207d098d03741ac394e7faaf1a955b6) for the concrete SLRU implementation and its documented capacity-dependent outcomes. [MLX-LM](https://github.com/ml-explore/mlx-lm/tree/736ab0d8c1d0edc79c3d798044fe6904e2fa23be) supplies the previously established canonical recurrence basis, and [TensorFold](https://github.com/ashhart/TensorFold/tree/0311c9280878775ea8167d2e26bb7068d7cebc6f) was examined for already-covered expert-axis ideas. Those last two are research references, not additional new speedups. Exact source/caller links and adaptation limits are retained in [Turbo research](../turbo-kernels.md), [GDN/QSA research](../gdn-attention.md) and [SSD research](../ssd-schedule.md).
