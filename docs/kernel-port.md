# EXL3 CUDA → Metal port inventory

The pinned source of truth is ExLlamaV3 commit
`ca5270c4b842876ddbe9a28594fbb6eac516cdf2`. PonyExl3 commit
`8e7fa6b1556f59fc669e25087903b279b9b0346f` is used only for independent
compatibility and performance checks.

| Upstream CUDA component | mlxl3 target | Status |
| --- | --- | --- |
| `quant/codebook.cuh`, `exl3_dq.cuh` | fused trellis + codebook decode | working reference + Metal |
| `quant/pack.cu` | trellis pack/unpack | working reference + Metal |
| `quant/hadamard.*`, `hadamard_inner.cuh` | input/output Hadamard | working via MLX primitive |
| `quant/signs.*` | FP16 signs/scales | working; packed legacy signs still queued |
| `quant/reconstruct.*` | exact matrix reconstruction | working CPU + Metal/MLX |
| `quant/exl3_gemv*` | tiled low-bit `m = 1` QMV | working K 1–6/8; K7 scalar fallback |
| `quant/exl3_gemm*` | batched dense GEMM | vector M<24; 32x64 matrix M=24–47; 64x64 matrix M≥48 |
| `quant/exl3_gemv_int8*` | int8-activation QMV | queued |
| `quant/exl3_moe*` | mapped gate/up/down expert QMV | working correctness-first path |
| `quant/quantize_tiles*` | Metal Viterbi trellis search | temporary Pony implementation |
| Python LDLQ/Hessian pipeline | MLX converter | temporary Pony implementation |
| cooperative autotune / kernel map | Apple-GPU shape tuner | queued |

## Porting rules

1. Preserve the serialized EXL3 layout exactly; no private on-disk variant.
2. Validate every Metal primitive against a simple CPU oracle before fusion.
3. Optimize decode (`m=1`) before batched prefill because interactive LFM use is
   bandwidth dominated.
4. Keep kernels specialized by `K`, codebook and common shapes. Generic fallbacks
   remain available for correctness.
5. Benchmark cold compile, warm latency, effective memory bandwidth and tokens/s
   separately.

PonyExl3's useful comparison points are its CPU reference codec, fused Metal
GEMV/GEMM, MoE dispatch and Metal trellis search. Its model loader currently
does not include Ling's `bailing_hybrid` / `BailingMoeV3ForCausalLM`
architecture. Its current loader also reads older JSON codebook flags and can
misdecode newer scalar `.mcg`/`.mul1` markers; comparisons use a local
compatibility correction. mlxl3 instead pins MLX-LM commit
`44b42cc137763309b0662284ce12d7a95b8c5d99`, which contains the upstream
Bailing V3 KDA/MLA/cache implementation, then replaces eligible projections
with EXL3-backed modules.

## Current measured decode kernels

Generic kernel changes are gated with `benchmarks/kernel_bench.py`. The quick
suite covers dense, expert, mapped small-batch, K3/K5, and all three codebook
families; the full suite adds a vocabulary head, expands to K2–K6, and sweeps
batch sizes 1–64. Store machine-local runs under `benchmarks/results/local/`.

```bash
python benchmarks/kernel_bench.py --suite full \
  --output benchmarks/results/local/baseline.json
```

Warm medians on the M5 in this workspace, checked against Pony output:

| Shape / codebook | mlxl3 | PonyExl3 | max difference |
| --- | ---: | ---: | ---: |
| 1024 → 6144, K3 default | 0.247 ms | 0.257 ms | 0 |
| 1024 → 6144, K3 mul1 | 0.271 ms | 0.283 ms | 0 |
| 3584 → 1024, K3 default | 0.246 ms | 0.244 ms | 0 |
| 1024 → 248320, K6 mul1 | 4.094 ms | 4.129 ms | 0 |

The full 0.8B fixture is 16.51 ms/token (60.6 tok/s) in mlxl3 and 16.42
ms/token (60.9 tok/s) in the corrected Pony runtime, with identical logits.

The first generic small-batch pass amortizes trellis decode across groups of
2, 4, or 8 activation rows. Same-process A/B medians against the former mapped
QMV path on the 10-core M5 are:

| Shape / rows | mapped QMV | vector QMM | speedup |
| --- | ---: | ---: | ---: |
| 1024 → 4096 / 2 | 0.280 ms | 0.228 ms | 1.22× |
| 2048 → 1792 / 8 | 0.456 ms | 0.251 ms | 1.82× |
| 1024 → 4096 / 16 | 0.813 ms | 0.309 ms | 2.63× |
| 2048 → 2048 / 32 | 1.354 ms | 0.408 ms | 3.32× |
| 1024 → 4096 / 64 | 2.526 ms | 0.654 ms | 3.86× |

Split-K is selected from both input work and available output parallelism.
Wide projections now avoid redundant partial-output buffers and reductions;
tall/narrow matrices retain split-K. Focused same-process QMV comparisons show
1–3% lower latency, while mapped MoE projections improve by 2–7%. On the full
LFM fixture, the same 128-token benchmark moved from 61.47 to 63.14 tok/s
(+2.7%) with unchanged 4.02 GB peak memory and deterministic text.

For longer prefills, 32x64 and 64x64 QMM kernels decode a 32x64 weight stage
once into threadgroup memory and use Metal simdgroup matrix instructions for
the token rows. The 32-row tile crosses over at M=24; the 64-row tile takes
over at M=48:

| Shape / rows | vector QMM | matrix QMM | speedup |
| --- | ---: | ---: | ---: |
| 1024 → 4096 / 24 | 0.358 ms | 0.298 ms | 1.20× |
| 2048 → 6144 / 26 | 0.814 ms | 0.507 ms | 1.60× |
| 2048 → 2048 / 32 | 0.413 ms | 0.344 ms | 1.20× |
| 2048 → 6144 / 32 | 0.823 ms | 0.508 ms | 1.62× |
| 2048 → 2048 / 48 | 0.522 ms | 0.447 ms | 1.17× |
| 2048 → 6144 / 48 | 1.127 ms | 0.781 ms | 1.44× |
| 1024 → 4096 / 64 | 0.649 ms | 0.411 ms | 1.58× |
| 2048 → 6144 / 64 | 1.412 ms | 0.773 ms | 1.83× |

On the full LFM model with a 51-token prompt, the 64-row kernel improves median
prefill from 109.53 to 113.89 tok/s (+4.0%); decode stays statistically
unchanged. Model-wide measurements made while running on battery are noisier
than the alternating same-process kernel A/B figures above.
