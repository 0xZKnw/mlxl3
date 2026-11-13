# Qwen3.8 27B: local kernel work (2026-09-05)

Model: `WatchDG/Qwen3.8-27B-ABLITERATED-EXL3-2.75bpw`, pinned to
`b09c100c0a2885bbf573eac3b7e2c385327d4083`. Apple M5, CLI, greedy generation,
330 prompt tokens and 64 generated tokens. No speculative decoding,
requantization, vocabulary pruning or changes to the model's accumulation precision.

## Enabled changes

- Exact MUL1 byte sum using two independent 16-bit lanes inside a 32-bit
  integer. The floating-point codebook conversion and QMV accumulation are
  unchanged. Disable with `MLXL3_MUL1_SWAR=0` for comparison.
- M5 TensorOps prefill uses 64 rows for large MUL1 matrices when the prompt
  is already 64-aligned and at least 128 tokens. BK remains 16; small/ragged
  prompts and vocabulary heads retain the original tile. Disable with
  `MLXL3_DENSE_PREFILL_M64=0`.

## Evidence and limits

The final order was baseline → byte sum → both → both → byte sum → baseline.
All runs were on battery; absolute performance dropped during the earlier
screening, so early and late timings must not be compared as a speedup.

| Final medians | Baseline | Byte sum | Both |
| --- | ---: | ---: | ---: |
| Decode tok/s | 4.890 | 5.635 | 5.639 |
| Prefill tok/s | 48.62 | 51.07 | 51.86 |
| TTFT s | 7.11 | 6.70 | 6.69 |
| Peak MLX GB | 12.995 | 12.995 | 12.995 |

These are short local runs, not universal guarantees. Earlier screening
reached 8.23 tok/s with the byte sum, but **10 tok/s was not achieved**.
MLX allocation metrics are not total process physical footprint.

All 65,536 MUL1 states match the previous Metal implementation and CPU
oracle bit-for-bit. A 256-token prefill followed by 16 forced decode steps
preserved logits and all recurrent/KV cache states exactly (max error zero).
All final comparison runs generated identical text.

Rejected/not enabled: full codebook LUT, N-major weight transpose, cooperative
shuffle loads, K3 grouped windows, altered SIMD group counts, loop unrolling,
fast math, packed codebook vectors and FP16 byte-dot decoding. Experimental
drivers remain under `benchmarks/benchmark_qwen38_*.py`; they do not change
runtime defaults unless explicitly invoked.

Installed locally only; no GitHub push or release for this work.
