# Decode investigation — 10 September 2026

Research only: no production defaults, installed app, dependencies or model
weights changed. Earlier uncommitted Ling work is preserved.

## Environment and limits

Apple M5, 24 GiB unified memory, macOS 27.0 (26A428), MLX 0.32.2.
The machine started on battery and later switched to AC. Thermal state was
not controlled. Small differences are noise, not validated speedups.

Normal Qwen loading failed during warmup while compiling TensorOps:
`get_destination_cooperative_tensor`: no matching member function, with a
rejected template requirement in the macOS MetalPerformancePrimitives header.
The custom QMM source and the new SDK type constraints need investigation;
the root cause is not yet established. This is a CLI reproduction, not a
separate GUI test.

All successful measurements below set `MLXL3_TENSOR_QMM=0` and
`MLXL3_TENSOR_SEGMENTED_QMM=0` on both sides. This is a benchmark-only
workaround: prefill and TTFT numbers do not represent the normal TensorOps
configuration. Decode still uses the serialized EXL3 QMV path.

## Experiments

Three alternating measured pairs per comparison, 96 generated tokens,
warmup excluded. Differences below are medians of paired decode changes,
not ratios between the fastest runs. Generated text matched in every measured
pair; no new bitwise forced-logit/cache gate was run for these rejected paths.

| Candidate | Model | Uncached decode change | Decision |
| --- | --- | ---: | --- |
| Larger Metal command-buffer budgets | Qwen3.6-35B-A3B 2.49 bpw | −0.38% | Reject |
| Same budgets | LFM2.5-8B-A1B 3.10 bpw | −2.41% | Reject |
| Compile all 24 feed-forward blocks | LFM2.5-8B-A1B | +0.11% | Noise |
| Compile all 30 feed-forward blocks | LFM2.5-2.6B 4 bpw | +0.30% | Noise |
| Compile shared dense/grouped/routed QMV entrypoints | LFM2.5-8B-A1B | −3.06% | Reject |

The command-buffer candidate sets `MLX_MAX_MB_PER_BUFFER=1024` and
`MLX_MAX_OPS_PER_BUFFER=1000`. Qwen uncached peak MLX allocation increases
from 12.422 to 13.302 GB. One Qwen pair appears dramatically faster only
because its baseline suddenly slowed; it is not evidence of a 40% gain.
LFM cached decode also regressed (−1.59% median). Neither model supports
enabling this setting globally.

The direct-QMV series ran after AC was connected, separately from earlier
series. Its first pair was especially unstable; the other two also regressed.
The existing stateless compiler was reused for feed-forward experiments.
An older `work/compiled_qmv_bench.py` targets aliases no longer used by the
current forward path; its results must not be treated as current QMV tests.

Local raw evidence (ignored build artifacts):

- `build/decode-command-buffers-qwen-fallback-20260910.jsonl`
- `build/decode-command-buffers-lfm8-20260910.jsonl`
- `build/decode-ff-compile-lfm8-all-20260910.jsonl`
- `build/decode-ff-compile-lfm26-20260910.jsonl`
- `build/decode-qmv-compile-lfm8-ac-20260910.jsonl`
- `build/bench_decode_ff_compile.py` (diagnostic runner)

## Most useful next investigations

1. Restore TensorOps compatibility on macOS 27 before full-path validation.
   Do not turn the benchmark workaround into a silent production default.
2. Investigate larger native/compiled decode regions: attention, normalization,
   RoPE and cache updates together. Compiling isolated FFNs/QMV did not help.
   Cache position and changing context shapes need handling without graph
   recompilation per token. Current Qwen recurrent blocks and several MLPs
   are already compiled, so this is not a claim that graph compilation is new.
3. Revisit grouped projections only where they are currently excluded:
   mixed bit widths/codebooks and compacting-copy overhead. Inspected Qwen
   projections already use compatible K=4 groups; no Qwen gain is assumed.
   Prior LFM grouping gained little decode and hurt prefill/memory.
4. Profile actual GPU dispatch and memory behavior before another QMV layout
   change. Weight-file size divided by tok/s is not measured GPU bandwidth.
   LUT, preload and several layout variants have already been rejected.

A one-token LFM8 CPU profile shows many QMV/Hadamard wrappers, but its Python
timings are neither GPU timings nor proof that eliminating wrappers improves
end-to-end decode. The paired compile experiments demonstrate that distinction.

## Primary references consulted

- [External MLX command-buffer accounting investigation](https://github.com/defai-digital/ax-engine/blob/main/docs/performance/mlx-upstream-issue-gather-qmm-buffer-accounting.md):
  hypothesis checked against local MLX source and tested above, not a gain
  transferable from the author's machine.
- [MLX extension documentation](https://ml-explore.github.io/mlx/build/html/dev/extensions.html).
- [ExLlamaV3 environment variables](https://github.com/turboderp-org/exllamav3/blob/master/doc/env_vars.md):
  native attention/CUDA graph execution is architectural inspiration only;
  Metal does not provide the same CUDA graph API. Reduced-precision activation
  modes and speculative decoding are outside this investigation.

Conclusion: no new general decode optimization validated for production in
this round. Keep the existing engine unchanged rather than ship regressions.
