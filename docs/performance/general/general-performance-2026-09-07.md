# General performance R&D — 7 September 2026

Local work after `27b899ac67b665e1665b0600fd7f02c7fd2ad41b`: Apple M5,
24 GiB unified memory, MLX 0.32.2. No new checkpoint quantization, reduced
calibration, approximate attention, or speculative decoding. These changes
have not been bundled into the installed desktop app or published as a release.

## TensorOps prefill: hoist EXL3 address calculations

The QMM kernel now computes each cooperative element's permutation, bit window
and relative word offsets once, outside the input-depth loop. The loop changes
only the weight-tile base pointer. TensorOps tile geometry, codebook arithmetic,
rounding and accumulation order are unchanged. Per-lane address storage is
bounded; an unexpected cooperative capacity uses the complete original path,
not an early return with unwritten outputs.

The 87 kernel checks cover K=1–8, all three codebooks, all supported BM/BN/BK
tiles, strided grouped weights, depth/offset boundaries, a forced capacity
fallback, dtype/padding and invalid shapes. Full-model forced continuations on
LFM2.5-2.6B and Qwen3.6-35B-A3B (512-token prompt, eight steps, cache forks)
match logits and all recurrent/KV cache state bit-for-bit, with finite logits.

Three alternating production LFM2.5-2.6B EXL3 4 bpw comparisons, 2,571 prompt
tokens and 32 generated tokens: **+6.84% median prefill throughput**
(paired range +6.78–9.36%). Decode is unchanged within measurement noise
(−0.22% median); the measured uncached peak allocation is identical. Cached
requests evaluating only four tokens do not gain from this prefill kernel.
Raw evidence: `build/qmm-hoist-lfm-production.jsonl`. The earlier prototype's
results are not added to this final comparison.

Qwen3.6-35B-A3B EXL3 2.49 bpw, three alternating pairs (`--repeat 128`,
32 generated tokens): **+6.09% median prefill throughput** (range
+4.24–6.25%). Uncached decode +0.20% and cached decode −0.18% are noise-level,
not decode gains. Cached TTFT was about 1.4% slower; that tiny suffix does not
use the changed TensorOps path. Generated texts match in all LFM/Qwen pairs.
Raw evidence: `build/qmm-hoist-qwen-production.jsonl`.

Enabled by default only within the existing M5 TensorOps path; use
`MLXL3_QMM_ADDRESS_HOIST=0` for the original addressing. Other backends remain
unchanged. `TENSOR_KERNEL_ABI` is now 2: old optional tuning profiles must be
regenerated, not silently reused with different kernel performance.

Reproduce in separate processes:

```sh
.venv/bin/python -m pytest -q tests/test_qmm_address_hoist.py
.venv/bin/python benchmarks/compare_runtime.py models/LFM2.5-2.6B-EXL3-4bpw --output build/qmm-lfm-new.jsonl --pairs 3 --tokens 32 --repeat 256 --baseline-env MLXL3_QMM_ADDRESS_HOIST=0 --candidate-env MLXL3_QMM_ADDRESS_HOIST=1
.venv/bin/python benchmarks/validate_forced_decode.py models/Qwen3.6-35B-A3B-EXL3-2.49bpw --output build/qmm-qwen-forced-new --steps 8 --prompt-tokens 512 --fork-every 4 --baseline-env MLXL3_QMM_ADDRESS_HOIST=0 --candidate-env MLXL3_QMM_ADDRESS_HOIST=1
```

## Session reuse: retain a useful cache under a memory budget

The previous session-pool policy reset an entire active conversation if its
snapshots together exceeded the budget, even when its completed-turn cache
alone fitted. It now evicts old block snapshots first, then keeps an admissible
completed-turn cache (or stable-prompt fallback). The final budget check remains.

A real-model test also exposed a transcript bug: restoring a template-consumed
reasoning opener invented a newline. LFM's template ends in bare `<think>`;
that added byte invalidated the next turn's exact prefix. Restoration now uses
the literal suffix of the template, including only its actual whitespace.

LFM2.5-2.6B EXL3 4 bpw, two real exchanges, a 3,230-token first prompt and
a deliberately constrained session-cache budget. Three alternating measured
pairs after an excluded warmup, with an unpruned-cache correctness control:

| Second turn | Old whole-session reset | Retain admissible cache |
| --- | ---: | ---: |
| Median TTFT | 5.1517 s | 0.1673 s |
| Prompt tokens evaluated | 3,529 | 29 |
| Prompt tokens reused | 0 | 3,500 |

This is **96.75% less TTFT for this eviction scenario**, not a universal decode
speedup. The transcript fix applies on both sides of this policy comparison.
Generated text and token IDs match the cold baseline in all pairs. All cache
states match the unpruned warm control bit-for-bit. Cold recomputation uses a
different batch partition and its floating-point cache bytes do not match;
it is not presented as an all-input bit-exact cold/warm claim.

Reproduce:

```sh
.venv/bin/python benchmarks/bench_session_prune.py --output build/session-prune-new.jsonl --pairs 3
.venv/bin/python -m pytest tests/test_session_prune.py tests/test_cli.py tests/test_v1_regressions.py -q
```

Raw evidence: `build/session-prune-lfm-controlled.jsonl`. Earlier failed pilots
remain locally: the first response was truncated at 64 tokens; a second pilot
found the invented newline; a third exposed the need to distinguish cold
recomputation from an unpruned warm-state control. None is a speedup claim.

## Streaming code preparation

Completed thinking/answer/code views use equality gating. Active code retains
stable lexical chunks and reparses its tail. Replacements and ambiguous escaped
fragments use the original complete scan. It does not change the highlighter's
regular expression or turn display highlighting into a language compiler.

Initial `Swift -O` CPU preparation microbenchmark, 30 small appends per size:

| Code size | Complete preparation per append | Incremental preparation |
| --- | ---: | ---: |
| 64 KiB nominal | 3.295 ms | 0.237 ms |
| 256 KiB nominal | 13.045 ms | 0.501 ms |
| 1 MiB nominal | 52.482 ms | 1.640 ms |

These are chunk preparation timings, **not FPS, CoreText layout timings, or
model tok/s**. 1,194 streamed fragments plus Unicode, replacement, escape,
fence/table/math and SwiftUI-state checks passed. The command in
`tests/streaming-render-check.swift` builds the actual Swift sources; set
`MLXL3_RENDER_BENCHMARK=1` for timings. Parent Markdown chunking remains a
separate cost; this cache alone does not remove that scan.

Three additional bounded string-length checks preserve grapheme boundaries
without counting the complete suffix unnecessarily. A separate optimized-build
A/B for a long line measured 3.561 → 2.946 ms (262,144 ASCII bytes) and
54.674 → 29.909 ms (262,144 Unicode graphemes, ~2.88 MB). Ordinary prose showed
smaller reductions, about 2.6–5.4%. These results are separate experiments and
must not be multiplied by the incremental-code speedup above.

## UI bridge: do not wait for decoding on the main thread

Display callbacks used to synchronously join the engine's IO queue just to
validate a process UUID. An occupied JSON decoder could therefore stall the
main thread. A native `Mutex` now protects only the UUID. Start/stop retain
their IO barriers; stop invalidates the ticket before waiting. Every decoded
event and buffered delta carries its original ticket, so an old in-flight
parse cannot be relabeled as a new process's output.

Five synthetic pairs with the IO queue deliberately occupied for 50 ms:
median callback delay **53.877 → 0.034 ms**. This is neither FPS nor inference
throughput. `tests/bridge-dispatch-check.swift` includes its standalone build
command and checks stale/current callbacks, stop during actual decoding,
reload, cooperative cancellation and stale/current process termination.
The Swift build, streaming/state checks, desktop hardening and timeline checks
passed with the final bridge.

## Quantization: avoid unused CPU work

The output-only measurement patch and private LDL-buffer reuse are described
in [the quantization notes](metal-quantization-optimization.md). Three alternating
four-projection K=4 comparisons of full diagnostics versus output-only scoring:

| Pair | Full diagnostics | Output-only | Time reduction |
| --- | ---: | ---: | ---: |
| 1 | 42.659 s | 42.016 s | 1.51% |
| 2 | 46.297 s | 43.098 s | 6.91% |
| 3 | 45.279 s | 44.964 s | 0.69% |

Packed-weight, scale, reconstructed-output fingerprints and selected scores
are identical. Median paired reduction: **1.51%**, with substantial run-to-run
variation; no full-model conversion-time extrapolation. Both sides use in-place
LDL, so this measures only the optional diagnostic omission.
Raw evidence: `build/generic-quant-output-metrics.jsonl`.

The combined four-change comparison (proxy omission, in-place private LDL
buffer, zero-shrinkage diagnostic reuse, reconstruction-only GSS) then measured:

| Pair | All four previous paths | All four optimized paths |
| --- | ---: | ---: |
| 1 | 43.336 s | 42.381 s |
| 2 | 43.389 s | 42.030 s |
| 3 | 43.673 s | 42.599 s |

**2.46% median paired time reduction**, all fingerprints and scores exact.
This comparison supersedes, rather than adds to, the proxy-only percentage.
Raw evidence: `build/generic-quant-preparation.jsonl`. The two avoided
float32 `N x N` temporaries are each 462,422,016 bytes at `N=10752`, in
different phases; this is not a claim of a 925 MB process-peak reduction.

## Rejected or still experimental

- Removing a prefill MoE gate/up copy passed numerical gates but reduced real
  Qwen prefill throughput by roughly 3–6%; removed from production.
- Decode gather/Hadamard fusion passed numerical gates but early Qwen timings
  were inconsistent. Three final LFM8 pairs measured only +0.46% median decode
  uncached and +0.02% cached, around 79 tok/s. Removed rather than enabling an
  unproven gain. Raw evidence: `build/lfm8-decode-gather-final.jsonl`.
- Replacing 64-bit QMM extraction with a 32-bit funnel was slower; rejected.
- Preloading QMV words produced mixed synthetic timings and remains excluded.
  QMM address hoisting instead proceeded to the real-model validation above.
- Extending existing grouped projections to LFM's `w1`/`w3` names gave only
  +1.33% median decode throughput, but −6.59% prefill throughput, worse cached
  TTFT and about 171 MB more cached-request peak allocation. Rejected; the
  compacting copies of grouped slices need solving first. Three paired runs:
  `build/lfm-w1-w3-groups.jsonl`.
- Caching empty recurrent-cache metadata reduced a CPU stub wrapper from
  roughly 580–600 ns to 403–434 ns. About 180 ns saved across 30 layers is only
  5.4 µs/token (roughly 0.02% of 25 ms). Removed the additional singleton/device
  bookkeeping; no model speedup was claimed from the 42% microbenchmark ratio.
  Raw evidence: `build/recurrent-metadata-cpu.jsonl`.

## Compatibility found during measurements

The LFM2.5-8B-A1B fixture could not pass header validation: 22 descriptor
entries named router biases `feed_forward.gate.expert_bias`, while the actual
tensors and MLX architecture use `feed_forward.expert_bias`. Validation now
accepts that precise legacy alias only for `lfm2_moe`, only when the original
key is absent and the alternative shape is correct. Wrong shapes, missing
ordinary/quantized weights and unrelated architectures remain rejected.
No checkpoint file or weights were edited.
Header validation (8,840 entries), full loading and generation then passed on
the LFM8 fixture; this is no longer only a synthetic metadata test.

## Method and primary references

Final gates with address hoisting enabled by default:

- Full local Python/Metal suite: **548 passed, 2 skipped** (uninstalled local
  fixtures for checkpoint and Ling integration).
- Upstream conversion measure/Hessian/Metal tests with MLXL3 search installed:
  **47 passed**.
- Swift build, bridge race/latency checks, streaming/state checks, desktop
  hardening, chat timeline and isolated MCP/preferences/Metal-footprint checks
  passed. No installed app bundle was replaced.
- Ruff passes on the new Python checks/benchmarks and kernel/tuning files;
  existing checkpoint/CLI lint warnings were confirmed against the unchanged
  baseline and not mixed into this performance patch. Staged whitespace checks
  pass excluding `.patch` files, whose blank context lines retain the unified
  diff's required space prefix.

Local final logs: `build/general-rd-final-pytest.log`,
`build/general-rd-final-upstream.log`, `build/general-rd-final-mcp.log`.

The `inference-engineering` skill guided workload matching, serialized heavy
tests, alternating comparisons and separate kernel/request/render boundaries.
GPU and heavy UI tests never intentionally run together. Thermal state is not
controlled; power was recorded and the later session test ran on AC charging.

- [Apple: SwiftUI performance and Instruments](https://developer.apple.com/videos/play/wwdc2025/306/)
  — identify repeated body work and cache expensive preparation.
- [MLX: lazy evaluation](https://github.com/ml-explore/mlx/blob/main/docs/src/usage/lazy_evaluation.rst)
  — timings must include materialization; async graph submission is not a
  measured GPU duration.
- [Apple: Metal 4 tensor operations](https://developer.apple.com/videos/play/wwdc2025/315/)
  — the existing TensorOps path remains the basis of prefill experiments.
