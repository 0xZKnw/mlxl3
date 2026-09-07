# Exact Metal quantization optimization

This accelerates **conversion**, not inference. It keeps the calibration data,
LDLQ feedback order, regularization, scale search, codebooks, K and tail-biting
passes unchanged. No QAT, fewer calibration rows, or reduced search is involved.

**Latest round (September 7): a further −42.2% at K=2 and −3.38% at K=4
versus the second-round converter**, on four complete LFM projections with
one worker on both sides. Each of those results uses three alternating pairs.
The earlier K=4 round measured −30.84% against a different baseline; do not add
these percentages. Single-pair K=3/5/6 results are reported separately below.
These are not full-model timings. The kernels are enabled automatically by
the existing LFM conversion adapter; no worker-count change is required.

## Retained changes

- K=3 cost history now fits in 32 KiB of threadgroup memory. The pin reuses a
  dead cost slot; a barrier protects it from being overwritten by the next pass.
- K>=3 no longer allocates an unused device cost buffer. K=2 retains the
  reference device-cost path.
- Search keeps at most two asynchronous launches in flight instead of retaining
  all scratch buffers for the entire input matrix. At the default per-launch
  budget this is about 512 MiB of scratch, plus inputs and results. The CPU can
  enqueue the next batch while Metal finishes the previous one.
- The converter retains up to 512 MiB of reusable allocator cache between LDLQ
  row buffers, flushing excess cache at the same existing checkpoints. This is
  a cache-retention threshold, **not a cap on total process memory**.

In the first round below, codebook lookup tables, register lookup caches, alternate threadgroup sizes,
vectorized candidates, compact backpointers and pruning were also tested.
They did not provide a sufficiently consistent improvement to enable them then.
The second round combines a different access pattern with an exact half lookup;
its results are reported separately below.

## Reproduce

The optional converter checkout is `references/PonyExl3`, based on
`8e7fa6b1556f59fc669e25087903b279b9b0346f`, with the existing local LFM adapter
changes. It is not a dependency of normal CLI/Desktop inference.

The bounded-cache patch is already applied locally. On an equivalent new
checkout, apply it once:

```sh
git -C references/PonyExl3 apply ../../scripts/patches/ponyexl3-bounded-conversion-cache.patch
```

The LFM conversion script enables MLXL3's search automatically. Use its
`--reference-metal-search` switch to use the original search implementation.
That switch does not revert the separately applied allocator-cache patch.

```sh
.venv/bin/python -m pytest -q tests/test_quantize_metal.py
.venv/bin/python benchmarks/bench_quantize_metal.py --repeats 7
.venv/bin/python benchmarks/bench_quantize_metal.py --layers --repeats 3
```

The last command requires the existing LFM source weights, conversion plan and
2048-row calibration capture under `build/lfm26-quants/4bpw`. It converts only
three representative layers in memory. It does not overwrite a checkpoint,
register a model, publish anything, or start the four-variant conversion batch.

Paired runs alternate their order, check SHA-256 of packed weights/scales and
reconstructed outputs, and report median wall time and active MLX/Metal peak.
That peak excludes NumPy/CPU allocations and MLX's free allocator cache; it is
not whole-app RAM or a separate dedicated VRAM measurement on Apple Silicon.

Tests cover K=2..8 and all three codebooks, including saturated/zero tiles,
chunking, validation errors and cancellation. Equality is byte-for-byte, not
only a tolerance on quantization error. No additional KL/PPL claim is made.

## Search measurements — 2026-09-06

Apple M5, 24 GiB unified memory, MLX 0.32.2, battery power. Seven paired runs,
MCG codebook, seeded normal inputs; both kernels compiled before measurement.
Each row includes dispatch, synchronization and output materialization, not
just GPU execution. Heating makes these timings unsuitable for comparison
with measurements from an earlier battery/temperature state.

| K / tiles | Reference | Optimized | Speedup | Active Metal peak, before → after |
|---|---:|---:|---:|---:|
| 3 / 128 | 34.27 ms | 32.25 ms | 1.063× | 542 → 537 MB |
| 3 / 672 | 181.32 ms | 165.56 ms | 1.095× | 2,843 → 539 MB |
| 4 / 128 | 32.68 ms | 32.46 ms | 1.007× | 271 → 269 MB |
| 4 / 672 | 147.98 ms | 140.99 ms | 1.050× | 1,424 → 539 MB |
| 5 / 128 | 29.08 ms | 27.81 ms | 1.046× | 136 → 135 MB |
| 5 / 672 | 135.10 ms | 136.00 ms | 0.993× | 714 → 539 MB |
| 8 / 128 | 23.51 ms | 23.47 ms | 1.002× | 17.37 → 17.24 MB |
| 8 / 672 | 119.32 ms | 121.34 ms | 0.983× | 91.18 → 90.49 MB |

Not every shape became faster in this first round; it did not establish a K=8
speed gain. All compared outputs matched byte-for-byte. Raw local results:
`build/quantize-metal-search-async.jsonl`.

## Full LFM layer conversions

Final implementation, same Mac on battery, three alternating pairs per layer.
Each includes basis/global-scale search, the full 2048-row Hessian, LDL,
feedback search, packing and metrics. The synthetic search and full-layer
measurements ran sequentially, not concurrently. The reference also restores
the original unconditional allocator-cache flushes.

| Layer (layer 0, K=4) | Reference | Optimized | Speedup | Active Metal peak, before → after |
|---|---:|---:|---:|---:|
| `conv.out_proj` | 5.042 s | 4.608 s | 1.094× | 0.814 → 0.538 GB |
| `feed_forward.w1` | 23.252 s | 22.982 s | 1.012× | 3.193 → 0.933 GB |
| `feed_forward.w2` | 31.337 s | 29.461 s | 1.064× | 2.959 → 1.097 GB |

Packed weights, both scale tensors and reconstructed calibration outputs had
identical SHA-256 checksums in every pair. The ~1% difference on `w1` should be
treated as effectively unchanged speed, not a robust performance claim. Its
main benefit is the ~71% reduction in active Metal peak. The sum of these three
medians improves by about 4.3% in elapsed time, but these three layers are not
a complete-model benchmark or a prediction of full conversion duration.

Raw results with individual runs, checksums and device information:
`build/quantize-metal-layers-final.jsonl`. The earlier serial-batch experiment
is retained separately in `build/quantize-metal-layers.jsonl`; it is not the
default implementation. Previously published model weights remain unchanged.

## Second round — 2026-09-07

These are conversion optimizations, automatically used by `scripts/quantize_lfm26.py`.
No inference kernels, GUI, quantization recipe, calibration rows, feedback size,
regularization, search tolerance or model weights were changed. No publishing or
full-model conversion was performed during these tests.

### Retained kernels

- **K=4, DEFAULT/MCG:** one thread handles 16 aligned states sharing the same
  predecessor. Eight half2 lookup loads use a **128 KiB predecoded codebook**.
  These codebooks already round their decoded values to half; storing those
  values in FP16 introduces **no additional rounding**. A test compares all
  65,536 entries byte-for-byte after a half→float round trip. Error computation
  and winner comparison remain float32; predecessor registers use exact ushort2.
- **K=5, DEFAULT/MCG:** paired decode and shared predecessor reads, retaining
  procedural codebook decoding. MUL1 keeps the original path at K=4/5.
- **K=7/8:** exact distance-ordered search with a predecessor-cost lower bound.
  Sorting is done once per K/codebook. Remaining candidates are skipped only
  when their lower bound is strictly greater than the current best; equal-cost
  candidates still use the original lowest-predecessor tie-break. The initial
  step and two tail-biting passes are unchanged.
- Earlier K=3 shared-memory, bounded scratch and allocator-cache changes remain.

The successful half lookup is **not** the earlier float lookup experiment:
the smaller table and 16-state access pattern must be considered together.
See the original [EXL3 quantizer](https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/modules/quant/exl3_lib/quantize.py)
for context. We do not adopt CUDA's half error arithmetic.

### Cumulative four-projection K=4 comparison

Same Apple M5 / 24 GiB / MLX 0.32.2, now on AC power. Three alternating pairs.
Each run measures `conv.out_proj`, `feed_forward.w1`, `feed_forward.w2`, and
`conv.in_proj` from LFM layer 0, including computed scales, the full 2048-row
Hessian, LDLQ feedback, packing and quality metrics. No model output is written.

| Pipeline | Median wall time | Maximum active Metal memory |
|---|---:|---:|
| Original search and unconditional row-cache flush, 1 worker | 66.354 s | 2.960 GB |
| Current kernels and bounded cache, 3 workers | 41.331 s | 2.823 GB |

**37.71% less wall time**, with identical SHA-256 hashes for packed weights,
both scale vectors and reconstructed calibration outputs in every comparison.
This is an end-to-end **four-projection measurement-stage benchmark**, not a
timed conversion of all 166 projections. The memory column excludes CPU/NumPy
and free allocator cache; it is not total app RAM. The existing worker memory
gate remains enabled. Worker defaults have not been increased.

```sh
.venv/bin/python benchmarks/bench_quantize_workers.py --compare-original --workers 1 3 --repeats 3
.venv/bin/python benchmarks/bench_quantize_workers.py --compare-previous --workers 1 1 --repeats 3
```

The second command isolates the new kernels from worker-count changes and keeps
the **previous MLXL3** batching/cache optimizations in its baseline. Raw logs:
`build/quantize-cumulative-lut-k4.jsonl` and `build/quantize-vs-previous-k4.jsonl`.

That single-worker comparison completed three alternating pairs:

| Pipeline, 1 worker on both sides | Median wall time |
|---|---:|
| Previous MLXL3, including bounded scratch/cache | 63.202 s |
| Current MLXL3 with the new kernels | 43.711 s |

**30.84% less time**, with identical packed weights, scales and reconstructed
outputs across all six runs. Active Metal peak remains approximately 1.098 GB;
the decoded codebook and half lookup add only about 384 KiB of retained arrays.
This establishes the improvement without increasing parallelism or its memory
requirements. The paused batch script and worker defaults are unchanged.

### Final search API measurements

Seven alternating pairs per row, including dispatch and materialization:

| K / tiles | Original search | Current search | Speedup |
|---|---:|---:|---:|
| 3 / 672 | 151.788 ms | 137.801 ms | 1.102× |
| 4 / 128 | 27.562 ms | 18.433 ms | 1.495× |
| 4 / 672 | 125.566 ms | 83.838 ms | 1.498× |
| 5 / 672 | 117.145 ms | 106.474 ms | 1.100× |
| 7 / 672 | 108.491 ms | 54.363 ms | 1.996× |
| 8 / 672 | 106.511 ms | 30.258 ms | 3.520× |

These figures use the final production API, not just prototype kernel launches.
All outputs match exactly. Raw log: `build/quantize-search-final.jsonl`.

### Full-projection K=8 comparisons

Three alternating pairs per projection, one worker on both sides:

| Projection | Original | Ordered search + bounded cache | Speedup |
|---|---:|---:|---:|
| `conv.out_proj` | 3.801 s | 1.314 s | 2.894× |
| `feed_forward.w1` | 17.601 s | 5.693 s | 3.091× |
| `feed_forward.w2` | 24.333 s | 11.133 s | 2.186× |

Every packed-weight, scale and reconstructed-output checksum matches. These
results do not count toward the K=4 reduction above. Reproduce with
`benchmarks/bench_quantize_metal.py --layers --bits 8 --repeats 3`.
Raw log: `build/quantize-layers-ordered-k8.jsonl`.

### Rejected experiments and measurement discipline

- A one-off 3-worker pilot showed −20.3%, but repeated tests did not confirm it;
  it was not accepted as a gain. The new kernel gain is measured separately.
- Smaller scratch batches, extra asynchronous fences removal, a native bounded
  allocator instead of manual cache checkpoints, larger vectors, shared-memory
  swizzles, traceback swizzles, reduced-register product generation, compact
  fresh-state storage and loop/threadgroup hints did not yield consistent gains
  beyond the retained kernels.
- Nearest-candidate and ordered pruning at K=4 were slower. The ordered search
  is only enabled for K=7/8.
- Relaxed Metal math failed the exact-output gate even on normal random input.
  **Safe math mode is retained.** No lower-quality arithmetic was accepted.

Discarded prototypes remain recoverable locally under
`build/quantization-research/`; they are not imported by the converter. The
retained benchmark scripts and tests exercise production code.

Final checks: **48 MLXL3/adapter tests and 36 upstream Metal/Hessian tests pass**
with the optimized search installed (84 total). Gates include all three
codebooks, K=2..8, chunking, LDLQ feedback, zero/saturated/non-finite inputs,
invalid arguments, cancellation and exhaustive half-codebook round-trip checks.
Ruff passes for the changed production, test and retained benchmark files.

Measurement follows the local `inference-engineering` skill: fixed workload and
quality contract, isolated alternating A/B runs, medians, exact-output gates,
and a clear distinction between kernel microbenchmarks, complete projections
and full-model wall time. Timed search spans are host wall time around GPU work,
not hardware GPU timestamps. No new KL or perplexity claim is needed or made.

## Third round — broader bit-width coverage, 2026-09-07

This section supersedes the historical kernel descriptions above where noted.
The target of **20 minutes (or 10 minutes) for a complete LFM conversion has
not been validated**. The previously discussed ~27 minutes was an extrapolation
for the measurement stage, not a timed full-model conversion. No checkpoint was
regenerated in this round.

### Retained changes

- K=2/3/5/6 now share predecessor reads and an exact half codebook lookup,
  grouping 4/8/4/4 states respectively. All three codebooks are supported.
  Exhaustive half→float round-trip checks now cover MUL1 too. Error arithmetic
  remains float32 with the original winner/tie rules.
- K=2/3/5/6 store only the fresh predecessor label in an unsigned byte and
  reconstruct the remaining bits during traceback. A distinct sentinel retains
  the reference's default predecessor zero, including saturated/non-finite
  paths. This halves **backpointer scratch per tile**, not total application RAM.
  Exported indices and the EXL3 checkpoint format are unchanged.
- K=4 DEFAULT/MCG groups independent half2 table reads ahead of arithmetic.
  This is source-level read scheduling, not a hardware prefetch intrinsic.
  MUL1 retains its reference K=4 kernel; K=7/8 retain the previous exact ordered
  search. No speculative decoding or approximate search is involved.
- The existing two-launch fence, 256 MiB per-launch scratch budget, bounded
  allocator cache, calibration recipe and worker defaults remain unchanged.

### Four complete projection measurements

Apple M5, 24 GiB unified memory, MLX 0.32.2, Python 3.12.14, AC power.
Each run converts layer 0 `conv.out_proj`, `feed_forward.w1`,
`feed_forward.w2`, and `conv.in_proj`, including scale search, the full
2048-row Hessian, LDL, LDLQ feedback, packing and metrics. Both sides use
one worker and the same local dependencies/caches. Runs are sequential, with
alternating A/B order. The baseline is the saved second-round implementation,
not the original PonyExl3 converter.

| K | Paired runs | Baseline | New | Less wall time |
|---|---:|---:|---:|---:|
| 2 | 3 (medians) | 114.357 s | 66.091 s | 42.2% |
| 3 | 1 (preliminary) | 74.503 s | 55.805 s | 25.1% |
| 4 | 3 (medians) | 44.052 s | 42.563 s | 3.38% |
| 5 | 1 (preliminary) | 57.673 s | 44.877 s | 22.2% |
| 6 | 1 (preliminary) | 64.669 s | 43.385 s | 32.9% |

Every pair matches SHA-256 of packed weights, both scale tensors and
reconstructed calibration outputs for all four projections. K=3/5/6 were
also checked with repeated search microbenchmarks, but their complete-projection
numbers above remain single-pair observations, not robust median claims.

Individual total wall times (seconds), retained here without requiring local logs:

| K | Baseline runs | New runs |
|---|---|---|
| 2 | 108.645390, 114.356772, 114.560378 | 65.818863, 66.090721, 68.580831 |
| 4 | 43.920192, 44.051950, 44.781204 | 42.562622, 42.527148, 43.569207 |

K=2 maximum active Metal memory across these runs was 1.579 → 1.368 GB;
the first pair alone showed a small increase. This is not a universal total-RAM
reduction. CPU allocations and free allocator cache are excluded.

Raw local fingerprints/timings: `build/quantize-round3-layers-k{2,3,4,5,6}.jsonl`.
The trusted local baseline snapshot is
`build/quantization-research/quantize_round2.py`; neither these generated logs
nor prototype sources are required by production or included in the repository.
The retained benchmark can compare a supplied baseline source snapshot:

```sh
.venv/bin/python benchmarks/bench_quantize_workers.py --bits 2 --workers 1 1 --baseline-module build/quantization-research/quantize_round2.py --repeats 3
```

Without that historical snapshot, `--compare-original` compares the published
implementation against the pinned upstream converter instead; it is a different
baseline. `--compare-previous` is a K=4-only older-kernel comparison, not an
alias for the full second-round snapshot.

### Scope and discarded trials

The search helper is architecture-independent and supports K=2..8, including
2-bpw linear weights. It is automatically installed by the LFM adapter; other
architecture converters must explicitly integrate it and validate their own
tensor discovery/export. This does not establish support or a fixed percentage
gain for every model. Mixed-BPW recipes benefit according to the K values and
projection shapes they actually use. Inference/decode kernels are unaffected.

Register-resident full lookups, single-row cost storage, K=4 compact
backpointers, branch pruning, survivor-path convergence checks, double-buffered
lookup pipelines, alternate layouts/threadgroups and balanced batch sizes did
not consistently beat the retained implementation. K=7/8 lookup changes and
triangular LDL block slicing gave insufficient end-to-end evidence to retain.
Unrun prototypes are not presented as tested improvements. No relaxed math,
smaller calibration set or reduced-quality quantization was accepted.

The implementation reuses PonyExl3's source generator and existing conversion
hooks rather than introducing a second converter framework (the local
`ponytail` skill). Benchmark discipline follows `inference-engineering` as
described above: exact-output gates and separate reporting for microbenchmarks,
complete projections and full-model time.

Final third-round checks: **103 local tests and 36 upstream Metal/Hessian
tests pass (139 total)** with the final source, including the K=4 read scheduling.
The local set covers quantization, the LFM adapter, checkpoint perplexity math,
CLI behavior and the reference codec. The upstream set runs with MLXL3's search
installed. Ruff and shell syntax checks pass for the published helpers.

## CPU preparation and measurement diagnostics

Four independent converter patches reduce work outside the trellis search:

- Candidate measurement using `output_rel_rms` omits the unused Hessian proxy,
  avoiding the two CPU matrix products for `H @ error` and `H @ reference`.
  Output reconstruction and the selected score are unchanged. Omitted proxy
  fields are `null` in measurement records. Other scores and oracle comparisons
  retain the proxy; the layer API also retains it by default. The measurement
  API's `full_metrics=True` restores the complete diagnostics. A reuse-cache
  entry or resumed record missing required output/proxy/oracle diagnostics is
  recomputed, and the cache entry is upgraded without changing its
  weight-determining key. Selecting a different available score on resume
  refreshes each record's score; requesting an uncomputed metric in the offline
  plan optimizer raises an error instead of substituting another metric.
- LDL normalizes its private Cholesky buffer in place. Each column block is
  independent of the later diagonal blocks, and the input Hessian remains
  untouched. This removes one float32 `N x N` copy: 462,422,016 bytes at
  `N=10752`. `block_ldl(..., copy_cholesky=True)` restores the historical copy.
- Zero Hessian shrinkage reuses the already-computed off-diagonal diagnostic:
  both statistics describe the same matrix. Nonzero shrinkage still computes
  both diagnostics. This removes one full-matrix square/reduction, without
  changing Hessian preparation, LDL or the reported values.
- Global scale search requests reconstruction without the packed trellis it
  discards. Its search, reconstructed sample, every score and selected scale
  remain unchanged. `quantize_inner_matrix_direct` keeps packed output by
  default; the new `return_packed=False` option is used only by scale search.
  Round-trip verification still packs and checks when explicitly requested.

These patches are applied to the local converter checkout. On a new checkout,
apply them after the bounded-cache patch above:

```sh
git -C references/PonyExl3 apply ../../scripts/patches/ponyexl3-measure-output-only.patch
git -C references/PonyExl3 apply ../../scripts/patches/ponyexl3-ldl-inplace.patch
git -C references/PonyExl3 apply ../../scripts/patches/ponyexl3-hessian-diagnostic-reuse.patch
git -C references/PonyExl3 apply ../../scripts/patches/ponyexl3-gss-reconstruction-only.patch
.venv/bin/python -m pytest -q tests/test_quantize_measure_metrics.py
.venv/bin/python benchmarks/bench_quantize_workers.py --compare-measure-metrics --workers 1 1 --profile --repeats 3
.venv/bin/python benchmarks/bench_quantize_workers.py --compare-ldl-copy --workers 1 1 --profile --repeats 3
.venv/bin/python benchmarks/bench_quantize_workers.py --compare-hessian-diagnostics --workers 1 1 --profile --repeats 3
.venv/bin/python benchmarks/bench_quantize_workers.py --compare-gss-packing --workers 1 1 --profile --repeats 3
.venv/bin/python benchmarks/bench_quantize_workers.py --compare-preparation --workers 1 1 --profile --repeats 3
```

The CPU tests use a small deterministic substitute for trellis search and run
the actual LDLQ feedback, metrics, plan selection, cache and resume paths. The
LDL gate covers C/F layouts, retries and absence of aliases with caller-owned
Hessians. The preparation gates compare zero/nonzero-shrinkage diagnostics and
every GSS score/selected scale, including a NumPy substitute for Metal arrays.
These CPU tests do not establish Metal output parity. The separate
four-projection A/B commands check packed weights, both scale tensors,
reconstructed calibration outputs and exact candidate scores. The diagnostics
comparison uses in-place LDL on both sides; the other preparation comparisons
retain full diagnostics on both sides. Each comparison changes only its named
optimization and rejects simultaneous comparison modes. The combined
`--compare-preparation` command instead compares all four historical paths
(full diagnostics, Cholesky copy, duplicate Hessian diagnostic and GSS packing)
against all four optimizations, in three alternating pairs with the command
above. The same exact-output and score gates apply. No new speed claim follows
from the removed work alone.

Measured on Apple M5, MLX 0.32.2, with four LFM2.5-2.6B projections at K=4:
the combined comparison took 43.336 → 42.381 s, 43.389 → 42.030 s, and
43.673 → 42.599 s in three alternating pairs. Median paired time reduction:
**2.46%**, with identical packed weights, scales, reconstructed outputs and
scores. Raw local evidence: `build/generic-quant-preparation.jsonl`. This is a
measurement-stage sample, not an extrapolated full-model conversion time.
