# AR33 T02 / T03 — capability snapshots and dispatch descriptions

2026-11-05. Source baseline `01da8f013c4eeb1a8b1fdeb2ec0fab0e347620d4`
(native qualified `82fff5a5`). [Preregistered matrix](matrix.md), issue138.
Read `opti.md`, [historical C05](../flash-ar30-20261105/research-attention.md#at30-5--cache-immutable-capabilitygeometry-qualification)
and [historical R5](../flash-ar30-20261105/research-runtime.md#r5--validated-cpu-encode-replay-for-stable-native-segments-reserve).
No profiling resumed. Source was prepared before execution; the narrowly granted
CPU proof is recorded below. Unit tests/build/GPU and speed are coordinated by root.

## T02: smallest owner-bound implementation

`native/src/pipeline_caps.rs` owns each PSO through `Pipeline<P>` and retains an
optional immutable `[execution_width, maximum_threads]` snapshot beside it.
No pointer registry, shared cache, new allocation, mutable alias or `DerefMut`.
The production adapter reads the exact two `metal0.33` properties as `u64`;
there is no truncation or normalization in the cache. PSO values must remain
immutable, as they do for the retained Metal pipeline object. Clone retains the
same PSO and copies its snapshot; it does not query again.

OFF: construction performs no capability queries; each getter calls the original
Metal getter. ON: construction queries each property once; getters return the
snapshot. Existing guard expressions, short-circuit order, casts, error messages,
dynamic shape/device/offset checks, fallback decisions and dispatch dimensions
remain at the original caller. Do not reject all pipelines at construction:
that would change errors for an unsupported but unused optional/model path.
The optimization hoists immutable observations, not dynamic validation.

Approved integration scope is intentionally bounded:

| Family / real caller | Existing repeated checks | Integration |
| --- | --- | --- |
| HC `encode_stream_norm_into`, `encode_read_prepared`, `encode_after_write` | SIMD32 and maxima640 / dynamic GEMV threads | Wrapper in `hc::Pipelines`; same expressions. |
| GDN `encode_for_consumer`, `encode_norm_gate` | Dense32/256, step32/64, prepare/norm32; whole-GDN eligibility32/1024 | Wrapper in `gdn::Pipelines`; preserve trace/private/feature gates. |
| PLE `compute` | Route32/640, key32/256, value32/128 each invocation | Wrapper in `ple::Pipelines`; no cache/read/history changes. |
| Generic model ops RMS/dense/GQA/RoPE | SIMD32 and maxima derived from current width or fixed thresholds | Wrapper in `model_ops::Pipelines`; dynamic geometry unchanged. |
| QSA load + lazy NAX + optional AR20 paths | Base geometry at load; chains/prefetch geometry per optional encode | Wrapper in `qsa::Pipelines`; lazy NAX wrapped at actual creation. |
| MoE shared switch / final logsumexp | Switch32/512 per row, LSE32/1024 per normalization | Parent wraps exactly these fields. |

QSA base load checks are **not** recurring per-token savings. AR20 chains and
prefetch are OFF in the retained baseline; their checks are covered but cannot
justify a speed claim. Projection QMV/NT2/dense validation is mostly at load and
remains raw; router dot/softmax, optional mapped/finish-mix/greedy, converter and
other raw pipelines remain outside this narrow implementation. No claim of
eliminating every Metal property call or of a measured1600 queries/token.
`MTLDevice.maxBufferLength`, device identity, runtime availability, storage and
allocation bounds are untouched.

Parent wiring (central files deliberately not edited by this agent):

1. Register `pipeline_caps` with mac/direct OR test OR kani, like other CPU policies.
2. Define `CachedPipeline = crate::pipeline_caps::Pipeline<metal::ComputePipelineState>`.
3. Parse `MLXL3_NATIVE_CACHED_CAPS` strictly0/1, defaultOFF, once per `Metal`.
4. Keep the raw loader and raw `Metal::encode` API; add `cached_p(name)` mapping
   successful raw creation through `Pipeline::new(raw, cached_caps_enabled)`.
5. Pass `cached_p` to the five family loaders; use it for both shared switches
   and logsumexp. Store the bool for lazy QSA NAX construction. Raw encoding
   receives the same underlying PSO via immutable dereference.

Test-only thread-local `pipeline_caps::stats()` returns cumulative
`[cached_width, cached_maximum, live_width, live_maximum]`; construction ON counts
its two real observations as live. Capture deltas **after** model loading;
ON-row deltas must show nonzero cached reads and zero live reads **within this
wrapper domain**. OFF must show live reads and no cached reads. Release builds
contain no counter code. Root must qualify the actual wired path and retain
independent state/logit/continuation/CLI checks before MAIN32 speed.

Four prepared deterministic tests cover OFF laziness, per-access queries, ON/clone
single observation and constructor/access counter deltas, SIMD31/32/64, zero, threshold639/640 and full
`u64::MAX` values. The actual constructor/getter Kani harness is
`pipeline_caps::proofs::cached_pipeline_preserves_full_width_and_dynamic_capability_guards`:
symbolic enable, full64-bit width/maximum and required width/threads; no assumptions,
loops or GPU stubs. It checks exact observations and acceptance/rejection, with
five covers. This is a CPU model of immutable properties, not a proof of Metal,
FFI immutability, the final module wiring, concurrency or numerical generation.

### CPU proof receipt, 2026-11-05

Under root's CPU grant only, Kani0.68.0/CBMC6.11.0 ran the exact capsule harness
and existing `flash_support::hc_kernel_flags_are_off_by_default_and_require_exact_m5_literals`
with `--lib --no-default-features -Z stubbing --exact --jobs1 --output-format regular`
in the separate `build/ar33/kani-target`. No user stub/assumption was supplied.
Both pass in20.26s: capsule106SUCCESS/5satisfied covers; flag240obligations =
233SUCCESS+7UNREACHABLE std internals,6satisfied covers; total346obligations,
zero failures,11covers. Flag domain is symbolic bytes0..4/valid UTF-8 and
M5/presence booleans, unwind10. Capsule domain is full64-bit inputs, no loops.
Cratewide unsupported caller_location/foreign-function warnings were retained;
neither selected harness reaches a failing unsupported construct. The seven
inaccessible checks concern std alignment/UTF-8/format internals, no business
assertion or cover. All owned processes joined; no timeout or GPU execution.
[Full command/versions/source hashes/domains and per-check review](cpu/kani-caps-flags.json),
[complete log](cpu/kani-caps-flags.log). The unit tests remain delegated to root;
this CPU proof does not qualify native numerical correctness or a speed gain.

## T03 structural gate: no dispatch-description prototype

The smallest resident EXL3 segment is prepare → QMV → finish
(`projection.rs::encode_rows` → `encode` → `encode_expert_stage`; `gpu.rs::encode_bound`).
Every dispatch already receives stack arrays of buffer references and inline
parameters; `set_bytes` avoids a parameter allocation. There is no CPU command
record heap allocation in this helper to remove. A description would still have
to bind freshly allocated prepared/partial/output buffers, offsets, PSO and
parameters, create/end the same encoders, and dispatch the same grids. Retained
scratch C04 is OFF and does not permit borrowing a previous frame's outputs.

Whole GDN is already one dispatch after resident projections, with freshly owned
history/state/output buffers; HC uses optional fused paths and workspace leases.
QSA key count/slab growth/sparse selection changes per row. A larger replay must
validate current owners/leases/extents/offsets/features **before the first emit**,
and must preserve the original error and commit boundaries. Expert backings and
cohorts are dynamic; recording them revives the excluded expert/P04 domain.

Fixed PSO and grid metadata can be cached, but most of its preparation is a few
constant integer operations/stack values. T02 already removes the demonstrated
immutable property reads. A new descriptor here would add binding records and
invalidation handling without a demonstrated reduction in Metal calls,
allocations or dispatches. **T03 gate is negative for this minimal port: no
prototype/model/speed test proposed.** This is a source-level opportunity
decision, not a measured claim that replay is always slow. No encoder sharing,
command-buffer recommit, numerical reassociation or CPU-ID scheduling change.
