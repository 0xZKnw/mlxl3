# AR31 observer/path audit — source only, 2026-11-05

Read the AR31 preregistration in `opti.md`, AR30 closed trials, native8 test,
`flash_profile.rs`, native dispatch/wait/readback/allocation sites, reader
timings, direct bridge and generation-stat boundaries before this audit.
No executable edit, model/weight read, capture, test, compiler, prover or new
measurement was performed. Only this note is written; root owns the protocol.

**Conclusion:** native8's191.814ms mean row is not the wall-time budget of the
qualified production CLI's10.409/10.253tok/s samples (approximately96.1–97.5ms
per decode token). Source proves multiple observer and protocol differences,
but does not quantify their contributions or explain the entire discrepancy.
Do not subtract the two observations and label the remainder “observer cost”.
`StageRetention` is specifically **not a native-path explanation**.

## What the historical observation actually measures

The retained AR30 profile used libtest `4577d8cd…`, seven observed decode rows,
12,870 host events, and the qualified AR10 defaults; its
[report](../flash-ar30-20261105/profile/report.json) and
[summary](../flash-ar30-20261105/profile/summary.json) preserve the exact artifact,
conditions and limitations. The later10.4 baseline is release CLI `f24201e3…`,
new OFF/source freeze in [AR30 freeze](../flash-ar30-20261105/freeze.json), with
32generated/31decode and the actual bridge/stat path. These are different
processes/binary configurations/row populations and noncontemporaneous samples.

`gpu/flash.rs:751–814` profiles `model.forward(input)` followed by full host
`log_probs` and canonical greedy selection. Prefill and first selection occur
before `profile::start(false)`. Each row additionally validates every248320
input Half logit for finiteness. The complete event Vec is serialized after
`finish`, outside the row intervals. Independent state/terminal digests and
`forward_trace(7)` quality checks occur outside the observed loop; they are not
191.8ms per-row work. Their prior cache/memory history can still differ from the
CLI run, so equality of outputs does not establish identical performance state.

## Proven differences, with causal limits

| Site | What source establishes | What it does not establish |
|---|---|---|
| `flash_profile.rs:78–103`; all `retain_test_stages` callers | Thread-local StageRetention defaults true, but its consumers are MLX `qwen4.rs`; native `gpu/*` and native8 never call it | It cannot account for native8 keeping all MLX intermediates or explain a native2× slowdown |
| `flash_profile.rs:193–230,359–382` | `start(false)` creates unreserved Vecs. Every active span owns a String, pushes Event/stack entries, calls the clock at start/end and borrows TLS RefCell state. There are about1839events/observed row | No measured per-span overhead or proof that allocation alone accounts for the gap |
| `gpu.rs:64–91` | Every libtest allocation increments test counters, even with trace inactive. Active tracing additionally reads clock/TLS and attributes bytes to the innermost event | Release CLI omits these cfg(test) operations; a libtest trace-OFF run is not a release baseline |
| Same allocation wrapper | Its elapsed interval starts before `max_buffer_length`, validation/newBuffer and counter updates; it is not an isolated newBuffer syscall duration | The reported18.33ms allocation API wall time is neither pure allocator CPU time nor recoverable release time |
| `gpu/moe.rs:274–275,368–370,494–495,557–558,631–638` | cfg(test) builds Instant/Vec/serde_json timelines even when TRACE is inactive. JSON construction is before hot/wave/final command commits; read timings are expanded into JSON before final commit | This may alter when ready reads are collected/cohorts formed, not merely add a fixed reporting surcharge. The amount and changed wave geometry are unmeasured |
| `gpu.rs:1064–1088`; `gpu/moe.rs:63–81`; `projection.rs:77–94` | cfg(test) counters/spans and post-completion GPU timestamp queries are absent from release. Timestamp event metadata builds JSON on the main thread | Historical native.gpu_interval self duration is only about0.448ms/row; this particular observed scope alone is not a96ms explanation |
| `gpu/gdn.rs:506–529` | cfg(test) additionally retains a clone of the gated MatrixView in PendingGdn | It adds host/resource retention; no extra GPU norm/conv kernel or trace materialization follows from this clone |
| `gpu/flash.rs:779–813` vs `main.rs:3330–3533` | native8 records seven model+selection rows. Production stats include decoder/splitter/filter/output work and final tail processing; native8 does not | The CLI's extra bridge work cannot explain why the old native8 observation is slower, and early7-row averages need not equal31-row cache/routing averages |

Only active span creation grows the trace Vec; the outer JSON conversion/print,
ancestry validation, digest subprocesses and terminal state checks are outside
the timed rows. Do not misattribute their total test runtime to decode.
The per-MoE JSON and span metadata above **are inside** the observed rows.
Test worker start/end timestamps and `ReadBatch.trace.push` are also compiled
only under cfg(test), at `ssd_reads.rs:179–222,934–990`.
The trace's Rust String/Vec/serde heap allocations are not Metal `buffer()`
requests and therefore are absent from the3639.7request/row count. That count
cannot bound the observer's total CPU heap allocation cost.

## Kernel/path equivalence checked in source

There is no profile-active branch selecting a different native math kernel.
The active observer does not set an attention `trace` flag. native8 calls
ordinary `model.forward`; `FlashNext::hidden(...,None,None)` keeps the accepted
cross-layer deferral and passes ordinary non-traced layer forwards.
The historical receipt verifies238cross-layer and14joined GDN paths over seven
rows, with zero separate GDN bodies. This is positive evidence against blaming
a wholesale trace-mode GDN fallback in this diagnostic.

Low-level GDN trace uses Shared intermediates and disables whole-GDN/private
fusions (`gdn.rs:208–238,285–289,376–404`); low-level QSA trace similarly selects
Shared storage (`qsa.rs:508–518`). Those are real alternative paths, but native8
does not request them during observed rows. The high-level `forward_trace`
quality helper captures layer outputs/state and disables cross-layer deferral;
it must stay outside a production timing window. Do not replace it by
`forward_trace` per measured row to obtain more detail.

All accepted kernels are demanded by native8/default-policy assertions. New
AR20/AR30 options remain OFF in the proposed10.4 baseline. Source cannot prove
identical instruction scheduling/register pressure between cfg(test) and a
release binary or rule out system/cache/IO variation. The observed timing is
therefore unsuitable as a release budget even though its numerical path passes.

## Why the old timeline cannot assign every release millisecond

`flash_profile.rs:337–343` stores completed command-buffer GPUStartTime/end
alongside CPU Instant-relative spans, explicitly with **no clock mapping**.
The [historical analyzer](../flash-ar30-20261105/profile.py.txt) groups GPU
duration by ancestry and rejects duplicate `(address,start,end)` observations.
It sums durations, not a clipped GPU interval union or a dependency critical
path. GPU/CPU components and enclosing host scopes overlap; their sum cannot
be191.8ms or the release96ms budget. Native central wait includes sleeping,
scheduling, GPU service and time after prior submission; it is not CPU compute.

Coverage is central helpers, standalone projection/head and MoE completions,
not every dispatch or hardware utilization. `Metal::encode_bound`
(`gpu.rs:1160–1182`) creates/binds/dispatches/ends each compute encoder, but
records no kernel timestamps or stable dispatch IDs. Whole GDN and HC chains
are fused composites and cannot be separated into conv/norm/etc kernel time by
renaming a span. The source cannot expose time *inside* a fused kernel without
hardware/shader profiling. Cold reads report worker-wall intervals, not NAND
traffic; current timestamps omit a complete enqueue/dequeue/CPU-ready timeline.

## Smallest useful production observer

Prefer one release CLI observer in the real bridge flow, OFF and ON in the
**same built binary**. Do not promote the test profiler wholesale into release:
that would inherit JSON and unconditional test bookkeeping. No model/kernel,
command-buffer boundary, producer wait, capture mode or sampling semantics
should change to make measurement convenient.

1. Typed fixed-size records: numeric event-kind/row/layer/CB/dispatch/read IDs,
   parent ID, start/end ticks, and a few integer payloads. Preallocate records
   and stack before measured decode; intern static names outside it. No String,
   serde Value, formatting, source dumps or file writes on hot paths. Store
   counts separately where individual events are unnecessary. Overflow marks
   coverage incomplete with explicit dropped counts; do not grow/reallocate.
2. Main-thread scopes first: full request/prefill, token decode boundary,
   hidden/layer, route-readback, cache plan/admission, CPU encode, commit, wait,
   head/readback, log_probs/selection, tokenizer/filter/emit/tail. Preserve the
   preceding cross-layer producer ID: the first command in a GDN consumer can
   also execute the prior layer's HC write. One command must not be charged in
   full to both layers.
3. At existing dispatches attach static numeric kernel family/PSO identity,
   dimensions/NT/splits, layer and CB ID, without per-dispatch timestamp queries
   or added encoder/barriers. This gives geometry/count coverage and CPU encode
   cost; it does **not** give GPU kernel durations. Label once per existing CB
   or PSO outside the timed loop when an external trace needs readable names.
4. Record unique monotonically increasing CB IDs at actual create/commit/wait
   sites. Cover central commands, standalone projection submit/read, router
   submit, MoE hot/cold/final and their drains. Read GPUStartTime/end once after
   an already-required completion; do not add waits, scheduled-handler fences
   or new command buffers just to read timestamps. Retaining commands until
   end-of-request would change lifetimes: extract scalars at normal completion.
5. Worker rings need the same read ID and row/layer/lease-generation lineage:
   enqueue, worker dequeue, read syscall begin/end, completion enqueue, consumer
   ready/collection. Reuse the existing job boundary/owned destination contract.
   Attribute host queue/worker service/consumer wait separately; call it read
   syscall wall time, not SSD hardware latency. Independent per-thread bounded
   storage avoids a new global contended logging lock.

This first coarse mode can explain production host waiting/encoding/gaps and
GPU **command** union before a more intrusive per-kernel pass. Allocation needs
only per-scope counts/bytes initially; if allocation-wall timing is requested,
bracket exactly the allocation API and report the observer overhead separately.
Enable the timer branch only for an active requested capture; OFF must avoid
clock reads, ObjC timestamp queries and per-record allocation.

## Clocks, exclusive budgets and deeper tools

Use one explicit macOS host clock for CPU events and GPU host timestamps. Apple
documents GPUStartTime/GPUEndTime as host seconds relative to system mach time,
available only after completion; [GPUStartTime documentation](https://developer.apple.com/documentation/metal/mtlcommandbuffer/gpustarttime).
Recording raw mach ticks plus timebase once avoids assuming Rust Instant's
private epoch equals those seconds. Keep origin/conversion precision/error in
the report. Bracket a calibration if clocks are mixed; never align by row index
or absolute Python monotonic values from separate processes.

Raw hardware counter ticks require separate mapping. Apple's
[sampleTimestamps](https://developer.apple.com/documentation/metal/mtldevice/sampletimestamps())
and [conversion guidance](https://developer.apple.com/documentation/metal/converting-gpu-timestamps-into-cpu-time)
describe paired CPU/GPU samples before/after a counter interval, warning that
sampling may trap to the kernel. Sample sparingly at capture boundaries, not
at every dispatch. These raw GPU ticks differ from the already-host-time
command-buffer properties; do not apply the wrong conversion twice.

For exact kernel/pass GPU time and hardware bottlenecks, prefer a bounded
external Metal System Trace/Time Profiler pass before writing per-dispatch
timestamp-buffer machinery. Apple documents parallel CPU/GPU/resource tracks
and optional utilization/limiter counters in
[Metal workflows](https://developer.apple.com/documentation/xcode/metal-developer-workflows)
and [performance analysis](https://developer.apple.com/documentation/xcode/analyzing-the-performance-of-your-metal-app).
Availability/templates/permission and overhead on this Mac remain root's
capability check, not a result of this audit. Counter sampling or shader replay
can serialize/replay work; mark a separate intrusive capture, never silently
replace the production throughput observation. GPU busy intervals alone do not
prove occupancy, arithmetic utilization, DRAM bandwidth or cache behavior.

Report two coordinated views, not a sum of CPU+GPU+IO durations:

- Main-thread wall partition: exclusive child unions per row with encode,
  allocation, routing/selection/readback, emit and blocked/unattributed gaps.
  Wall intervals include descheduling; CPU-running time requires scheduler
  tracks or a separately identified thread CPU clock.
- GPU/read workers: clipped interval unions and dependency edges to the actual
  main-thread wait/completion. Preserve useful work overlapping CPU/IO. Sum
  mutually exclusive row-wall buckets to measured row time; show GPU/IO overlap
  separately and any uncovered time explicitly. A critical-path claim needs
  submit→execution→completion and demanded-read dependencies, not ancestry alone.

The authoritative production denominator remains `main.rs:3501–3533`:
`decode_seconds = elapsed - first_token_time`, tokens=`generated.len()-1`.
TTFT is recorded before first decoder/output emission (`main.rs:3372`), while
elapsed is after final decode/tail/filter processing. For32generated, retain
31forward+next-selection transitions, with first-token emit and final tail as
separate boundary buckets. Nested model row spans should not silently omit
that CLI overhead or change the published stats denominator. Prefill, warmup,
serialization, final observer flush and shutdown need explicit boundaries.

## Required qualification before interpreting a new profile

Root must preregister the actual bounded capture matrix before any run. Reuse
existing independent component/model/CLI oracles where functions are unchanged;
verify finite/nonempty events, unchanged golden tokens/text, actual default
flags and full state/continuation for an executable observer change. CPU tests
and Kani should cover the real bounded recorder/stack/overflow/tick arithmetic;
that would not prove Metal scheduling or hardware counters.

Compare fresh OFF/ON production windows with the same model/context/prompt,
warmup/cache/readers and31decode denominator; record row distributions and
route/cache/miss/cohort/CB counts so schedule perturbation is visible. Show
observer overhead/drift, do not retrospectively scale191.8ms into96ms. A profile
with dropped events, missing commands, invalid clocks or changed path is
partial/diagnostic, not a complete per-millisecond attribution. File writes and
formatting happen outside the timed window; failures persist false/incomplete
status and bounded children are joined. No new optimization follows until
those contemporary measurements identify the actual production bottleneck.
