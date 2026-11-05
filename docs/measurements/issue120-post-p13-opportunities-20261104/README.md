# Flash Next AR: next opportunities after P01/P13

**First choice: defer a final MLP HC-write into the following regular GDN command. Second choice: a bounded progressive cold-expert wave.** These have concrete scheduling opportunities in the existing profile and can reuse current kernels. Kernel fusions remain promising but lack per-kernel attribution. No new optimization or speed gain is validated by this research.

Scope: qualified PR124 source c146e9a4, Flash Next checkpoint65c89531/EXL3 average2.05bpw, Apple M5/24GiB, AR only. No MTP/DFlash, canceled greedy P04 or canceled expert-encoder reuse. The original personal checkout is untouched. No executable source, GPU/model, benchmark, build, proof, installation or publication was changed/launched for this research.

## Evidence used

The latest qualified native8 diagnostic is **after** P01 direct expert reads and the corrected P13 chain, not the old MLX trace. MAIN87/context4096,7decode rows,3GiB/48×53slots,4readers,lookahead0,prefixoff. [Original aggregate](../issue120-hc-gdn-chain-20261104/final-quality/native8-safe-summary.json), full profile SHA a34cd74a04cd805f3bc1cab8e7beaadb5bb19f1bb574d9b327cf9f12bbba0424.

| Existing completed-command group | Count over7rows | Sum of GPU intervals / row |
|---|---:|---:|
| MoE |886|39.130ms|
| HC-read + GDN joined |252|21.855ms|
| Existing HC-write/read/router chain |336|9.889ms|
| Separate attention/head HC-read |91|2.494ms|
| Final separate HC-write |336|0.162ms|
| Head projection |7|3.548ms|

These are intrusive composite command intervals, not individual kernel durations. Command intervals overlap:155/249 Hot→W1 pairs overlap in this very profile. Their sums are not device utilization or additive wall time. Inclusive host waits, allocator APIs and worker-local read intervals are also distinct quantities. No predicted tok/s or recovered milliseconds are calculated from them.

## Ranked candidates

| Priority | Candidate | Concrete opportunity | Main cost/risk |
|---|---|---|---|
| 1 | Final HC-write → next HC-read → GDN |34 regular-GDN boundaries/row can potentially remove34 command buffers and waits, preserving existing P13 | Deferred error publication and input ownership; no shader/math change |
| 2 | One additional progressive cold wave |141/336lots have≥3remaining records; final collection currently waits for all | Extra submission can cost more than hidden tail work; cap one extra CB |
| 3 | HC up→mix fusion |97dispatches and97temporary allocations/row removable structurally | Exact reduction/casts/sigmoid and changed workgroup mapping; occupancy unknown |
| 4 | HC down+inject joint GEMV |96dispatches/row removable, same normalized input |324output guard and view-offset trap; keep two zero-offset outputs initially |
| 5 | Per-HC four-buffer workspace |388allocations/row avoidable after warmup;8,194,560B retained capacity | Lease/trace/error/ledger/idle plumbing; observed API envelope1.172ms/row |

Priorities compare engineering evidence and risk, not promised percentage gains. P08 is the smaller first kernel experiment if low implementation risk matters more than P09's additional materialization removal. These two kernel candidates are not established winners from the current composite timings.

### 1. Next regular GDN consumes the previous final HC-write

The final MLP HC-write still has48 separate CBs/row:0.162ms GPU,0.682ms encode and7.771ms inclusive host wait in the diagnostic. The tiny GPU work and repeated synchronization make this a particularly concrete next boundary. The7.771ms is not a recoverable-time promise.

[The producer](https://github.com/0xZKnw/mlxl3/blob/c146e9a4a351fb159477c50fe83117447dcbed2b/native/src/gpu/gdn.rs#L415) already waits for `experts.completed()` before its final write. [HC-write](https://github.com/0xZKnw/mlxl3/blob/c146e9a4a351fb159477c50fe83117447dcbed2b/native/src/gpu/hc.rs#L424) uses GPU/dtype/layout and three input views, without HC weights. A private pending write could own streams, completed expert branch and gate, then be consumed by the next regular GDN in one **write → HC-read → GDN** CB.

It is essential to preserve the existing read+GDN chain: joining write+read and then putting GDN in another CB would give back that saving. The newly produced `written` view must also be returned and passed to the consumer's own residual injection; passing the producer's older streams would be incorrect.

**The count is34, not35.** [PLE is layer1](https://github.com/0xZKnw/mlxl3/blob/c146e9a4a351fb159477c50fe83117447dcbed2b/native/src/gpu/flash.rs#L212), GDN0 has no predecessor, and twelve QSA consumers remain outside the first scope. Thirty-five next joined-scope pairs in the trace include PLE's prelude; the initial offline grouping was corrected before recommendation and is preserved for transparency. Expected active-path counters are34triples +2existing doubles per row, with separate final writes48→14. Supporting all34 requires deferring the common residual write for eligible producer types; PLE/QSA *consumers* and public component APIs must force completion first.

Keep the pending local to the existing [whole-token/prefill transaction](https://github.com/0xZKnw/mlxl3/blob/c146e9a4a351fb159477c50fe83117447dcbed2b/native/src/gpu/flash.rs#L521), outside snapshots/prefix/last-hidden. Trace/capture paths stay synchronous initially. The activation oracle must therefore run an untraced model forward with nonzero34/2/14 counters; a trace-only comparison could exercise zero deferred writes. Preserve `experts.completed()` and all routing/read-wave code so demanded SSD work is not structurally postponed. Test errors after the producer has advanced, both encode failure and late consumer failure, plus old prefix restoration. The profile's same-clock target gaps are observations, not global GPU idle or independent CPU/GPU alignment. [Detailed grouping](research-cross-layer.json).

### 2. Stop waiting for the entire cold tail before a useful second cohort

There are1,097logical records/7rows,1,369,069,164bytes, and2,263hits among3,360route occurrences. The first ready cohort contains one record in290/301cold batches. After the intermediate W1 submission, [the code](https://github.com/0xZKnw/mlxl3/blob/c146e9a4a351fb159477c50fe83117447dcbed2b/native/src/gpu/moe.rs#L328) uses `ready(true)` and collects the complete remainder.

Pairing all336layer timelines with886GPU CBs succeeds exactly. W1→Final GPU intervals have median gap0.425ms/p951.257ms. This is a target-command interval; it neither proves I/O was exposed nor measures NAND or global idle. Worker times use another CPU epoch. The proposed bounded policy applies to141lots: only when≥3records remain, gather≥2ready records, submit at most one extra cohort, then collect the final tail. `ready(false)` itself blocks for the first result and drains available completions; it is not a nonblocking poll.

Preserve the canonical `plan.routes` mix order, immutable completed records, generations, leases, deadlines and error draining. An extra CB can lose performance, so require actual cohort counters and no added reads, expert work or thread count. A common four-worker queue is a later alternative: only96/336lots contain>4jobs, and the profile lacks enqueue/worker IDs to establish imbalance. Do not bundle both scheduling changes in the first experiment. [SSD analysis](research-ssd.json), [GPU-interval pairing](research-ssd-gpu-gaps.json).

### 3–4. Native HC kernels worth isolating

**Up→mix:**97HCreads/row produce10,240temporary `up` values, then read them for a2,560value mix. A fused workgroup can produce the four stream-specific rows and mix them. Structurally this removes97dispatches,97allocations and7,946,240logical write/read bytes/row in the actual F32 workload. It does not remove the dense up products or weight reads. Remapping workgroups must retain each K320/BN1 FMA and shuffle order, the cast to the source dtype before sigmoid, ordinary `exp` in mix and stream0→3 accumulation with contraction disabled. Current data does not isolate its GPU cost or predict occupancy/spills.

**Down+inject:** both plain F16 matrices use the same normalized input. A single81-group narrow GEMV can cover320down rows plus4injection rows while preserving K10240/BM1/BN8 arithmetic. Initially retain two zero-offset outputs. Simply concatenating to324 rows is insufficient: `flash_hc_dispatch` rejects324, and `hc_activate` ignores view offsets. This removes96dispatches but promises no weight-byte, MAC or allocation reduction in that minimal variant. These are distinct from historical HC22 two-token batching and HC24 activation epilogues. [Kernel and dtype audit](research-kernels.json).

### 5. P03 workspace, not a naive GDN ping-pong

The four HC intermediates (normalized/down/active/up) do not escape as model state; mixed and gate outputs do. A per-instance lease can retain four typed buffers until the enclosing command finishes, use a fresh fallback when busy/dtype/trace requirements differ, and account8.195MB in the ledger and idle cleanup. The profile associates1.172ms/row of allocator API time with these388requests. This is not a maximum possible end-to-end gain, but is less compelling evidence for a large boost than the synchronization targets.

By contrast GDN frames request115.458MB/row but their allocation API envelope is only0.211ms. Those bytes are not a measured copy. Snapshots retain old frames, so two-buffer ping-pong without exclusivity/COW can corrupt prefixes and rollback. [Allocation and lifetime audit](research-alloc.json).

## Secondary or excluded

- Intermediate attention HC-write in Private storage is a tiny internal-mode trial:48mode changes/row, zero allocations removed. Shared write and same-size Private norm have different observed allocation API means, but storage is not an isolated cause. Public readbacks/final hidden/trace must stay valid.
- Shared-expert local scratch is a separate shape currently excluded from routed scratch;192allocations/row might be avoided. Time is not isolated, and gate/up dependencies need checking. Routed scratch is already active, not disabled by native profiling.
- A routed-down output slab can remove432allocations/row without reducing requested bytes, but may add resource hazards across expert waves. The current overlapping Hot/W1 intervals make a blanket slab a poor first experiment.
- Unequal layer misses suggest P07 static budgets, but seven MAIN rows lack initial LFU/prefill history. Use an independent calibration prompt and the existing fixed-total slot interface; do not tune on evaluation MAIN/HOLD. Global cache/ARC and old LES variants are not new candidates.
- Large HC/router/a-b weights are **already F16**, confirmed by five checkpoint headers/40.29MB metadata, zero payload read. There is no giant expanded F32 matrix to compact. Only small norm/gain vectors remain, about2MB potential HC-gain reduction.
- Head input is already passed as a GPU view. GPU greedy P04 and expert encoder reuse were canceled; sparse QSA beyond2052tokens, multirow prefill and MTP/DFlash need different workloads.

## Concrete first experiment, if implementation follows

Start with candidate1 alone, initially flag OFF, same GPU code and allocations. Before speed: actual34/2/14path counters, independent full finite logits/hidden/84states, imposed continuation, deferred-error rollback, retained prefix, trace/capture fallbacks and real CLI golden. Then one MAIN/HOLD AR32 A/B/B/A against frozen final P13 CLI3dfea63e at the same3GiB/4readers/depth0 setup. One GPU model, no compilers/provers/concurrent inference; warmup plus one measured response, exact complete outputs and cleanup. Audit adaptation before judging a disappointing result, including an OFF/refactor ablation if it changes the legacy path.

No new gain, build, executable check or GPU experiment was performed here. [Source/artifact manifest](manifest.json) identifies the existing profile and notes; the original dirty checkout and published PR124 remain unchanged.
