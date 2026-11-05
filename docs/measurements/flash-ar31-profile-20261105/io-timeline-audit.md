# AR31: SSD, command ownership and production timeline audit

2026-11-05. Source-only follow-up to the `AR31-DEEP-PROFILE` preregistration in `opti.md`. Inspected the explicit AR30 root worktree, HEAD `afb9fde3137a5c968ad54d0a4feeb7bcff414cee` **plus its existing frozen working-tree changes**, as described by AR30 `freeze.json`. No model, trace capture, benchmark, build, test or prover was launched for this audit. No executable code was changed. Line references below describe that working tree; the final AR31 capture must pin the exact source and binary again.

The retained path is accepted 13 defaults ON, AR20's twelve options and AR30's four options OFF, direct Metal, plain AR, four readers, depth zero. AR30's stable 32-token samples were 10.4093 and 10.2530 tok/s; these are previous measurements, not an AR31 result. The native8 intrusive ~191.8 ms/row cannot be apportioned into the roughly 96–98 ms/token production budget by scaling its categories.

## Main findings

1. **Release has no raw expert-read timeline.** `Completed.timing`, the per-MoE epoch, `ReadBatch.trace` and serialized SSD events are all `cfg(test)`. Existing worker intervals omit queue residence, submit preparation, channel delivery and cache publication. The accepted single-miss split stores only one aggregate start/end for two physical read jobs.
2. **A wait is not a GPU execution, and a layer is not a command.** A central command can contain HC, gate and attention work. Route consumers then wait on that already completed command twice. MoE commands may be consumed by a later layer before their producer handle is joined. The same producer is also retained in `FlashMoeRow.last` for reuse/drain. Command IDs and dependency edges are necessary before assigning time to layers.
3. **The clocks cannot currently be overlaid.** Main profile spans use one TLS `Instant` epoch, SSD events another epoch per MoE call, and command GPU start/end are recorded in floating-point host seconds without a mapping. The root span named `native.ssd_timeline` is created *after* encoding and final submission; its own start is not the MoE epoch.
4. **Expert bytes omit other production I/O.** Embedding reads and PLE n-gram misses are synchronous on the inference thread. They are inside the end-to-end row, but outside the expert worker trace. All are logical file reads, not measured NAND traffic.

## Production boundaries and dependencies

| Boundary | Source | What is really complete / what to record |
|---|---|---|
| Request to generation | `direct_bridge.rs:207–263`; `main.rs:3210–3307` | `started` precedes rendering, tokenization, admission and prefill. Record these separately from decode, plus request ID, budget, prefix reuse and actual path flags. |
| Token selection and visible output | `main.rs:3323–3405` | Select from prior logits, check EOS, set TTFT, append token, incrementally decode and emit deltas **before** the next forward. The first output uses prefill logits; the final output has no subsequent forward. |
| Next forward | `main.rs:3406–3460` | With all new options OFF, `model.forward(next)` runs exactly while `generated.len() < budget`. The retained 32-output request has 31 decode forwards. Speculative branches and C03 must remain disabled and marked absent. |
| Decode accounting end | `main.rs:3464–3532` | Full token decode, splitter/filter tail and visible output occur before `elapsed` is read. `decode_seconds = elapsed - ttft` therefore includes post-TTFT output work and final tail, not only 31 forwards. Completion event serialization itself occurs later. |
| Output backpressure | `main.rs:3013–3018,3133–3148` | Every unbuffered delta takes stdout, serializes JSON, writes newline and flushes. Record duration/bytes, not text, so a slow bridge reader is distinguishable from model work. Preserve the benchmark's existing reader behavior. |
| Baseline selection | `direct_bridge.rs:15–38`; `gpu/flash.rs:687–724` | `log_probs` uploads 248,320 Half logits, runs canonical GPU Half LSE in a central CB, reads 2 bytes, performs rounded Half subtraction into a CPU F32 vector, then CPU greedy. It is not a raw argmax and is not a head-only GPU time. |
| Token embedding | `gpu/flash.rs:364–389` | `TensorInfo::source`, 5,120-byte synchronous `read_exact_at`, BF16→Half conversion, expansion into F32 streams, upload. Record file-read and CPU preparation separately, before layer zero. |
| Layer / deferred boundary | `gpu/flash.rs:467–532` | 48 logical layer calls; a pending HC write may be encoded with the **next** GDN. Keep producer-layer and consumer-layer fields on the CB; do not assign all work to the layer active at completion. |
| PLE row misses | `gpu/ple.rs:101–121`; `flash_ngrams.rs:115–146` | Up to 16 row lookups per PLE call; each miss performs a synchronous 82-byte read and decoding/cache work. Record hit/miss and logical bytes, plus one aggregate CPU scope; do not count hits as SSD reads. |
| Head and host logits | `gpu/flash.rs:640–678`; `gpu/projection.rs:76–96` | Snapshot/forward transaction, mixer, head encoding, asynchronous head submit, explicit completion and 496,640-byte host copy. Snapshot/restore costs and head preparation are distinct from head GPU execution. |

`scripts/benchmark-flash-bridge.py:208–239` exercises this resident JSONL bridge with T=0, top-k=1, penalty=1, MCP false and prefix reuse false. The diagnostic native8 test calls the model directly, not these decoder/output/request boundaries.

## SSD service: schedule, ownership and missing timestamps

The demand service is lazy and shared by all native MoE layers (`gpu.rs:428–440`). Its default has four named `native-ssd-N` workers and four bounded `sync_channel(10)` queues (`ssd_reads.rs:529–603`). Each submitted *work item*, including each half of a split record, advances round-robin assignment (`906–919`). `try_send` fails rather than blocking on a full queue. The previously trialed dynamic common queue is opt-in and is not this baseline; it uses one channel and a mutex held only while receiving (`566–586`).

The relevant production dependency chain is:

| Step | Source | Required fields / missing distinction |
|---|---|---|
| Route ready on CPU | `gpu/moe.rs:284–290` | Producer CB ID, route readback copy, canonical ten IDs. `indices()` already calls `completed_scores()`, and the next call repeats the wait. |
| Retire prior records | `gpu/moe.rs:294–340,733–748` | Accepted record reuse drains prior commands, then drops old slot views before destinations can be reused. Record reuse-retirement begin/end and the retired CB IDs, not a second GPU execution. |
| Reserve misses | `gpu/moe.rs:318–340` | Layer/cache identity, lease generation, canonical route position, hit/miss, old/new backing ID. An expert/slot pair alone is not globally unique. |
| Validate descriptors | `ssd_reads.rs:759–795`; `checkpoint.rs:96–109` | `source()` performs metadata validation; expert extent is checked with another metadata call, followed by FD cloning (or optional B11 file acquisition). This is main-thread work before enqueue. Keep immutable source identity/offset/length, never infer I/O start from route publication. |
| Obtain destinations | `ssd_reads.rs:697–741` | Shared backing reuse or new Metal allocation occurs before `start_payloads`. Record allocation/reuse and backing ID. For legacy byte payloads, reserve/resize happens instead on the worker (`192–195`). |
| Submit / enqueue | `ssd_reads.rs:821–904,906–919` | Batch/job/part IDs, target worker, enqueue-attempt and success/failure times, logical bytes, requested offset. Deadline starts before item enqueue, not when worker starts. A partially enqueued split must remain an incomplete/error batch. |
| Worker read | `ssd_reads.rs:164–225` | Dequeue, destination preparation, file-read begin/end, result, completion send, ticket release. Current test timing starts before test gates/allocation and ends before channel send; it is not pure syscall duration. |
| Split read | `ssd_reads.rs:323–414,870–893` | Two disjoint part IDs/ranges share one logical record/backing. Current timing is first part's start→last part's completion, losing queue skew, each part's read time and the aggregate mutex work. Do not count two logical records or assume simultaneous start. |
| Wait for first wave | `gpu/moe.rs:491–550`; `ssd_reads.rs:964–994` | Hot/shared CB is committed first. `ready(false)` blocks for at least one completion then takes all immediately available completions. Record wait begin/end **and actual returned cohort**, not a nominal fixed wave size. |
| Publish and encode cohort | `gpu/moe.rs:501–550` | Receive→`completed_record`→view reconstruction→generation check/publish→cohort encode. Shared payloads are passed through; legacy payloads upload/copy (`gpu.rs:407–425`). Read completion is not yet GPU-consumable publication. |
| Wait for remainder | `gpu/moe.rs:552–586` | First cold cohort is committed while remaining reads run; normal baseline `ready(true)` then waits for all remaining records. AR20 extra-wave/`ready_min(2)` is OFF. |
| Final mix | `gpu/moe.rs:588–657` | A final command includes the last cold cohort and canonical mix, or all work if entirely hot. Mark which read-publication IDs and prior commands it consumes. No cold-read children on an all-hit call is valid and must be explicit. |

`Read::run` owns `File`, destination and active ticket until the read finishes. Shared destinations are CPU-writable only after the old GPU commands completed and old projection views were dropped. The split's `Arc<SplitRecord>` retains the full destination while two jobs write disjoint ranges. Tracing must not clone or recycle those resources early, or delay ticket release behind trace serialization.

Current raw timings (`Completed.timing`, `SplitState.started`, `ReadBatch.trace`, MoE timeline) are all **test-only** (`ssd_reads.rs:126–140,249–257,935–941`; `gpu/moe.rs:274–275,629–663`). `ReadBatch::poll` does not append its completion timing, so optional prefetch is not fully represented even by this trace. Depth zero creates no prefetch pool (`gpu/flash.rs:265–272`); if a future capture enables it, its separately owned service and all `PrefetchPool::submit/poll/drain` publication/discard events need separate namespace (`gpu/prefetch.rs:25–164`). Do not silently enable it to obtain a timeline.

Successful `read_exact_at` bytes measure requested logical file traffic. Latency includes scheduling, page faults, page-cache behavior and copies; these events cannot identify physical SSD service or NAND bytes. Keeping the API whole records one `read_exact_at`, not its possible short-read/EINTR retry syscalls. Correlate OS file/disk traces if available; do not replace this API with a custom loop just to expose syscall internals in the first low-overhead capture.

## Complete command submission/wait inventory

There is one inference `Metal.queue`, created at `gpu.rs:877`. Source sites below all use it; external/test queues are not evidence of production parallel queues. A CB can contain many encoders, so a composite interval is not a kernel interval.

| Submit path | Commit and wait / owner | Old native8 coverage and release implication |
|---|---|---|
| `Metal::command` | `gpu.rs:1063–1091`: new CB, encode closure, commit, helper wait, terminal status check | Test central encode/wait plus GPU interval. CB creation/autorelease-pool cost precedes encode span, commit precedes wait span. Used by HC/GDN/QSA/PLE/mixer/LSE/export and optional greedy; record semantic dispatches inside it. |
| `Metal::dispatch` | `gpu.rs:1114–1141`: new CB, encode, commit, helper wait, status | No corresponding `native_gpu_interval` call. Public codec/prepare/QMV APIs at `1200–1320`; no retained Flash row caller identified. Instrument centrally so unexpected use is visible; require zero or explain actual hits. |
| Projection `submit_encoded` | `gpu/projection.rs:895–907`: commit, return owned CB/destination. `ProjectionOutput::read`, `76–96`, waits and copies | Test projection wait/GPU interval. Head is an actual retained caller. No new wait in tracing; collect timestamps after its existing read completion. |
| Standalone `FlashGate::dispatch` | `gpu/gate.rs:126–135`: commit, return `RouteOutput` owning CB | No independent GPU interval at its wait. Retained depth-zero joint HC/gate path should use central helper instead; assert path counts, rather than assume one gate CB per layer. |
| Standalone `Metal::route` | `gpu/router.rs:56–64`: commit, owned `RouteOutput`; `23–29` direct wait | Same missing GPU interval. `indices()` calls completed wait, then caller often asks scores again. Record unique CB once; both CPU waits remain valid separate operations. |
| MoE hot, intermediate and final | `gpu/moe.rs:429,495–498,558–579,638–649`: commits into `self.last`; clones retained by `MoeOutput` | Test GPU interval collected in `MoeOutput::completed`, `61–89`, not at commit. `drain_commands`, `733–748`, waits again but does not log a new interval. Owner is both cache retirement and consumer join. |
| Conversion-only queues | `gpu/quantize.rs:11–26,66–76`; `gpu/convert_matmul.rs:178–232` | Separate devices/queues for conversion APIs; absent from plain AR by caller inspection. Namespace if ever active, otherwise exclude from AR totals. |

`RouteOutput::completed_scores` and `ProjectionOutput::read` call `wait_until_completed` directly, bypassing `Metal::wait_command`; central/MoE/drain use the helper. B10 spin remains OFF. Instrumenting only the helper would miss two real wait families; instrumenting only commit sites would miss CPU copies/readiness consumption.

`Metal::encode_bound` (`gpu.rs:1161–1181`) creates one compute encoder, binds buffers and parameter bytes, dispatches, ends it. This is the common point for optional static kernel IDs, CB ID, geometry and input/output backing ranges. CPU encoding duration is not GPU execution time. Pipeline pointer/name lookup should be precomputed outside the row. Trace aliases/ranges only for relevant produced/consumed views, not by adding thousands of owned buffer clones.

### Cross-layer consumers and joins

In the retained chain, `LayerResidual::forward_after_attention_read` invokes `read_after_write_with_gate` (`gpu/gdn.rs:663–677`; `gpu/hc.rs:571–603`). A single central CB writes the previous attention residual, reads MLP HC and generates routing. It has already completed when routing reaches `forward_routed`; repeated route waits are then normally negligible, not another router kernel.

Accepted deferred MoE completion can return `LayerOutput::Pending`. The next regular GDN's `forward_pending` encodes the preceding HC write and current HC/GDN in one central CB (`gpu/gdn.rs:808–850`). Only afterward does `write.finish_producer()` join the original MoE producer (`110–118`). Same-queue submission plus input-buffer dependencies allow the consumer to be submitted without an early CPU fence. `DeferredHcWrite::Drop` still joins an abandoned producer (`122–126`). Preserve this order: adding a timing wait at layer end would destroy the overlap being measured.

## What old native8 can and cannot establish

`gpu/flash.rs:751–811` starts `profile::start(false)` only after prefill and first selection, records seven direct model-forward+selection rows, and separately validates outputs/state. It omits production request preparation, tokenizer, JSONL writes/backpressure, outer generation completion and prefix admission. Its spans and allocation accounting are useful structural evidence, not delivered bridge throughput.

The observer has TLS `Trace`, nested stack, dynamically growing `Vec<Event>`, owned `String` names and `serde_json::Value` (`flash_profile.rs:12–29,76–79,210–232`). Each allocation request increments test TLS counters and may call observer timing (`gpu.rs:52–86`). Thus observer work can itself change allocation, cache and CPU/GPU pacing. The native test path does not call the MLX-only `retain_test_stages` consumers found in `qwen4.rs`; that knob is **not an established explanation** for this native8/release gap.

The corrected AR30 analyzer (`profile.py.txt:16–70`) validates ancestry and exclusive allocation totals, and rejects duplicate `(command_address,start,end)` triples. Its reported host categories remain **inclusive**: parent layer/head/central scopes overlap child encode/wait/copy scopes. The GPU totals are composites grouped by ancestry, not per-kernel time or utilization. A raw command address is not a lifetime-stable ID: addresses may be reused and completed commands can have multiple owning/waiting wrappers. Keep the previous analyzer as a diagnostic, not as the critical-path accounting implementation.

## Minimal instrumentation proposal if tools cannot correlate the path

This is a design for a separately preregistered implementation, **not implemented or measured here**.

1. One process-wide monotonic epoch shared by inference and reader threads; numeric, typed records with static event kinds in preallocated per-thread bounded buffers. No per-event JSON/String allocation, formatting, locking global event collection, resource clone, or disk/stdout write. Dump after the request and cleanup. Include observer enabled state, capacities, event counts and overhead comparison. IDs are monotonically assigned without reuse within a run; overflow is an explicit incomplete outcome.
2. IDs: `run/request/token/phase/layer/moe_call/read_service/batch/job/part/queue/command/backing`. Carry lease slot/expert/**generation**, byte offset/length and canonical route position. Identify producer and consumer layers separately. Token IDs must distinguish prefill, first selection, 31 decode forwards and final output, so no off-by-one denominator.
3. SSD records at source validation begin/end, destination acquire/reuse, enqueue attempt/success/failure, dequeue, read begin/end, completion send, consumer receive, lease publish/discard and cohort encode/commit. Keep aggregate and split-part records; aggregate readiness depends on both parts. For normal `read_exact_at`, record whole API outcome rather than claim syscall counts. Record active jobs/queue assignment/high-water and returned cohort membership.
4. Command records at creation, CPU encode begin/end, semantic dispatch, commit and existing wait begin/end with caller/reason, plus terminal status/error. Give each new CB one logical ID and queue ID at creation; raw pointers are optional diagnostics. GPU timestamp extraction happens once after existing completion, not once per owner/wait. Commands failing or dropped before commit are explicit terminal states. A completion handler, if needed by tools, must only append a bounded record; it cannot block or retain model state unexpectedly.
5. Buffer readiness records bind producer command/read-publication IDs to consumer commands and CPU readbacks. Track views by backing ID plus range/offset/type; do not treat aliases as different allocations. `MatrixView::read_output` (`gpu/view.rs:79–102`) assumes the caller already completed its producer; it does not wait itself. An exporter/CPU copy must therefore carry its explicit producer edge.
6. Separate main CPU **wall intervals** from actual CPU running time. A span around `recv_timeout`, `wait_until_completed`, file read or stdout flush includes blocking/scheduling. Without scheduler samples, label it host wall time, never CPU busy/utilization. Optional allocator/request counters should be a separate level if per-request timing disturbs pacing.

### Clocks and interval accounting

All thread records can use offsets from the **same** process `Instant`, copied into workers. Existing local MoE offsets cannot be mapped by assuming their enclosing trace span began at zero. Cross-process comparisons use independent duration scales, not a shared `Instant` origin. The earlier campaign's invalid cross-process monotonic subtraction is not reusable calibration.

Current GPU values (`flash_profile.rs:334–340`) explicitly lack CPU-epoch mapping. Preserve raw timestamps and clock labels. Only place GPU execution on the CPU lane after the capture tool or an independently verified Metal/host calibration provides anchor pairs and an error bound; record calibration drift/resolution. Do not assume seconds-to-nanoseconds conversion aligns clocks. If unavailable, report separate CPU and GPU lanes and causal edges, with no numerical cross-clock overlap percentage.

Per token, construct a DAG: CPU sequence; per-worker queued job sequence; successful enqueue→dequeue→read→completion→receive→publish; producer buffers→GPU consumers; command creation/encode→commit→GPU completion→CPU readback; queue ordering and actual joins; output flush→next forward. A cohort is ready at its **latest** required publication; a GPU consumer cannot execute before its producer/queue dependencies. Compute the longest dependent path over measured non-overlapping nodes, retaining unexplained scheduling gaps. Do not add read latency, CPU wait for that read, enclosing MoE time and GPU execution as independent costs.

Report separately: end-to-end decode duration; main-thread exclusive scope coverage; worker wall intervals and queue latency; distinct-command GPU interval union; kernel counters when available; CPU/SSD/GPU overlap only with calibrated common clocks; unexplained/uncertain time. GPU interval union is command coverage, **not device utilization** or proof every cycle was active. Inclusive scopes and child sums are diagnostic views, never an additive time budget.

## Completeness, backpressure and failure obligations

- Saturated trace buffers increment per-producer dropped counts and record first overflow; any drop, unmatched begin/end, missing ID, timestamp inversion, unfinished batch/CB or unavailable clock mapping degrades the report explicitly. No partial report may assert complete attribution or parity.
- All-hit/no-split/no-prefetch cases must record explicit zeros and path bits. Enqueue failures/channel disconnection, truncated source, allocation failure, stale generation, partial split submission, one-part failure, deadline, GPU error, cancellation/EOS and stdout failure keep their real outcomes and partial evidence. A failed command is not a zero-duration success.
- `ReadBatch::Drop` receives only until its existing deadline (`ssd_reads.rs:997–1009`); `ReadService::Drop` joins workers only when its active count is zero (`923–933`). `ensure_reads_idle` is an assertion, **not** a drain (`gpu/memory.rs:77–83`). Trace closure must distinguish logical batch drop, physical completion and joined/detached workers. No claim that a deadline cancels `pread`, and no attempt to reuse a still-owned destination.
- Keep output draining and bounded controller shutdown/join identical to baseline; otherwise trace output itself can stall the bridge pipe. Write trace artifacts outside the timed token loop. Trace writing cannot replace or suppress the production complete/error event.
- Before a speed/critical-path claim, use independent numeric state/logit/golden and actual-path checks, plus observer OFF/ON identical production requests. If typed instrumentation is added, regression cases should cover wrap/overflow, unique IDs, nested/exclusive interval accounting, split aggregation, repeated waits deduplicated by CB, partial failure and incomplete cleanup. Attempt Kani on actual bounded ID/buffer/completeness state logic; such a CPU check would not prove Metal, I/O, scheduling or performance.

## Decision

Existing source evidence is enough to locate low-cost markers and to reject naive addition of native8 categories. It is insufficient to reconstruct the retained production critical path, physical SSD utilization or GPU utilization. First try tool tracing against the unchanged production CLI; add the narrow typed SSD/command timeline only for correlation data unavailable there. Requalify observer overhead and numerical behavior before interpreting milliseconds. No optimization or new speed result is proposed by this source-only audit.
