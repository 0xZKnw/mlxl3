# P13 adaptation audit: retained owners and disabled-path overhead

Read-only review under `I120-P03-P13-ADAPTATION-REVIEW-20261104`, comparing base
`aed9a27a` with the source qualified as CLI `e621c430`/libtest `bd2fe763`.
P03 HC workspace/in-place activation was proposed but never implemented or
benchmarked. This audit concerns P13 HC→GDN command chaining. No executable
change, build, proof, inference or new measurement was performed by this audit.

## The intended path executes, but the refactor also affects OFF

`gdn.rs:330–345` calls the actual flag/trace/owner/shape predicate and concatenates
the existing HC and GDN encodings. The physical four-row model fixture records
144actual joins, with36commands/waits removed per row; this includes35GDN
layers and one PLE body. Both F16/F32 component cases, full states/logits and
continuation pass. QSA uses its existing path. There is no evidence that the
qualified model silently bypassed the optimization.

The GPU buffer allocations, shader passes, bindings, grids and arithmetic in
the moved GDN body remain unchanged. However, even flagOFF uses the new
`PendingGdn` (`gdn.rs:46–54,127–128,272–319`). The old `trace.then_some([...])`
eagerly constructed its array and, when false, dropped the eleven existing
views before commit. The new array stays alive until completion. `_previous`
contains two more buffer owners; `_input` and `_out_half` add two. Thus15Rust
buffer-owner releases per body—540per row—move from encoding to after wait.
They are **moved releases, not540new allocations or necessarily540extra calls**.
The array is not a newly allocated GPU payload or heap Vec.

For only these36HC/GDN pairs per row, source-level counts are:

| Item | Base | Candidate OFF | Candidate ON |
|---|---:|---:|---:|
| Outer command buffers/waits/autorelease pools |72|72|36|
| Completion-status queries |72|108|72|
| Additional owned CB retain/release pairs |0|36|36|

The extra status check is `finish`'s publication guard. In ON, it offsets the
removed separate HC status query. Candidate OFF also evaluates the new
eligibility branch. Counts are not timings or an emitted-instruction analysis.

## Retention safety and scheduling limits

Metal-rs0.33.0 `src/commandqueue.rs:33–38` maps `new_command_buffer` to
`commandBuffer`, separately exposing the unretained variant. The installed
SDK27.2 `Metal.framework/Headers/MTLCommandQueue.h:27–34` explicitly says the
standard method maintains strong references to used resources. Every affected
buffer is bound directly by `gpu.rs:762–773`. Releasing the Rust view wrapper
after encoding therefore differs from releasing the underlying resource while
the command still needs it.

`Metal::command` (`gpu.rs:655–681`) wraps encoding, commit and completion in an
autorelease pool. The joint path retains HC temporaries through GDN instead of
draining the separate HC command first. Its larger post-wait release burst can
shift CPU work before the next routing/SSD stage. Conversely, mandatory GDN
CPU encoding moves before the joint GPU start. Neither effect has an isolated
timing. The baseline11.208msHCend→GDNstart interval includes work that moved;
it is not an established recoverable GPU-idle budget.

The HC→MLP/router/SSD continuation (`gdn.rs:401–434`, `moe.rs:230–249`) preserves
its dataflow and demand submission point after attention. No SSD/cache policy
or contribution order changed. The new refactor may change when CPU reaches
that point; its cost is unknown. Joined diagnostics use
`native.hc_gdn.joined`, while the legacy `native.gdn.body` scope remains in
`forward`; their inclusive durations must not be compared as identical scopes.

## Smallest justified next distinction

Quality OFF/ON shares this refactor. Comparing e621ON against stock d419
therefore combines chaining with refactor overhead. MAIN confirmation samples
overlap (reference6.924512/7.056745, chain7.025383/6.906091tok/s; median−0.356%);
the first HOLD screen's+1.197% is a separate short result. Neither attributes a
cost to retained owners.

The newly preregistered d419OFF/e621OFF screen is the appropriate next
ablation: it measures the net disabled refactor/guard cost, not Pending alone.
No result from that campaign is claimed here.

If it motivates a later adaptation, the smallest bounded candidate is to
restore optional trace views and remove redundant previous/input/out-half
Rust holders, relying on the existing retained-resource command contract.
Keep the owned command and Completed publication guard; keep HC outputs
owned and preserve trace export, errors, retained snapshots and rollback.
That is a possible lifetime adaptation requiring its own preregistration and
physical checks, not a proven speed fix. No such edit was made.

## Lean Pending V2: source-only review after its preregistration

The following is a later review under `I120-P13-LEAN-PENDING-20261104`;
V1/e621 and all its measurements remain preserved. V2 restores
`steps: Option<[...;11]>` and the original eager `trace.then_some` in encoding,
removes `_previous/_input/_out_half`, and keeps the owned command, next frame,
values and Completed publication guard. `finish` now consumes the encoded
trace option, avoiding a second boolean that could disagree with it.

No lifetime defect was found in this source review. The standard queue method
sets retainedReferences=true. Apple explicitly includes buffers and pipeline
states among encoded resources and says their references are released after
GPU execution. The GDN helper has already dispatched and ended each encoder
before its Rust temporaries drop. All bindings use the existing direct
setBuffer path; V2 introduces neither unretained commands, argument-buffer
indirection, naked Rust-view pointers nor new unsafe code.
[Apple makeCommandBuffer](https://developer.apple.com/documentation/metal/mtlcommandqueue/makecommandbuffer%28%29),
[retainedReferences](https://developer.apple.com/documentation/metal/mtlcommandbuffer/retainedreferences).

For an encoding error, Metal::command never commits. For an encoded but
NotEnqueued command, finish rejects before state publication; dropping that
unsubmitted work cannot schedule GPU reads. For a submitted command, the
standard retained-resource contract protects the temporary backings. The
owned command remains necessary so its later status query does not borrow an
autoreleased object after its pool ends. TraceSome retains all eleven views
until export, and next/values remain explicitly owned. Layer/PLE rollback is
unchanged.

The existing two physical fixtures remain appropriate unchanged oracles:
F16/F32 untraced outputs, thirteen trace tensors, trace fallback, malformed
encoding, NotEnqueued rejection and exact recovery; then four model rows,
144actual joins, full finite84-state/logit continuations and initial archived
hashes. They must run against the newly built V2 artifacts. Reused CPU policy
Kani results do not prove Metal retention or concurrency.

V1's Rust owners also keep resources alive after Metal releases its encoded
references. Moving the last-owner cleanup back to the command's resource
lifetime is a concrete adaptation, but its thread, latency and throughput
effect are unmeasured. V2 still adds an owned-command retain/release and
completion guard relative to stock; zero refactor overhead does not follow
from the source. No V2 numerical or speed result is claimed by this review.
