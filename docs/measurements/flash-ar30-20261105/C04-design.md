# C04 resident scratch: bounded private slot source prototype

Original design audit on574e4a8c with C02 source50ba935c layered above it.
Root owns issue135, preregistration and the common qualification. The design
below was written before C04 implementation; the dated source handoff at the
end records what is implemented and what remains unverified. No C04 compiler,
prover, test or GPU execution was performed in this worktree.

## Small safe boundary

Use `FlashProjection::encode` / `encode_with_scratch` (`projection.rs:451–477`),
not the generic `encode_expert_stage` as the cache admission boundary. That
wrapper passes an empty `ExpertStage`, and the result is only a fresh destination
Buffer. Prepared and partial buffers are borrowed internally, encoded through
finish, and never returned or inserted into MatrixView/TensorStorage, snapshots
or user-visible structures. The public `ProjectionOutput` contains only command,
destination Buffer and count (`:51–55`).

Admission for the first prototype must require all of:

- Flag ON and exact qualified M5; true resident, `!generic`, `!mtp_hidden`,
  `!routed_record`; input batch exactly1; validated fixed extents.
- Private destination storage. Shared/public/trace calls keep current fresh
  behavior; output is always fresh even on a scratch hit.
- This wrapper's own default stage and no caller-provided scratch. Do not admit
  externally prepared inputs, destinations or pending-finish paths.

The `partial.clone()` that really escapes is
`encode_ready_cohort_impl`→`PendingExpertFinish` (`projection.rs:1553–1567`).
It is a separate routed-expert/axis path and must stay excluded. The existing
`FlashExpert::encode_into` invokes `encode_expert_stage` directly (`:1820–1889`),
so it also bypasses the proposed wrapper cache, including shared experts. This
is a simpler, stronger boundary than trying to inspect Buffer retain counts.

## One slot per projection; reserve before encoding

Field: `RefCell<Option<ResidentScratch>>`, where the entry owns prepared,
partial, exact byte sizes and the latest owning CommandBuffer. No global
buffer pool, no raw Buffer/Rc uniqueness inference.

1. Validate current command owner and unsubmitted state, input/shape/device and
   required sizes before admission. `try_borrow_mut` failure falls back fresh.
2. Existing entry is reusable only if its owning command is **Completed**, it
   differs from the current command, lengths match exactly, both Buffers are on
   the same device, and prepared/partial are distinct allocations.
3. Replace the entry's owner by a clone of the **current** command before any
   kernel can be encoded. Then borrow its pair for the existing implementation.
   Any second encode against the same unsubmitted command sees an occupied
   pending entry and uses a fresh pair. Never replace/evict that pending entry.
4. Empty slot: reserve budget, allocate both buffers before encoding anything,
   then publish the pair with the current owner. A failure before publication
   may undo its unused budget charge. A failure after publication leaves a
   non-reusable pending/error entry and uses fresh fallback on later requests.
5. No waits, callbacks, blocking allocation retry or early state publication
   are added. Failed/unsubmitted commands can conservatively pin one slot until
   projection drop; that is bounded and honest, with a counter, rather than a
   claim that they are completed. Outputs/frames/KV/prefix ownership is unchanged.

Current Metal command buffers retain tracked resources. Completion alone was
not sufficient for arbitrary MatrixView aliases in the original arena idea;
it becomes sufficient for overwrite only together with the audited private
no-escape boundary above and updating the owner before encoding. A future caller
that returns prepared/partial must bypass the cache or carry a real lease.

## Concrete size bound and admission reserve

For batch1, bytes are `2*input + 4*output*splits`; destinations are excluded.
`reference_splits` (`flash_support.rs:647`) makes6144→2560 split2 and the listed
2560-input projections split1. These are source-derived logical sizes, not RAM
or an allocator measurement; no checkpoint payload was read.

| Current fixed family | Prepared+partial bytes per layer | Layers | Total bytes |
|---|---:|---:|---:|
| GDN qkv2560→10240, z2560→6144, out6144→2560 |108544|36|3907584|
| QSA q2560→12288, k/v2560→512, out6144→2560 |101376|12|1216512|
| QSA index_qk2560→640, configured at context4096 |7680|12|92160|
| **Total** | | |**5216256 (4.975 MiB)**|

That is168 populated slots/336 cached buffers and at most336 fewer fresh
prepared/partial requests per warmed decode row in this exact scope. Some
branches may miss or bypass; actual hit counts must establish the reached path.
The existing vocabulary head uses Shared output and is excluded; admitting a
future private2560→248320 head would add998400bytes. Shared experts, routed
cohorts and recurrent frames are excluded, so do not attribute all2092.7 layer
allocations or the115MB frame payload to this feature.

A32 MiB hard cache-admission cap is ample, but must be real. A minimal central
budget counter may conservatively charge each newly published cache pair until
Metal/context teardown, without refunding published charges. This bounds all
admitted slots even if projections are dropped while commands still retain
resources, and avoids completion callbacks. It can exhaust after repeated model
reloads on the same Metal instance and then falls back fresh; expose that limit
explicitly. Do not call this cumulative charge live allocated RAM. Reclaimable
budgets instead require a completion-owned reservation token and are a larger
design; never refund on a pending entry's ordinary Drop alone.

Charge/reserve the configured32 MiB when optional feature admission is computed
(`gpu/memory.rs::admit_load`) so lazy retained scratch does not bypass request
memory limits. Allocation failures before publication roll back only their
unsubmitted reservation. Budget exhaustion is a fresh fallback, not an error
or a wait for an uncommitted command to release memory.

## Ledger and indispensable future checks

Add cached prepared/partial to `FlashProjection::record_memory` under a separate
`resident_projection_scratch` Backings category; existing Backings deduplicates
by buffer pointer and sums checked lengths. Keep these out of fixed weights and
fixed-residency collection. With the wrapper-only scope, FlashExpert's separate
weight collector has no populated cache slots to miss. Report actual cached
bytes and budget charges separately; counters should distinguish new slot,
completed reuse, pending/same-command fallback, budget fallback and failed slot.

Before enabling or measuring, tests must exercise:

- Original full projection outputs after completed reuse; retained destinations
  and snapshots stay unchanged and destination Buffer identities stay fresh.
- Two encodes in one CB; unsubmitted A plus committed B; in-flight A; completed
  A followed by reserved-but-not-yet-completed B. No illegal pair reuse/eviction.
- Failure before both allocations, after publication/encoding and aborted CB;
  fresh fallback remains correct, no waits, truthful counters and bounded charge.
- Generic/routed/MTP/batched/Shared/trace bypass; exact extents/device/alias
  guards;32 MiB cap, overflow, reserve rollback and ledger no-double-counting.

Use a real CPU state-machine helper for selection/reservation and Kani over
statuses/owner identities/budget/shape bounds. It does not prove Metal completion
or resource retention. GPU tests must use independent canonical projections,
poisoned scratch, real status transitions, actual counters and complete state/
logit/CLI qualification. Root's new intrusive profile reports18.33ms allocation
API overall but only0.35ms for GDN frames; no recovered time or boost follows
from the source counts above.

Estimated implementation scope: roughly100–150 production lines for private
slot selection, conservative budget reserve, field/init and ledger, plus useful
tests/proof harnesses. No shader or arithmetic change is necessary. This is
feasible as a later, separately qualified wave after C01/C02/C03, with the
conservative abandoned-slot and budget behavior documented rather than hidden.

## C02 proof handoff (not executed here)

After root's single common CPU build, the focused command is:

```text
CARGO_TARGET_DIR=<root-private-target> /Users/justin/.cargo/bin/cargo kani --lib --no-default-features --jobs1 --output-format regular --harness ar30_router_policy::paired_router_actual_geometry_preserves_every_reference_output_bound
```

The helper/harness is actual Rust, no assumptions/stubs/loops: symbolic enabled,
two shape u32s, two capability u32s and a symbolic group. It checks exact
qualification, fixed pair output coverage, safe final output offset, and four
reachable covers (batch1/max batch/unsupported width/insufficient TG). Keep
unwinding/safety checks and inspect unreachable/internal/unsupported results.
Register the exact name in the combined Kani manifest first. No Kani or physical
test result is claimed by C02 commit50ba935c; parent integrated it as392ea4be.


## 2026-11-05 source implementation handoff — unqualified

Root explicitly authorized the narrow C04 prototype after the original design
and preregistered its ownership/budget details before code. There is one slot
per resident projection, not a general pool. Production changes are about200
lines including the actual CPU budget/guard helpers and load/request accounting.
No Metal shader, FMA order, Half boundary, output allocation, recurrent frame or
snapshot ownership changes.

The wrapper short-circuits enabled+valid budget before querying command status,
so the OFF path adds no ObjC status query. It then guards nonGeneric/nonMTP/nonRouted,
batch1, Private destination and NotEnqueued command. Only the wrapper's empty
ExpertStage can reach the cache. Pair lengths/device/identity must match exactly;
Completed and a different CB are required for reuse. Owner replacement happens
before any prepare encoding. Pending, error, same-CB, borrow conflicts and cap
exhaustion keep the existing fresh path. Outputs are always fresh. No waits,
callbacks, completion handlers, implicit synchronization or pending eviction.

Budget cap32MiB covers all published slots. Allocation failure before publication
refunds its unused claim; after publication even a host encode failure keeps the
pending owner/charge. A completed entry can be removed/refunded by releaseIdle or
Drop. Pending/error Drop conservatively leaves its charge until Metal teardown.
Refund uses checked subtraction; an invalid charge disables admission/reuse and
is surfaced by ledger/releaseIdle rather than panicking or hiding the error.
Drop refunds the exact stored claim, with no arithmetic. Request and model-load
admission reserve32MiB whenever enabled **or charged>0**, preserving accounting
when a test switches the option OFF after publication. Ledger records actual
cache backings under resident_projection_scratch and conservative charge in
resident_scratch_charge_bytes; a dropped abandoned slot may have no current
backings but a nonzero charge. These quantities are not RSS.

C03 now sends its vocabulary head through Private encode_row. It qualifies for
the same safe no-escape wrapper, adding998400bytes to the original table:
**6214656bytes (5.927MiB),169slots/338buffers** if all historical fixed projection
families and index_qk are reached. This is logical source size, not measured
allocation/RAM/speed. Public Shared head and trace paths still bypass. The
optional feature remains OFF: MLXL3_NATIVE_RESIDENT_SCRATCH=1 is exactM5 only.
The model qualification must toggle gpu.resident_scratch.enabled and observe
its cfg(all(test,macos,direct-metal)) reuses counter for C04 separately.

New actual Rust harness:
resident_scratch_policy::completed_slot_admission_and_reservation_are_bounded.
Symbolic full-u64 used/bytes, seven booleans for admission, three booleans for
ownership selection and full-u16 requested positions at context4096. It checks
actual reserve/eligible/reusable plus Cell-backed Budget claim/refund, disabled
claim immutability, invalid over-refund rejection and real ResourcePlan reserve.
No assumptions, stubs or loops. Six covers include successful claim/refund,
Completed reuse, same-CB refusal, exact-cap admission, full-cap rejection and
position4096. **Not run here**; root owns Kani registry and execution.

New physical test (ignored/exclusive until root's explicit selection):
gpu::projection::resident_scratch_tests::physical_ar30_resident_scratch_preserves_retained_outputs_and_pending_owners.
It loads only resident qkv, captures canonical OFF bytes, then checks eight
complete outputs/81920finite Half words by exact byte equality, with two actual
completed reuses and46080cache bytes. It poisons both temporaries, retains prior
outputs, verifies fresh destination IDs and complete results, performs two
encodes in one unsubmitted CB plus a different completed CB, checks Shared
bypass, rejects wrong-width input before mutation, checks deduplicated ledger
and completed idle refund, then deliberately sets invalid host threadgroups to
cause failure **after** prepare encoding. That failed command is never submitted;
its slot cannot be reused/released, fresh fallback matches the canonical oracle,
and its dropped charge stays explicit. GPU Scheduled/Committed status is not
forced by this small fixture; the same production predicate admits only
Completed. Full model states/continuation/rollback/CLI belong to root's combined
gate. No speed claim or numerical qualification is made from source review.

Checks actually performed: rustfmt (installed absolute path, edition2024,
skip_children=true) for changed source; git diff --check. Both passed. Source
inspection found all FlashProjection initializers and all current release/ledger
callers; standard CI rust.yml already runs PR/main Rust checks and Kani registry.
No build, clippy, unit test, Kani, physical test or benchmark was run here. Root
must register this real harness and inspect all common results before enabling
or measuring C04. Parent opti.md is owned/updated by root; no index/root edits
or push were made here.
