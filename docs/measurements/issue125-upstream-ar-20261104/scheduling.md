# Current-engine scheduling audit

Source-only research, 2026-11-04. Local source: published PR129, `63787f1cd137248fe6d1bb482e354ed458395ba9`. Accepted baseline: L1 enabled on exact M5, four new options disabled. No prototype, GPU run, rebuild or new speed measurement. The user explicitly excludes further L3/L5 tests.

## The remaining MoE boundary

`native/src/gpu/gdn.rs:447–463` calls `experts.completed()` before constructing either a deferred or immediate HC write. `native/src/gpu/moe.rs:51–64` waits for every submitted MoE command and checks each status. The final MoE command already encodes the canonical route-ordered reduction, commits at line399 and remains retained in both `self.last` and `MoeOutput.commands`. L1 defers only the subsequent HC write: it cannot overlap preparation of its consumer with this final wait.

The smallest candidate is to carry the already-submitted MoE output and its completion/error ownership into its GPU consumer, prepare/submit that consumer before waiting, and check all producer and consumer statuses before `attention.finish` or any other state publication. Begin with the existing34 eligible L1 boundaries, rather than a general scheduler. This may remove a host synchronization boundary; it does not remove the MoE arithmetic or the wait for demanded SSD records. A separate, more invasive possibility would merge the last MoE reduction submission into its consumer; unlike wait deferral, this could reduce CB count, but encoding the consumer before committing could delay the last cold-expert computation. These are separate hypotheses, not measured gains.

Apple documents automatic resource-conflict synchronization for tracked, directly bound resources on the ordinary `MTLCommandQueue`. Our `gpu.rs:61–83` creates device buffers without a heap/untracked override; `gpu.rs:812–838` directly binds them. This supports investigating a GPU dependency instead of a CPU wait. It does not justify deleting validation, assuming that disjoint CBs cannot overlap, or treating command completion as proof that previous commands succeeded. [Apple resource synchronization](https://developer.apple.com/documentation/metal/resource-synchronization), [blocking wait semantics](https://developer.apple.com/documentation/metal/mtlcommandbuffer/waituntilcompleted()).

Necessary invariants: retain all command/resources and cache leases; check every producer failure, not merely the last status; do not expose mutated GDN state on failure; preserve prefix/whole-token rollback; preserve canonical route reduction; keep trace/component/public readbacks synchronous; do not delay router readback or demanded SSD launch. `MoeOutput::multiplier_trace/read`, `FlashMoeRow::drain_commands`, the previous-row `self.last` checks and `DeferredHcWrite/LayerOutput::complete` are affected boundaries. A future implementation requires its own preregistration and actual numeric/error/CLI qualification.

## Corrected profile anchor

Read-only reanalysis of the published `../issue125-current-engine-20261104/selected-off-quality/native8-profile.json.gz`:7 decode rows,12311 events,238 L1 triples+14 doubles,336 MoE waits,887 completed MoE CB intervals. Full finite independent/continuation oracles passed in that original diagnostic. No profile was rerun for this research.

| Inclusive host span | Events | ms/row |
|---|---:|---:|
| Complete row |7|156.856110|
| Central helper wait |959|67.245000|
| MoE completion wait |336|27.440660|
| Central encoding |959|6.630190|

| Completed GPU composite | Events | Sum of intervals, ms/row |
|---|---:|---:|
| MoE |887|36.550309|
| L1 HC-write/read/GDN |238|18.695417|
| HC-write/read/router |336|9.111006|
| Other central helpers |182|5.654982|
| Projection |7|3.337161|
| Separate HC-read |91|1.890952|
| HC-read/GDN double |14|1.072548|
| Separate HC-write |98|0.083387|

For each MoE wait, selecting its last completed MoE GPU end and the immediately following recorded central GPU start gives336 same-clock gaps: median0.328104ms, p950.739625ms (sorted index `int(.95*(n-1))`), min0.105292ms/max1.984625ms. This covers all48 boundaries/row; the238-boundary L1 subset is not separately attributed here. This is a specific submission boundary observation, **not global GPU idle, CPU/GPU clock alignment, or recoverable time**. GPU intervals overlap; inclusive host spans include GPU work. None of these values can be added into a speed prediction.

## Already present or previously rejected

L1/P13 already batch HC/GDN and HC-write/read/router. GDN and expert projections already expose encode paths; routed expert scratch already shares prepared/partial buffers inside each expert encode. P01 direct SSD-to-shared Metal storage, static EXL3-K specialization and NT2 are already present. A generic recommendation to enable asynchronous execution, scratch reuse, direct reads or prefetch is therefore insufficient. Earlier expert-encoder reuse and P04 greedy remain excluded; L3/L5 will not be retested under this task.
