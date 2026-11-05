# AR37 / AR38 — source plan before the precise implementation grant

2026-11-05. Read `opti.md` AR35–AR38 and L1/P13/AR20/AR34 histories first.
Baseline PR141 `10397e9899eb5ec6a19e6b6f6af179500c6d3770`; T05 and thirteen
historical defaults ON. [Issue140](https://github.com/0xZKnw/mlxl3/issues/140).
This note contains **no prototype, test, proof, build, model or benchmark**.
AR3/AR4 require root's precise preregistration and implementation grant.

## Common API agreed with the AR2 owner

AR2 lives in the separate `issue140-ar2-20261105` worktree; do not redevelop its
QSA split in the AR1 worktree. The agreed non-generic interfaces are:

- `prepare_append(&self) -> PreparedQsa`: select/own slab and perform existing
  COW/growth copies outside the joined CB. Their commands are not removed.
- `encode_for_consumer(&self, cmd, x, prepared) -> PendingQsa`: encode index and
  attention, valid untraced when index is absent OR post-append length<=2051.
- `finish(&mut self, pending)`: require its own CB Completed and matching source
  owner/offset before publishing tail, slab, offset and high-water mark.

`PendingQsa` owns its CB and intermediates; abandonment drains submitted work,
never an uncommitted CB. The HC `ReadBuffers` lease is retained alongside it
until completion. No generic pending graph, replay or shared compute encoder.
AR2's own optimization requires an indexer; AR3 can also join HC when absent.
At length2052 with an indexer, sparse CPU selection keeps the original path.

## AR37a — HC-read into QSA

Current caller: `FlashQsaLayer::forward_deferred` calls `LayerResidual::forward`,
which synchronously `attn_hc.read`s before `FlashQsa::forward`. The smallest patch:

1. Add typed `FlashQsa::forward_after_hc(hc, streams, scope)` using AR2's split.
   Prepare the slab, then one `gpu.command`: `hc.encode_read` first, QSA encode
   using `read.mixed()` second; return both owned pending payloads.
2. After successful command, finish HC and QSA; pass the resulting read/output
   to `LayerResidual::forward_after_attention_read`, preserving the entire T05,
   demanded-SSD, routing and final-write implementation.
3. Add only a narrow HC accessor and `pub(super)` visibility for that existing
   residual continuation. Modify the QSA-layer caller, not GDN/PLE arithmetic.

Guard before encoding: new strict opt-in flag, same GPU/queue, valid HC shape
[1,10240], AR depth0, no trace or capture, and AR2's no-CPU-score condition.
Keep workspace owners through the command rather than releasing an early lease.
Existing synchronous API remains the fallback for every rejected scope.

`FlashNext::hidden` must convey its existing full-trace/capture exclusion to the
QSA-layer caller: `trace=false` inside current layer methods alone does not prove
that the outer call is untraced. Retain the whole capture fallback, not S2's
selective-capture optimization. Direct component calls use explicit scope.

Counters: joined-HC-QSA count and fallback reasons, QSA/index/attention command
counts separately; no COW-copy count folded into the saving. At this model's
12 eligible QSA layers, expect12 HC-read boundaries removed per ordinary row
above AR2. This is a structural expectation, not a measured time or gain.

## AR37b — previous HC-write into HC-read into QSA

`FlashNext::hidden` currently permits `LayerOutput::Pending` only before GDN;
`cross_layer_eligible` explicitly tests next-GDN. Reuse `DeferredHcWrite`:

1. Add a narrow `DeferredHcWrite::encode_before_hc(hc, cmd)` that invokes existing
   `hc.encode_after_write` on its privately held streams/branch/gate. This keeps
   accepted write/norm fusion and exposes no raw private fields.
2. Add `FlashQsaLayer::forward_pending`: prepare QSA slab/COW, encode that HC
   preamble and QSA in one command, retain read and QSA payloads plus the old
   `MoeOutput` producer. Always settle the old producer, including encode errors.
3. Verify command success AND `write.finish_producer()` before QSA `finish`;
   only then continue the unchanged residual MLP using the returned **written**
   streams. Consumer failure restores this QSA snapshot; the outer FlashSnapshot
   restores already advanced producer layers, exactly as the existing GDN path.
4. Add a next-QSA eligibility query and one QSA pending arm to the private layer
   loop. All trace/capture/sparse/depth/owner guards reject before producer deferral.

Counts: existing34 GDN triples remain34; this model has12 QSA consumers, so the
fully eligible composition can reach46 triples, excluding PLE and the last
write. Require12 actual QSA triples, not a rise in a shared counter alone.
AR37a/AR2/AR37b savings overlap; qualify the exact composition, no addition.

## AR38a — final HC-write, mixer and full head

Current `hidden` completes the final write; `prefill_inner` then waits for
`mixer.read`, followed by a separate head submission/read. Minimal explicit split:

1. Extract private `hidden_tail(..., defer_final) -> LayerOutput`; keep current
   `hidden` as a synchronous wrapper for all existing non-head/capture callers.
   Only a permitted last-layer/head-present invocation may retain the final
   `DeferredHcWrite`; no pending object goes into snapshots or `last_hidden`.
2. In the final-token `prefill_inner` branch, one command encodes tail write,
   `mixer.encode_read`, then existing `head.encode_row(..., Shared)`.
   Keep the written streams, mixer ReadBuffers, full248320-Half logits and old
   MoE producer owned until command completion and producer-error checks.
3. Read the complete logits without extra rounding/conversion; only then publish
   `self.hidden` and existing offset/end. Preserve the outer snapshot/restore
   on encode failure, producer error, head failure or cancellation.

Normal steady head path can remove two CB/wait boundaries: final-write and mixer
join the existing head CB. Prefix tokens/head-absent calls, full trace and five
captures retain their existing behavior initially. Do not add an extra forward,
change target-head call count or drop the final10240-F32 hidden materialization.
T05 public MoeOutput API stays intact; its HotOutput already retains F. The tail
consumer reads F's owned result, not H's internal slab or recyclable records.

## AR38b — optional device-greedy compatibility, separately qualified

`forward_greedy_inner` and `device_greedy_head` already encode the full head plus
canonical Half LSE/selection. Reuse `encode_device_greedy` in the joined tail CB;
retain full logits when the existing test capture requests them, and call
`read_device_greedy` only after all producer checks. Preserve dtype rounding,
finitude, first-index ties, counters and scalar failure rollback. This does not
enable the AR30 greedy option or replace full-logit validation by token equality.
AR38a may ship with this path falling back until its affected gate is complete.

## Necessary gates, with existing independent oracles

- AR37 component: original separate HC/QSA PSOs vs joined path; compare HC mixed
  and gate, QSA output plus every key/value/index tail/block. Reuse AR2 lengths
  3/4,1023/1024,2047..2052 and retained divergent-prefix/COW tests; add owner,
  trace/capture and sparse fallback, failures after HC/index encode, retry and
  unsubmitted/drop/late-producer failures. Do not repeat unaffected kernel grids.
- AR37 model: extend existing cross-layer full-state/continuation/84-archive
  oracle with twelve QSA joins/triples, GDN34 unchanged, capture fallback and
  late producer/consumer rollback. Pending must never enter snapshot/prefix state.
- AR38 component/model: complete10240-F32 final hidden and248320-Half logits
  vs stock, last-only/multi-token prefill, head absent, prefix restore, head error
  after layer advancement, retained outputs and abandoned consumer. Reuse the
  independent AR20 model fingerprints57606144/993280/81920/84 and real CLI.
- Meaningful Kani: actual eligibility composition (source owner, exporting,
  depth, index presence, checked post-append bound), pending publication guards,
  final-head scope. No CPU proof claim about Metal, queues or concurrent I/O.

Caller limitation: MTP also invokes `target.forward` directly (`gpu/mtp.rs`), so
no-captures alone cannot implement a strict runtime MTP exclusion. If required,
carry an explicit plain-AR session scope; otherwise label qualification plain AR
without claiming such a fallback. No S1/S2 or speculative benchmark is proposed.
Root owns precise preregistration, flags, grants, builds, proof runs and GPU tests.
