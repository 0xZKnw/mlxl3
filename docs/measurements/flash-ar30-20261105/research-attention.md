# AR30 attention, GDN and resident-projection source research

Research only, 2026-11-05. Base2035348319a76bd505ad6991ccdd510345225568:
accepted AR10 defaults ON, twelve AR20 options OFF. Read AR30 preregistration,
AR20 B01–B12 matrix, GDN/attention research, attention-cost audit and ALL12
profile before searching. No executable edit, prototype, test, compiler,
proveur, GPU, weight read, index edit or commit was performed. The new global
issue and per-trial preregistration must precede any such work.

The two strongest opportunities found are **resident-stage encoder reuse** and
**leased resident temporary storage**. They target native host overhead without
changing EXL3 arithmetic. Below are five concrete native gaps and one explicitly
historical A08 extension. They are not six demonstrated boosts. The last case
should not inflate a count of entirely new mechanisms. Historical10.303tok/s is
a short sample, not a fresh control or evidence that any case reaches11–12.

## Pinned primary sources and applicability

HEADs were queried directly from official GitHub APIs; raw source was read at
these immutable SHAs. Small research copies are in `/tmp/ar30-attention-research`
and prior identical-SHA oMLX files in `/tmp/ar20-gdn-research`; the permanent
evidence below is the pinned source links, not availability of those temp files.

| Project | HEAD | Result |
|---|---|---|
| [oMLX](https://github.com/jundot/omlx/tree/2165e75726ba83d9651dbaced61c97728da77bba) |2165e75726ba83d9651dbaced61c97728da77bba|Current HEAD matches AR20 research; new files examined include native SDPA loader/dispatcher, not merely old patch flags.|
| [MLX-LM](https://github.com/ml-explore/mlx-lm/tree/736ab0d8c1d0edc79c3d798044fe6904e2fa23be)|736ab0d8c1d0edc79c3d798044fe6904e2fa23be|Current HEAD matches AR20; fused GDN/normalization already substantially ported.|
| [MLX Swift LM](https://github.com/ml-explore/mlx-swift-lm/tree/c8dabc63e7018d0b1567b2171f8df5dd8a334a55)|c8dabc63e7018d0b1567b2171f8df5dd8a334a55|Real fused projection lifecycle and compiled GDN paths; arithmetic/runtime exclusions below.|
| [MLX Swift](https://github.com/ml-explore/mlx-swift/tree/5f117a2e7ee1ce92f8dcc1520bc07535c43846f9)|5f117a2e7ee1ce92f8dcc1520bc07535c43846f9|Official MLXFast API; this identifies MLXFast, not an invented independent “MLXSwiftFastLLM” engine. Exact latter name was not identified.|
| [Swiftlet](https://github.com/leonickson1/Swiftlet/tree/3d5c2352912824237354330c836a0d0e2bc75c56)|3d5c2352912824237354330c836a0d0e2bc75c56|Native Swift/Metal engine with actual single-encoder stages and fixed scratch Regions; strongest new implementation source.|
| [Metal Flash Attention](https://github.com/philipturner/metal-flash-attention/tree/8671cddc38f19a6eadb804dee6a3ca2954b8bf32)|8671cddc38f19a6eadb804dee6a3ca2954b8bf32|Matrix-tiled attention/register caching; not a drop-in canonical32-chain single-query implementation.|
| [MPS linear attention](https://github.com/mpsops/mps-linear-attention/tree/f0889940cf168dcc9e2541b061875835e63a095e)|f0889940cf168dcc9e2541b061875835e63a095e|Implemented GDN kernels, but inconsistent file/embedded state types need care; no native transplant recommended.|
| [BaseRT](https://github.com/basecompute/baseRT/tree/23bddc577e385840c8e0af4de850a4714833adda)|23bddc577e385840c8e0af4de850a4714833adda|Public ecosystem/API, not public engine kernel source in this tree; no hidden implementation inferred from throughput claims.|

## AT30-1 — reuse one encoder inside resident projection/GDN stages

**Priority high.** Swiftlet actually passes one opened encoder through a whole
GDN stage: [encDeltaCore1739–1767](https://github.com/leonickson1/Swiftlet/blob/3d5c2352912824237354330c836a0d0e2bc75c56/Sources/SwiftletCore/QwenMetalModel.swift#L1739-L1767),
[encoder creation641–675](https://github.com/leonickson1/Swiftlet/blob/3d5c2352912824237354330c836a0d0e2bc75c56/Sources/SwiftletCore/QwenMetalModel.swift#L641-L675),
[explicit buffer barrier1459–1461](https://github.com/leonickson1/Swiftlet/blob/3d5c2352912824237354330c836a0d0e2bc75c56/Sources/SwiftletCore/QwenMetalModel.swift#L1459-L1461).
Its [encodeGemv227–266](https://github.com/leonickson1/Swiftlet/blob/3d5c2352912824237354330c836a0d0e2bc75c56/Sources/SwiftletCore/MetalEngine.swift#L227-L266)
binds/dispatches on that encoder. This is implementation/caller evidence, not
an isolated benchmark for this mechanism.

Local `gpu.rs::encode_bound` creates and ends an encoder per dispatch.
`projection.rs::encode_expert_stage` performs prepare→QMV→finish through three
such calls even for immutable residents; `gdn.rs::encode_for_consumer` adds
multiple resident chains to the existing command buffer. Start with those
resident chains, retaining the command-buffer boundaries and every current
GPU wait/error/commit boundary. Open one encoder, establish dependency barriers
between existing kernels and close it before returning to a caller that can
blit or wait. Preserve literal kernels, FP16 materializations and FMA order.

This is **not permission to resume the canceled same-expert encoder/P04**:
exclude routed-expert/cohort encode paths, cold/hot scheduling and MoE producer
submission. The new domain is resident GDN/QSA/head work above current AR10.
Its benefit must be attributed by actual encoder count plus short end-to-end
controls; the old MLX encoder research and canceled expert pilot remain in
the ledger. No current recoverable milliseconds are established.

## AT30-2 — lease only resident prepared/partial temporaries

**Priority high, ownership gate first.** Swiftlet builds one fixed
[Regions layout530–561](https://github.com/leonickson1/Swiftlet/blob/3d5c2352912824237354330c836a0d0e2bc75c56/Sources/SwiftletCore/QwenMetalModel.swift#L530-L561)
and passes [TokenSlot1388–1406](https://github.com/leonickson1/Swiftlet/blob/3d5c2352912824237354330c836a0d0e2bc75c56/Sources/SwiftletCore/QwenMetalModel.swift#L1388-L1406)
offsets through the stages. Locally, resident `encode_rows` reaches
`encode_with_scratch(...,None)` and allocates prepared/partial buffers for each
projection. A06 already reuses expert scratch within a qualified cohort;
the new scope is cross-row resident intermediates, not an entirely new allocator.

Prepared/partial storage may be reused only behind a private entry that **never
lets these views escape**, or through explicit completion-bound leases. Current
MatrixView/TensorStorage and cloned Buffers are immutable ownership contracts.
“Last command buffer completed” alone does not exclude external aliases,
retained ExpertEncoded, trace captures or repeated encodes inside one command
buffer. Fresh fallback is required for retained/in-flight/same-CB repeat/trace/
MTP/public cases unless separately proved. Destinations, frames, KV caches,
prefix snapshots and current recurrent state are excluded from the arena.

This differs from the historical frame pool whose measured allocation API
cost was only0.2477ms/row and from excluded HC workspace. The relevant new
budget is the native resident prepared/partial allocation count above AR10;
it must first be measured in a preregistered diagnostic. Keep explicit reserve,
completion, release, failed-submit and rollback transitions; no wait-for-memory
cycle before command submission. MLX [#3689](https://github.com/ml-explore/mlx/issues/3689)
provides a concrete in-flight resource-retention failure; [#3142](https://github.com/ml-explore/mlx/issues/3142)
is a separate fast synchronization/distributed deadlock report, not proof of a
bug in this native path. Neither is an optimization to copy.

## AT30-3 — aligned vector loads/stores, same scalar arithmetic

oMLX's actual [load_vec16–36](https://github.com/jundot/omlx/blob/2165e75726ba83d9651dbaced61c97728da77bba/omlx/custom_kernels/attention/csrc/sdpa_decode_kernels.h#L16-L36)
uses wide aligned memory operations; [query/key calls116–141](https://github.com/jundot/omlx/blob/2165e75726ba83d9651dbaced61c97728da77bba/omlx/custom_kernels/attention/csrc/sdpa_decode_kernels.h#L116-L141)
and scalar tail [225–247](https://github.com/jundot/omlx/blob/2165e75726ba83d9651dbaced61c97728da77bba/omlx/custom_kernels/attention/csrc/sdpa_decode_kernels.h#L225-L247)
establish real callers.

Native `flash_qsa.metal` loads eight contiguous Half words per lane with scalar
subscripts; `gated_delta_packed.metal` copies32 contiguous F32 state words per
lane. A small load/store helper can use aligned float4 or bit-preserving Half
packs, then execute the **unchanged** scalar score/recurrence/reduction tree.
No tile-of-four softmax update, reassociation, Half-state conversion or new
scratch is admitted. Alignments are guaranteed only for these internal fixed
layouts; reject other offsets/strides. This is distinct from B03's four-key
live window. The compiler may already vectorize, so emitted resource/code
metadata and an indispensable component screen must establish any improvement;
source-level scalar counts are not hardware transaction counts.

## AT30-4 — specialize the genuinely dense gated QSA path

oMLX's [function constants119–138](https://github.com/jundot/omlx/blob/2165e75726ba83d9651dbaced61c97728da77bba/omlx/custom_kernels/attention/csrc/sdpa_decode.cpp#L119-L138)
build distinct cached PSOs for masks/layout/causality, used by the real
one-pass dispatcher. Native A09's `ar10_qsa_attention_gate` still contains
`params[1]?positions[t]:t`, although its real eligibility helper excludes sparse
positions. Compile a dense-only body with those addresses fixed, preserving
the exact32 stripe chains, original online-update order and Half gate.
Keep the canonical sparse/public/trace path; no new external or approximate
mask behavior. This is compile-time removal of a proven-dead generic branch,
not B02's extra producer/combine or a new attention algorithm. Its absolute
opportunity is probably smaller than the first two; no large boost is attested.

## AT30-5 — cache immutable capability/geometry qualification

oMLX builds and caches [the GDN decode plan1496–1576](https://github.com/jundot/omlx/blob/2165e75726ba83d9651dbaced61c97728da77bba/omlx/patches/qwen35_gdn_prework.py#L1496-L1576)
and the actual [decode caller2064–2074](https://github.com/jundot/omlx/blob/2165e75726ba83d9651dbaced61c97728da77bba/omlx/patches/qwen35_gdn_prework.py#L2064-L2074)
uses it while rechecking dynamic state separately.

Native projection load already caches static/fixed keys; do not count that
again. The gap is repeated immutable pipeline metadata at
`gdn.rs::encode_for_consumer` and `qsa.rs::encode_ar20_qsa`, including
thread_execution_width/max_total_threads checks. Cache capability verdicts
when the actual optional PSO is created, keep runtime flags/trace/dtype/cache
shape checks live, and keep absent-PSO fallback explicit. Scope this with
AT30-1's host path but evaluate it as one distinct host-plan mechanism, not
one candidate per getter. It is unmeasured and likely secondary; a cached
verdict must not outlive/refer to a replaced pipeline or collapse the test
Cells' OFF/ON semantics.
At the current baseline both AR20 QSA switches are OFF, so their optional
capability queries contribute no current work; the immediate measured target
would be GDN's always-reached metadata checks. Do not count hypothetical QSA
savings before enabling and qualifying another path.

## AT30-6 — A08 extension: producer appends K/V directly

This is a **previously documented A08 extension**, not an entirely new idea.
Swiftlet's real [encodeAttnKVAppend963–989](https://github.com/leonickson1/Swiftlet/blob/3d5c2352912824237354330c836a0d0e2bc75c56/Sources/SwiftletCore/MetalEngine.swift#L963-L989)
launches [the append kernel541ff](https://github.com/leonickson1/Swiftlet/blob/3d5c2352912824237354330c836a0d0e2bc75c56/Sources/SwiftletCore/Kernels.metal.txt#L541):
K normalization/RoPE writes the cache slot, V copies into its paired slot.
Native QSA's A08 produces a temporary kr then opens a blit encoder for two
1024-byte copies (`qsa.rs:575`).

New reason to revisit: removing that blit is a prerequisite for a contiguous
AT30-1 QSA compute-encoder span, so the corrected mechanism targets encoder
switching as well as one temporary. Retain native A08 arithmetic literally,
not Swiftlet's different F32 normalization. Use the **already selected** slab
and append offset; do not change high-water/COW, publication or rollback.
Retained-prefix, restore-and-overwrite, capacity-growth, failed encode and
trace tests must establish that only unpublished suffix slots are written.
No standalone large gain or independent added percentage is inferred.

## Important negative findings and non-candidates

- B01 gate hoisting remains a measured major regression; no new register/TG
  evidence justifies retry. Whole GDN, joint AB, QK norm/RoPE and gated store
  are already present. Native recurrence already caches local Q/K/V.
- B02/B03 are not repackaged as new cases. A possible **future corrected B02**
  would keep paired heads and all32 canonical chains inside one CTA, avoiding
  792,576bytes global scratch and the second dispatch; that exact design was
  not found implemented upstream in the sources read. Do not call it attested
  or prototype it before a separate issue/protocol. B03's next-window loads
  remain inactive for keys≤128, including MAIN87–119.
- MLX Swift LM's [merged PR740](https://github.com/ml-explore/mlx-swift-lm/pull/740)
  explicitly reports decode unchanged on its tested model, despite large
  training/prefill layer improvements. Its [F32 fast GDN adapter343–365](https://github.com/ml-explore/mlx-swift-lm/blob/c8dabc63e7018d0b1567b2171f8df5dd8a334a55/Libraries/MLXLMCommon/GatedDelta.swift#L343-L365)
  protects decay precision; its other kernel uses Kahan summation. Neither
  is a bit-exact replacement for our canonical native recurrence. Its
  [four-projection storage fusion182–224](https://github.com/ml-explore/mlx-swift-lm/blob/c8dabc63e7018d0b1567b2171f8df5dd8a334a55/Libraries/MLXLMCommon/FusedQuantizedLinear.swift#L182-L224)
  uses affine packed weight concatenation plus views. Arbitrary EXL3 qkv/z
  require distinct Suh transforms; B04 already addresses co-dispatch only.
- Metal Flash Attention implements [async/register caching38–96](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/Attention/AttentionKernel/AttentionKernel%2BCaching.swift#L38-L96),
  but its matrix tiling/softmax organization is a new numerical path and the
  public README calls out additional multihead/mask integration work. It is
  not evidence of a drop-in speedup for our single-query,24Q/2KV,D256 kernel.
- MPS linear attention needs caller-level audit: [Python state allocation50–65](https://github.com/mpsops/mps-linear-attention/blob/f0889940cf168dcc9e2541b061875835e63a095e/mps_linear_attn/delta_rule.py#L50-L65)
  is F32, whereas the separate [kernels.metal84–131](https://github.com/mpsops/mps-linear-attention/blob/f0889940cf168dcc9e2541b061875835e63a095e/mps_linear_attn/csrc/kernels.metal#L84-L131)
  declares Half in-place state. [mla.mm31–52](https://github.com/mpsops/mps-linear-attention/blob/f0889940cf168dcc9e2541b061875835e63a095e/mps_linear_attn/csrc/mla.mm#L31-L52)
  tries that file first, while [its embedded fallback273–300](https://github.com/mpsops/mps-linear-attention/blob/f0889940cf168dcc9e2541b061875835e63a095e/mps_linear_attn/csrc/mla.mm#L273-L300)
  is F32. This inconsistency depends on packaging/load path and is not asserted
  to have been executed here. It disqualifies copying a README headline as
  numerical implementation evidence. In-place recurrence also conflicts with
  our immutable retained frames.

## Minimum issue/protocol requirements before implementation

Keep AT30-1/2 as the first candidate group, AT30-3/4/5 as secondary mechanisms
and AT30-6 as an explicit historical extension. Deduplicate with the other
research agents before counting issue cases. No gain percentages are supplied.
The historical profile gives only relative context; a fresh preregistered
diagnostic after issue creation must confirm the current default bottleneck.

For any port, retain the independent original Metal/component branch or frozen
stock, full finite/nonempty outputs, all84-state archive/logits/hidden and real
CLI before MAIN32. AT30-1 needs dependency and encoder counts with retained
owners/failure at each boundary; AT30-2 needs actual leases, no-alias/no-inflight
proof harnesses plus real abandon/rollback/snapshot counterexamples. AT30-3/4
must preserve every FMA/reduction/Half boundary, poison outputs and exercise
alignment/shape/sparse/trace fallbacks. Numeric tolerance must not be weakened.
Two short controls do not establish a stable boost; follow the existing
32-token protocol and treat strong control drift as non-conclusive.
