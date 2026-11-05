# B02/B03/B04 cost audit of the common frozen source

Source-only audit while root owns MAIN32. No prototype, compiler, prover, GPU or
checkpoint read. No root Rust edits. The source line references below are in
the root AR20 worktree. Physical correctness was qualified by root, not inferred
from this cost audit.

## B02: QSA chains

`native/src/gpu/qsa.rs:1182` returns before allocating anything when both new
options are OFF. Optional PSOs at lines 48–53 are absent OFF. The extra runtime
cost OFF is the guard/helper call and flag checks; there is no per-row environment
lookup, scratch allocation or added dispatch. New shader text is still included
in the one Metal library compiled at startup (`gpu.rs:663`, `:704`), including
when OFF. That startup effect was not measured and is outside warm decode TPS.

ON, each of 12 attention layers allocates 3 private buffers at `qsa.rs:1217`:
3,072 + 3,072 + 786,432 = **792,576 requested bytes/call**. Per decode row this
is **36 new allocations, 9,510,912 requested bytes** (9.07 MiB), and one extra
dispatch per layer. These are requested transient bytes, not process RAM or a
peak allocation measurement. Producer writes full F32 partials and the combine
reads them; the output slab alone is 786,432 bytes in each direction per call.

Producer grid is 48×2 CTAs×128 threads, combine 24×1×1024. This creates 36,864
thread instances versus the canonical 24×1024=24,576; it halves the producer's
stripe groups by computing two queries per group, then adds a full combine.
Two-query KV reuse and removal of canonical 17 threadgroup barriers are real
source changes, but no measured register/occupancy benefit is established.
At 87 keys the logical KV-load reduction is about 1.07 MB/call, whereas the
new F32 output scratch alone adds about 1.57 MB write+read; hardware cache
reuse can further reduce the actual original KV traffic. These are source-level
traffic counts, not DRAM measurements or an additive runtime prediction.

Caller at `qsa.rs:589` uses the real dense/private FP32 path. Guard requires
1≤keys≤1023, no trace/sparse positions, SIMD32, producer maxTG≥128 and combine
maxTG≥1024. Trace, public/F16, sparse, long or unsupported capabilities fall
back to the prior canonical paths without new allocation or encoding. Both
new flags ON load four QSA PSOs up front, although normal short dense calls use
only chains-prefetch and combine; the base producer and standalone prefetch
PSOs remain available for controlled variants/fallback, not decode-time work.

## B03: four-slot prefetch

`native/shaders/flash_qsa_chain.metal:8` stores four K/V slots and up to eight
query scores before the ordered online updates. Floating-point score and
online-update order are preserved; real physical component/full-model gates
cover the actual compiled path.

Concrete short-context limit: next-window loads at lines 23 and 34 require
`t+(slot+4)*32 < keys`. **For keys≤128 none execute**; the initial four-slot
window covers every stripe. MAIN's approximately 87–119 keys therefore exercise
batched loading/scoring but no steady-state next-window software pipeline.
The larger K/V and score live arrays still exist in source. Exact compiler
register allocation/spills are unmeasured and must not be asserted from arrays.
Standalone prefetch adds no scratch or dispatch relative to canonical attention;
combined prefetch inherits B02's allocations/combine. A weak MAIN result would
not establish that the cross-window mechanism fails at 129–1023 keys. Any new
threshold or longer-context measurement would need a new preregistration.

## B04: resident qkv/z bundle

`gdn.rs:325` skips the pair entry completely OFF; `gpu.rs:801` creates its one
PSO only ON. Strict actual guards (`projection.rs:374`) require shapes
[2560,10240]/[2560,6144], K4/CB2/NT2, split1, resident non-generic/non-MTP/non-routed
zero offsets, no bias, same Metal instance, SIMD32/maxTG≥128, and one valid
2560-wide row. Production caller additionally requires nontrace/private FP32
consumer. Qualified flag selection is exact Apple M5. Current 36 pairs match
the metadata audit; no model metadata was reread in this review.

The bundle at `projection.rs:400` retains exactly the canonical six allocations:
two 5,120-byte transformed inputs, 40,960+24,576-byte F32 partials, and
20,480+12,288-byte Half outputs: **108,544 bytes/pair**, unchanged versus two
ordinary projections. Both distinct Suh preparations and both canonical
finish/H128/Svh operations remain. It changes six dispatches to five, saving
one QMV encoder, two buffer bindings (8→6), one parameter upload, and one
dispatch per pair. Across 36 pairs that is 36 fewer dispatches per row.

The single QMV still launches all 320+192=512 groups with 128 threads and
includes the canonical tile body literally (`flash_gdn_resident.metal:35`).
Weights, FMA count, separate transforms, partial materialization and Half
boundaries are not reduced. The group-uniform qkv/z selection adds pointer/stride
selection; its compiler/resource cost is unmeasured. No concrete correctness
or OFF-cost bug was found. This implementation is a dispatch optimization,
not evidence for a large compute/bandwidth reduction.

## Reused proof scope and receipts

The four actual helper/harness function bodies match their original SHA-256
values exactly, and the existing `ar10_qsa_gate_eligible` and
`gdn_private_epilogue_eligible` callees also match own proved source. See
`attention-proofs/root-source-functions-canonical.json` and `root-callees.json`.
The first extraction receipt included a trailing newline and therefore reported
four textual mismatches; the canonical receipt excludes that newline exactly
as the original proof metadata did. It is preserved as an extraction diagnostic,
not a source mismatch. Root's host/source attributes retain the same Kani
selection and B04 unwind40. This supports reuse of the existing 2 bounded CPU
checks (641 obligations, 7 covers); it proves no Metal scheduling, arithmetic,
allocator or end-to-end performance property.

Controller follow-up: strict duplicate-key marker rejection and the 47 small
Python checks/offline log reparse passed. Targeted Ruff check of root quality.py
and speed.py also passed; snapshots match. No additional inference was launched.

Read-only provenance check: all 11 audited native Rust/Metal files match
`freeze-common.json` exactly; `attention-audit-source.json` records their hashes.
Both selfcheck scripts also pass targeted Ruff, and own `git diff --check`
passes. No formatting of archived source snapshots was attempted.
