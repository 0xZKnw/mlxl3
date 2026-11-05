# AR30 — native AR kernel research, 2026-11-05

Status: source research only. No prototype, executable change, build, test,
proof, model read, GPU work or benchmark was performed for this note. Read the
AR30 preregistration in `opti.md` before this research, then the AR10 bank/matrix,
AR20 B05–B08 source/proof reports, 54Q, PERF89 and PERF111. This note is input to
the GitHub issue and experiment matrix that must precede implementation.

Baseline: `2035348319a76bd505ad6991ccdd510345225568`, qualified AR10 defaults ON,
all twelve new AR20 options OFF. The historical MAIN32 default/forced values
10.302951/10.019801 tok/s are short samples, not fresh controls or evidence of
sustained throughput. No candidate below has a measured native gain. Reaching
11–12 tok/s remains an objective, not a forecast.

## First choices

1. **Cooperative loads of packed EXL3 words, followed by SIMD shuffles**. Actual
   ExLlamaV3 code provides a directly relevant integer-rate implementation.
   Preserve the native codeword, Half materialization and accumulation exactly;
   change only where each packed word is loaded. This is distinct from B06's
   per-lane reader8 and B07's preloading of the same per-lane windows.
2. **Remap lane ownership to eliminate the reference-partials threadgroup
   round trip**, while retaining the exact row→group sum order. Upstream MLX and
   LLMEnv demonstrate register/SIMD output ownership, but their reduction trees
   are not valid native oracles. This has a larger structural target than a
   decoder peephole and also a larger correctness/engineering cost.

The next low-cost choices are the router's multi-output SIMD mapping and packed
extraction metadata. Uniform loop metadata and PSO resource screening are
conditional compiler/resource ideas. The small post-hash table is a mathematically
different follow-up, **not** an upstream-implemented EXL3 speedup.

## Primary source pins and identity

HEADs were read from GitHub branch API responses and the implementation files
were read at those exact commits. No floating branch link is used as evidence.

| Project | Inspected HEAD | Relevant implementation |
| --- | --- | --- |
| TurboEXL3, David Tai, Metal/Zig | `c3bb0e260091709f8ad64c84d1fd02808b12dfd1` | [lane helpers](https://github.com/davidtai/turboexl3/blob/c3bb0e260091709f8ad64c84d1fd02808b12dfd1/src/kernels/lane/helpers.metal#L60-L178), [dense decode GEMV](https://github.com/davidtai/turboexl3/blob/c3bb0e260091709f8ad64c84d1fd02808b12dfd1/src/kernels/decode/gemv_core.metal) |
| ExLlamaV3, turboderp, CUDA | `bd1f519172406a9560bd5c90e61000691bb69a07` | [GEMV metadata and coalesced/shuffle decoding](https://github.com/turboderp-org/exllamav3/blob/bd1f519172406a9560bd5c90e61000691bb69a07/exllamav3/exllamav3_ext/quant/exl3_gemv_kernel.cuh#L257-L402) |
| MLX core | `c69303b00f5f61860436e832f23fa8c34b96f93a` | [quantized GEMV](https://github.com/ml-explore/mlx/blob/c69303b00f5f61860436e832f23fa8c34b96f93a/mlx/backend/metal/kernels/quantized.h#L794-L927), [dense GEMV](https://github.com/ml-explore/mlx/blob/c69303b00f5f61860436e832f23fa8c34b96f93a/mlx/backend/metal/kernels/gemv.h#L122-L216) |
| TensorFold | `a6e25faea51b68a1d3879e5518c83cecc7479cf8` | [EXL3 projection and packed address maps](https://github.com/ashhart/TensorFold/blob/a6e25faea51b68a1d3879e5518c83cecc7479cf8/zig/kernels/metal/core/exl3_projection.metal#L115-L148), [vector packed loads](https://github.com/ashhart/TensorFold/blob/a6e25faea51b68a1d3879e5518c83cecc7479cf8/zig/kernels/metal/core/tile_qmv.metal#L25-L44) |
| llama.cpp | `61d447234482cc0807cab652d763e0fdb5f5d816` | [Metal multi-tile/MMA GEMV](https://github.com/ggml-org/llama.cpp/blob/61d447234482cc0807cab652d763e0fdb5f5d816/ggml/src/ggml-metal/kernels/mul_mv_mma.metal#L507-L537) |
| ik_llama.cpp | `656036d98503e154952f6c633cb3febc1a4f08b4` | [small nonlinear codebooks](https://github.com/ikawrakow/ik_llama.cpp/blob/656036d98503e154952f6c633cb3febc1a4f08b4/ggml/src/ggml-metal.metal#L4101-L4134), [real IQ4_NL lookup caller](https://github.com/ikawrakow/ik_llama.cpp/blob/656036d98503e154952f6c633cb3febc1a4f08b4/ggml/src/ggml-metal.metal#L6145-L6174) |
| Metal Flash Attention, Philip Turner | `8671cddc38f19a6eadb804dee6a3ca2954b8bf32` | [resource-screened PSO selection](https://github.com/philipturner/metal-flash-attention/blob/8671cddc38f19a6eadb804dee6a3ca2954b8bf32/Sources/FlashAttention/GEMM/GEMMDescriptor/GEMMDescriptor%2BPipelineCache.swift#L75-L123) |
| FastLLM's engine LLMEnv | `a01bc18ac0e9e4967af17ea400fd611e4f9cc026` | [SIMD-owned QMV outputs](https://github.com/c-vision/LLMEnv/blob/a01bc18ac0e9e4967af17ea400fd611e4f9cc026/LLMEngine/metal/qmv_multi.metal#L1-L46), [Q4 GEMV variants](https://github.com/c-vision/LLMEnv/blob/a01bc18ac0e9e4967af17ea400fd611e4f9cc026/LLMMetal/gemv/llmmetal_gemv_q4.h) |

TurboEXL3 and CUDA ExLlamaV3 are different projects. TurboEXL3's drafting
throughput is not AR throughput. FastLLM points to a separate LLMEnv engine;
this research does not invent a Swift engine identity from the search phrase
“MLX Swift FastLLM”. The inspected LLMEnv kernels are actual Metal sources.

“Implemented upstream” below means the mechanism has a real implementation and
caller. It does not mean an isolated speedup was established on this checkpoint,
quantization, M5 or native arithmetic contract. No external TPS claim is used to
predict a native boost.

## Candidate K1 — cooperative packed-word loads and shuffle reconstruction

**Evidence.** ExLlamaV3 lines295–337 load coalesced packed words. Its integer
K4 path at384–387 shuffles the previous lane's word; K2 at389–396 loads a group
then shuffles the two source words; K3 at398–402 does the same with its own
source indices. Its subsequent MMA is CUDA-specific and is **not** proposed.

**Local gap.** `native/shaders/_qmv_tile_kernel.metal:28–37,148–158` computes
per-lane source indices and reads two words for each of two windows. B06 at130
uses `ar20_window8`, which still reads the window independently per lane; B07
stages those reads, without cooperative word ownership. The mapped kernel has
the same pattern at39–48,216–218. Resident/routed production selection is in
`native/src/gpu/projection.rs` and the routed expert projection path; preserve
their existing NT2/A06-axis/split guards and output layout.

**Adaptation.** For K2/3/4, one SIMD loads the tile's unique16/24/32 packed
u32 words, broadcasts the required words with `simd_shuffle`, and retains the
existing window funnel extraction, canonical codeword and eight FMA chains.
No new checkpoint layout, scratch slab or dispatch. Lanes without an owned word
must still execute every required shuffle; use neutral data and do not return
early. Treat ring wrap explicitly. K>4 and unsupported layouts retain baseline.

**New reason/domain.** This is inter-lane data distribution, not another value
of reader8/preload. It attacks duplicate load instructions/cache transactions;
cache coalescing may already hide much of the physical traffic, so no bandwidth
multiplier is asserted.

**Gate/cost.** Exhaustive lane/window comparison against canonical `state()`
for supported K and adversarial packed patterns, nonempty independent frozen
routed K2 and resident K2/3/4 outputs, full states/logits/CLI afterward. No
reassociation. Expect1–2 live packed-word registers plus shuffles per lane and
no extra TG storage; actual register/PSO behavior remains unmeasured. First
micro should use one representative resident form and one routed form, not a
large grid sweep.

## Candidate K2 — SIMD ownership with the exact canonical reduction

**Evidence.** MLX `qmv_quad_impl:805–845` gives each quad8 output rows and
reuses one input load. `qmv_fast_impl:862–919` gives each SIMD4 output rows,
with register accumulators. LLMEnv `qmv_multi.metal:21–43` gives a SIMD each
output row. These are affine quantizers and use their own reduction trees.

**Local gap.** `flash_reference_reduction.h:2` selects the canonical path.
`_qmv_tile_kernel.metal:23,194,228–238` stores
`reference_partials[MB][NT][SG][256]`, synchronizes, then reads rows/groups in
the fixed order. MB1/NT2/SG4 means8192 bytes of TG float storage. 54Q changed
only this scratch indexing/layout and was not a register-ownership rewrite.

**Adaptation.** Change lane→element ownership so one SIMD contains the
canonical per-column partial chains needed by the final fixed row/group sum;
fetch them in that identical order with shuffles, leaving each chain's FMA
sequence unchanged. Do **not** replace the sum by `simd_sum` or an MMA tree.
First design must write down a bijective mapping for every one of the256 tile
elements and every split/group, including NT2. It may need more shuffles or
registers than it saves. If the mapping cannot preserve the current sequence
without cross-SIMD storage, stop/reduce scope rather than weaken the oracle.

**New reason/domain.** Removes or reduces the TG round trip/barrier instead of
swizzling it (54Q), and differs from B08's H128 SIMD rewrite after the QMV.
The upstream code attests ownership techniques, not this proposed exact EXL3
adaptation.

**Gate/cost.** Independent partial-chain fixtures plus frozen stock whole
projections; assert the full element/column count and finiteness. Exercise the
real NT2/split callers. Full continuation/state/logits/CLI before timing. This
is a second-stage candidate: moderate source change and potentially a larger
live accumulator set, so no broad parameter search in the first experiment.

## Candidate K3 — compress extraction metadata into one scalar register

**Evidence.** ExLlamaV3 lines257–273 packs six5-bit indices/shifts into a u32
and extracts fields at use. TensorFold EXL3 lines122–140 packs word/shift/next
metadata before its inner loop; its entire projection arithmetic is not a
native oracle.

**Local gap/adaptation.** `_qmv_tile_kernel.metal:28–37` keeps six uint metadata
values in `word0[2]`, `word1[2]`, `shifts[2]`. For K2/3/4 each word index and
shift fits5 bits, so a u32 can represent the same six fields. Retain exactly
the current address formula, two-word funnel and arithmetic; unpack at use.
Keep other K fallback, especially the head's wider word indices. Do not combine
this first experiment with B06 or change reader cardinality.

**Gate/cost.** Exhaust all supported K/lane fields against the old arithmetic,
prove pack/unpack bounds with the real helper if a CPU contract is introduced,
then frozen projection and model gates. Potential saving is live metadata
registers, traded for shifts/masks; the compiler may already optimize the old
arrays away. A correct implementation can easily be neutral or slower. This is
a separate integer-metadata mechanism, not a promised large gain.

## Candidate K4 — multiple router outputs per SIMD, shared input loads

**Evidence.** MLX dense `GEMVKernel` has TM register accumulators, loads
`v_coeff` once (`gemv.h:122–134,161–178`), then performs its per-output shuffle
tree (`201–207`). This is the multi-output dense GEMV mechanism, not an EXL3
quantizer port.

**Local gap.** `native/shaders/flash_gate.metal:3–16` launches one32-lane
SIMD per one of512 router outputs. Each independently traverses the same2560
input values. `native/src/gpu/gate.rs:118–135` actually dispatches512 groups
per row, so this is a production caller, not unused code.

**Adaptation.** TM2 or TM4 outputs per SIMD, each keeping the identical80-FMA
per-lane chain and16/8/4/2/1 shuffle tree. Reuse each input load across the TM
weights, maintain Half/F32 output materialization and leave softmax/top-k
unchanged. Preserve negative-stride/layout support and the SIMD32 capability
guard. This is not AR10's GDN a/b pairing or HC L3/L5.

**Gate/cost.** Compare every512 router logits and probabilities, route IDs and
scores, not just the selected experts; test actual negative strides and both
input precisions, then complete model trajectory.2 or4 live accumulators and
fewer CTAs can hurt occupancy. Weight traffic remains dominant. This is a small
production change suitable for one two-point microcomparison after quality.

## Candidate K5 — explicit uniform loop/control metadata

**Evidence.** MLX dense `gemv.h:155–159` uses `uniform<int>`/`make_uniform`
for loop stride, input size, iteration count, last iteration and remainder.
The loop at161 uses those values. This is source-level implementation evidence,
not an ISA/register-count measurement.

**Local gap/adaptation.** No `uniform<` or `make_uniform` use was found in the
native tree. The QMV parameters/split bounds and fixed projection dimensions
provide values known uniform to the relevant execution group. A narrowly
guarded implementation can mark only these proven-uniform values, leaving
lane indices, routes and data-dependent values ordinary. Check MSL/toolchain
availability before choosing this for an experiment; maintain old-source
fallback. This differs from B01's shared-memory GDN gate-value broadcast and
from changing loop dimensions/unroll factors.

**Gate/cost.** Compile/resource inspection is necessary after issue creation;
if the native toolchain lacks the type or the old compiler already infers
uniformity, record that and stop. No numerical operation changes. Existing
physical projection/model oracles still bind the actual shipped binary.
Potential cost is minimal source/PSO work; practical benefit is unmeasured and
may be zero. Do not call this an independently attested throughput boost.

## Candidate K6 — resource-screened PSO selection, one mechanism

**Evidence.** Metal Flash Attention's pipeline-cache source compiles four
tile/store variants (lines75–104), retains those with greatest
`maxTotalThreadsPerThreadgroup` (107–116), then chooses the last remaining
candidate (118–119). The source calls this occupancy, but the queried field is
a pipeline capacity proxy, not measured resident waves or real execution time.

**Local gap/adaptation.** `native/src/gpu.rs:712` constructs the chosen PSO;
capability guards check minimum thread capacity. The current kernel experiments
have not used an explicit capacity/resource screen to reject a correct but
resource-heavy variant before whole-model runs. Add metadata collection for a
very small, predeclared candidate family, choose only within a quality-qualified
family, and retain the fixed default fallback. All tile/thread/grid alternatives
in this family count as **one** mechanism, not several new optimizations.

**Gate/cost.** This is a way to avoid expensive unpromising trials, not itself
a measured speedup. More PSOs cost compile time/startup memory, so no runtime
autotuning sweep and no1000-PSO cache. B01's correct-but-slow result illustrates
why a resource hypothesis needs actual timing; this proxy alone cannot explain
that regression. Require exact selected-key/PSO metadata in the receipt and
independent numerical output gates before the normal short AR measurement.

## Follow-up K7 — small post-hash codeword table, explicitly a new inference

IK's small nonlinear codebook tables and actual IQ4_NL lookup demonstrate the
mechanism on Metal, but **no inspected upstream implements the following EXL3
factorization**. Native `exl3.metal:15–19` hashes16-bit state, sums four bytes
to a value in0..1020, adds0x6400 and executes a Half FMA. A1021-entry Half
table (2042 bytes), indexed by the exact byte sum, can in principle replace
only that final Half FMA. Keep the hash/byte sum unchanged; fill the table using
the canonical Half semantics, not a float approximation.

PERF89's rejected K3 eight-entry table was invalid because transition width K
does not reduce the16-bit decoded-state domain. This candidate has a different,
justified domain: the range of the post-hash sum, independent of K. It is also
different from B05's SWAR sum and Turbo's deferred MUL1 float algebra. All65536
states must agree bit-for-bit with the original canonical GPU function, with a
separately computed oracle and nonempty outputs. Cache traffic can be worse
than the removed FMA. Keep this as a conditional later experiment; it cannot be
advertised as an already attested EXL3 optimization or one of six guaranteed
large boosts.

## Historical exclusions and deferred alternatives

- **B05–B08** already implement SWAR byte sum, reader8, K preloading and H128
  SIMD. Their numeric gates passed; AR20 speed controls drifted heavily. They
  remain OFF, and this note does not rename four switches into four new ideas.
- **54Q**: scratch swizzle and reader64 were bit-exact in120 forms/24 negative
  controls. Reported micro changes were small/mixed; one control drifted13.29%.
  No native AR gain was established. K2 removes/reassigns the storage instead of
  repeating that swizzle, and K1 changes cooperative ownership, not reader width.
- **PERF111 head repacking** into paired output tiles was exactly invertible
  and bit-exact for M1/5/6/8, but used381,419,520 extra bytes (~0.355GiB) for
  only0.76–1.81% head micro changes. Do not repeat it as a fresh head boost.
  A sparse-expert, miss-time packed layout is a genuinely different domain,
  suggested by TensorFold's vector `uint4` loads and descriptor strides, but
  could add CPU copy/SSD-miss cost and conflicts with the small cache budget.
  Defer unless a separate cost model demonstrates a useful amortization window.
- **Generic split-K/NT/thread sweeps** are not new: native already splits and
  AR10 includes NT2/routed axis selection. ExLlama's contiguous-K accumulation
  or MLX/LLMEnv `simd_sum` changes canonical chains. Do not directly port it.
- **llama.cpp MMA** really implements multi-tile matrix fragments and vector
  loads; its quantizers and accumulation differ. Single-row AR can waste rows,
  and matrix reduction is not an exact replacement for native FMA chains. It
  is not selected under the current exact state/logit contract.
- **Whole expert gate/up/SwiGLU fusion, resident qkv/z bundling, down finish/mix,
  GDN gate hoist, QSA chains/prefetch** already occur in A/B matrices. New names
  or shape constants alone do not create new mechanisms.
- **MTP, DFlash, P04 argmax, HC L3/L5, canceled DECONE/expert-encoder reuse**
  remain excluded. A drafting TPS claim is not a portable AR gain.

## Proposed bounded implementation order after the issue exists

Start K1 and K4 independently; keep K3 available as a small register-pressure
candidate. K2 needs the explicit element/chain mapping reviewed before source
implementation. Decide K5 support and K6 metadata-only screening before spending
GPU time; do not count either as a large boost. K7 and new expert-record layouts
need a separate mathematical/cost preregistration and are not first-pass work.

Each actual selected experiment needs its own matrix row before source changes:
exact fallback/shape/K/cb/device guards, source/binary hashes, independent finite
full-count oracle, actual-path counter and failure cleanup. Reuse frozen stock
fixtures where the canonical contract still matches, but do not compare two
candidate branches as an oracle. Execute only affected component cases, then
one full-state/continuation/model/production-CLI gate per coherent composition.
Only after quality: one or two representative micro forms and the established
32-generated/31-decode AR protocol with contemporaneous unchanged controls.
If controls drift strongly, record nonconclusive rather than inventing a gain.

These are six distinct implemented upstream mechanisms with different native
adaptations, plus one explicitly derived follow-up. Several have modest or
unknown upside; the research did not find evidence sufficient to promise six
large boosts, let alone to add their unmeasured gains toward11–12 tok/s.
