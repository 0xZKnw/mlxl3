# AR20 — TurboEXL3 and native EXL3 kernel research

Source-only research under `AR20-RESEARCH-20261105`/`9e6eb39d`, above PR132/all twelve AR10 flags ON. Read `opti.md`, AR10 matrix/audit/kernel-bank report, the native NT2 audit, `docs/modeles/qwen/qwen38-decode-local.md`, opt54Q closure and old DEC-ONE/T04 history before proposing repeats. No executable source, model, compiler, proof or benchmark launched. Raw primary source reads were bounded per-file HTTP reads; no repository clone or model download. Only this research note is written in the root worktree.

## Identity and pins

**TurboEXL3 is [davidtai/turboexl3](https://github.com/davidtai/turboexl3)**, an Apple GPU Metal/Zig library, distinct from turboderp's CUDA ExLlamaV3. GitHub default-branch metadata and commit endpoints give `main` **c3bb0e260091709f8ad64c84d1fd02808b12dfd1**, commit time2026-11-02T16:12:47Z. Its own README says it serves Qwen3.8-Flash-Next through mlx-serve and links its kernel/source/result documents. Those first-party sources establish identity; secondary social/aggregator performance claims were not used.

Other fresh HEADs checked2026-11-05: llama.cpp `master` **2bc46baa9207d098d03741ac394e7faaf1a955b6** (10:03:44Z); TensorFold `main` **0311c9280878775ea8167d2e26bb7068d7cebc6f** (10:01:10Z). These advanced from the AR10 pins. The intervening llama changes mostly cover RoPE/metadata/backends, and TensorFold mostly adds Kimi/HIP/tokenizer/server work. No fresh Flash-Next Metal speed result was found in those intervening commits. Their existing expert-axis/dual-input/epilogue examples already motivated A03–A06 and are not counted again.

## What Turbo actually attests

[RESULTS.md189–244](https://github.com/davidtai/turboexl3/blob/c3bb0e260091709f8ad64c84d1fd02808b12dfd1/docs/RESULTS.md#L189-L244) times one routed MoE layer on M5 Max128GB/macOS27.0.1, hidden2560/expert640/top10, BF16/MUL1. Captured-routing K2 B1 form9 is70.7µs versus Sushi72.8µs, ratio1.030×; synthetic-routing K2 B1 is70.8 versus73.3µs,1.035×. This is evidence that the exact target geometry has an implemented optimized kernel, **not** a promise of a large AR model gain on M5/24GB/SSD. Server tables use MTP and are not AR evidence for us. No140tok/s claim is imported.

[KERNELS.md194–218](https://github.com/davidtai/turboexl3/blob/c3bb0e260091709f8ad64c84d1fd02808b12dfd1/docs/KERNELS.md#L194-L218) carefully distinguishes its own form3 arithmetic (within tolerance of format.project) from forms5–9 that preserve form3 bytes. Our exact arithmetic is a third contract. Import mechanisms, not Turbo's reductions, Half placement or tolerance.

## Five concrete candidates

### K1 — exact packed-byte MUL1 sum: missing native port

**Implemented upstream:** [lane/helpers.metal166–178](https://github.com/davidtai/turboexl3/blob/c3bb0e260091709f8ad64c84d1fd02808b12dfd1/src/kernels/lane/helpers.metal#L166-L178) hashes the codeword once, sums two independent byte pairs inside a32-bit integer, then sums the two16-bit lanes before the same Half FMA. EXL3_T1V chooses equivalent integer forms; count these as one mechanism.

**Local gap:** `native/shaders/exl3.metal8–19` still extracts four bytes separately and adds them serially. `flash_projection.metal17` maps all canonical QMV decoders to this helper; `_qmv_tile_kernel.metal112–115` calls it four times per half-lane group. This affects resident K4 and routed K2, including the new expert-axis PSOs. There is no added allocation, dispatch or changed quantization.

**History/new reason:** SWAR is **already enabled in the old MLX engine**, attested in `docs/qwen38-decode-local.md10–38`:65536 states and model states match, short Qwen27B medians4.890→5.635tok/s with battery/drift limits. The direct native code omitted it. This is a missing native port, not a novel algorithm or an extrapolated15% boost.

**Smallest adaptation/oracle:** swap only the integer byte sum for CB2/MUL1, preserve masking/wrapping multiply/1024 bias/Half FMA and every downstream FMA. Full65536-state CPU/original-Metal/ported-Metal bitwise check; then existing independent resident/routed stock words, shared full model+CLI counters before speed. A symbolic integer identity can be checked against the actual Rust helper if one is genuinely used for shared constants; do not rewrite Metal elsewhere and claim it proved. Integer pair bounds prevent16-bit carry. Full codebook LUT and deferred affine map are separate, previously rejected/incompatible paths.

### K2 — eight-codeword64/96-bit reader, above current fixed/axis PSOs

**Implemented upstream:** [lane/helpers.metal60–124](https://github.com/davidtai/turboexl3/blob/c3bb0e260091709f8ad64c84d1fd02808b12dfd1/src/kernels/lane/helpers.metal#L60-L124) loads the lane's ending64-bit stream window from three packed words for K≤6; K7/K8 use96 bits/four words. Codewords then use compile-time shifts. The source retains a slower IntWords reader as an oracle.

**Local gap:** `_qmv_tile_kernel.metal28–37,100–110` computes two four-codeword windows and loads two words for each. K2–K5 fit eight windows in64 bits; the same source body serves fixed resident and expert-axis PSOs. Replacing the reader must leave eight separate accumulator chains and canonical row→group reduction unchanged. No new weight layout or storage.

**History/new reason, explicit:** opt54Q is the **same reader idea**, not new: `opt-54-qmv/closure.json`/opti16001–16028 retain bitexact micro gains0.29–1.51%, rejected under a≥1%every-cell rule, with A/A drift and no model campaign. Native fixed dimensions, routed NT2/expert-axis and measured all12 composition are new execution contexts. Reconsider only as a **bounded native port above this new baseline**, not as a supposedly undiscovered big optimization. If ten wholly new mechanisms are required, do not count this one toward that number.

**Oracle:** CPU integer extraction oracle over each lane, K2–K5 and every stream wrap/shift, random deterministic words plus boundary bit patterns; GPU codec/source-original comparison, independent projection/expert stock outputs. Extend K7/K8 only with their own explicit window bounds; do not silently change generic/reference code. Measure as a compatible small member; no arbitrary isolated significance threshold. More registers/word staging can erase instruction savings.

### K3 — software-pipeline two K tiles without changing accumulation

**Implemented upstream:** [decode/gemv_core.metal62–85](https://github.com/davidtai/turboexl3/blob/c3bb0e260091709f8ad64c84d1fd02808b12dfd1/src/kernels/decode/gemv_core.metal#L62-L85) has EXL3_KU: load the encoded words of two input tiles before decoding either, then process the first tile and second tile in their original order. Its default isOFF; implementation exists, but no isolated published speed number was found. That distinction matters for “attested”.

**Local gap:** `_qmv_tile_kernel.metal52–136` interleaves load→decode→FMA for one `tile_k` before touching the next assigned tile. For canonical SG4, the next tile assigned to the same SIMD group is `tile_k+4`, not `tile_k+1`: preserve this stride and all eight acc chains. Stage only raw words/x values; finish each tile's FMAs in the same sequence. Fixed K/dimensions make this a bounded specialization. No new buffers/dispatches/barriers.

**History:** older dense-Qwen “loop unrolling” trials were rejected in `docs/qwen38-decode-local.md40–43`; opt54Q changed window extraction, not two-tile memory scheduling. New reason is a real upstream staged-load implementation, native fixed/axisPSOs and unchanged-FMA scheduling, not blind unrolling. Validate odd final tile counts and split tails; preserve fallback. Start with one depth2 variant, not a matrix of depths. Register pressure/occupancy may regress; instruction overlap is not measured from source.

### K4 — SIMD/register H128 with our exact radix rounding

**Implemented upstream:** [decb/had128.metal1–2](https://github.com/davidtai/turboexl3/blob/c3bb0e260091709f8ad64c84d1fd02808b12dfd1/src/kernels/decb/had128.metal#L1-L2) and [decode/fin7_4.metal3–5](https://github.com/davidtai/turboexl3/blob/c3bb0e260091709f8ad64c84d1fd02808b12dfd1/src/kernels/decode/fin7_4.metal#L3-L5) hold four values per lane, use five SIMD xor butterflies and two register butterflies. Turbo's current version has its own rounding; it cannot be pasted verbatim.

**Local gap:** `exl3.metal106–119` stores/reloads all128values through threadgroup memory and performs eight CTA barriers (initial+seven stages). Callers include prepare128, finish128, up-finish/SwiGLU, strided preparation, dual GU and dense finish. A one-SIMD helper can reduce this synchronization/storage, with all128values in32lanes×4registers. The target has many hundreds of these per row; no isolated time is yet measured.

**Strict adaptation:** stages1/2/4/8/16 map to lane xor1/2/4/8/16; round every lane's four values to Half **after stage8**; register stage32 combines[v0,v1] and[v2,v3], stage64 combines corresponding pairs, then Half again. Keep initial x×suh Half, pre-Hadamard partial Half, normalization and SVH Half exactly. Signed-zero/subnormal/order properties matter. New helper/32-thread PSOs should coexist with the original64-thread kernels as component oracles; do not alter every caller globally without explicit eligibility/counters.

**History/new reason:** opt54Q only reviewed Hadamard fusion statically. PERF68/F02 established that naive full-FP32/radix2 H128 is wrong; we preserve their exact two boundaries. Native H128 was a correctness port, not this register topology. Test deterministic finite Half extremes, cancellation, signed zero and subnormal corpora against the actual existing Metal helper and independent stock H128, plus full projections/expert/oracle model/CLI. Fewer barriers do not guarantee faster kernels if32threads increase per-lane work/register pressure.

### K5 — fused routed down finish and canonical route reduction

**Implemented upstream:** [decode/dfin7.metal15–37](https://github.com/davidtai/turboexl3/blob/c3bb0e260091709f8ad64c84d1fd02808b12dfd1/src/kernels/decode/dfin7.metal#L15-L37) finishes each selected expert in its own SIMD group, retains the terms in threadgroup memory, then reduces the routes. This changes a compute boundary, unlike A05's storage slab and A06's QMV axis.

**Local gap:** `gpu/projection.rs1339–1360` still launches down.encode_finish once per expert after grouped downQMV. `gpu/moe.rs540–562` binds all ten materialized Half contributions to moe_mix. There are up to480 routed down finishes per target row. Existing A05 replaces allocations only; it did not eliminate these dispatches.

**Bounded adaptation:** preserve ready hot/cold cohorts and their QMV submissions. Write raw Float32 down partials into canonical checked route prefixes of a10×2560slab (102400bytes), retain each complete record/SVH until final command completes, then run one final kernel with ten SIMD groups per128-wide block. Each group executes our exact partial/Half/H128/SVH finish; after one uniform barrier, execute **the actual eight-stripe+two-later-terms reduction of flash_moe.metal40–51**, not Turbo's linear FMA score loop. Shared branch/multiplier and dtype boundaries remain canonical. Potential480finishdispatches removed, not480GPU waits; scope remains one native AR row and bounded extra raw slab storage versus51200-byteHalf slab.

**Costs/risks:** retains raw partials across cohorts instead of reusing one temporary; output raw partial write API/owners/destination offsets must change consistently. It may shift overlap or increase live memory; do not delay QMV to fill all ten. No global arrival counter, uncancellable IO/GPU wait or untracked scratch is required. Kani can check real prefix/destination guards; physical nonempty oracles must cover distinct records, permutations/duplicate IDs, cold/hot subsets, F16/F32 and drop/error recovery, every route term and final mix. Model+CLI before speed. This is the largest new candidate here.

## Reserves and exclusions, not counted as new wins

- **fin7 GU finish→SwiGLU→down prepare** is implemented upstream, but our A03 already fuses up finish/SwiGLU, and old T04 already tested SwiGLU/down.suh (AR−1.64%, retired). Extending through native H128 would be new arithmetic topology with a concrete reason, yet it overlaps K4/A03 and needs separate root selection; do not inflate the count.
- **DEC-ONE last-arriver finish** is implemented and byte-tested to Turbo's chain, but our old POSTP09-DEC-ONE was explicitly abandoned by the user after inconclusive MLX results and extra swap. A native port would be a new context, not an untouched rerun; leave out unless root explicitly chooses it with that history visible.
- **Deferred MUL1 affine map** changes per-weight Half decoding/accumulation and is within tolerance only to the reference; not admissible under our bitexact contract. **Unrotated dense trunk**, changed split-K/quad folds, FP16 dot/matrix accumulations, prefill/multirow kernels and MTP performance claims are likewise not quick AR-exact ports.
- llama.cpp/TensorFold expert-grid, epilogues and dual-input mechanisms are already A03–A06. Their affine/GGUF arithmetic is not EXL3-compatible. No extra candidate is manufactured from an unrelated new HEAD.

## Selection recommendation

K1 and K4 are the strongest practical missing native paths; K5 targets the largest remaining dispatch boundary. K2 is a historical small exact idea worth retaining only with its new native context explicitly recorded. K3 has source implementation but not an isolated upstream measurement, so its attestation is structural/tests rather than a performance guarantee. These five are independent mechanisms; no rate/width/depth variant is another candidate. Root should combine with the other agents' distinct GDN/attention/SSD findings to reach ten genuinely different cases, then preregister the final matrix before prototypes.
