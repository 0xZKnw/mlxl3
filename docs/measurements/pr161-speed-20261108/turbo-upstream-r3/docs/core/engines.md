# The engines

This page lists every [Metal](glossary.md#metal) kernel that turboexl3 registers with [MLX](glossary.md#mlx), by the
name that it passes to `newKernel`. For each [fused path](glossary.md#chain), the page shows the chain of
[launches](glossary.md#launch) that the path replaces. It also tells why the fused path is faster and how its output
bits relate to that chain.

The Metal text of each kernel is a file `<folder>/kernels/<module>/<name>.metal`, beside the Zig file that launches it:
`<folder>` is `src/core` or `src/models/<model>`, and `<module>` is the name of that Zig file without `.zig` (so
`src/core/kernels/decode/`, `src/models/qwen38/kernels/attn_io/`). The steel headers lie directly in `src/core/kernels/`
(`steel_nax.metal`), and the texts of each folder's tests lie in its `kernels/test/`. A Zig file imports each text with `@embedFile`. A kernel can
be several numbered pieces that the Zig file joins, for example `gu3_1.metal` and `gu3_2.metal`.

The file headers (`//!`) and the `///` comments in `src/` hold the full statements; this page is the map. The
[glossary](glossary.md) defines every term, and a link marks the first use of a term on this page. The
[Configuration](api.md#configuration) section of the API page gives the choices of a
[consumer](glossary.md#consumer).

The kernels of the Qwen3.8-only paths (the shared expert and router in the decode launches, the attention projections and the decode-row merges) are described on [Qwen3.8's kernel page](../models/qwen38/kernels.md); every kernel has its own entry in [the kernel catalog](kernels.md) and the models' kernel pages.

## How to read this page

### The three stages of an EXL3 projection

The [format page](format.md#the-three-stages-of-an-exl3-projection) gives them: prepare, inner product, finish.

### Exactness classes

Each kernel is in one of three classes:

| class | what it means | why it holds | kernels in the class |
|---|---|---|---|
| [byte-identical](glossary.md#byte-identical) to a chain | every output element has the bit pattern of that chain's output | the kernel runs the chain's statements, in the chain's order, on the same values; only the launch shape changes | decode [forms](glossary.md#form) 5–9 (to form 3); the heads of the [shared expert](glossary.md#shared-expert) and the [router](glossary.md#router) fold (to the [host](glossary.md#host) chain); the fin fold of the shared expert (to `gu3s` and `fin7s`); [DEC-ONE](glossary.md#dec-one) (to the three-launch chain); the lean GEMV (to `dgemv`); the prefill options that only move work, and the 128-row tiles (to the option set without them); the [resident W_hat](glossary.md#resident-trunk) (to a dequantize per call); `linearSplit` and the `linear` pieces (to `linear`); [ATTN-IN](glossary.md#attn-in-out), ATTN-OUT and the four-member prepare (to the host chain); every decode-row merge (to the host chain); the clamped variants where no value reaches the limit (to the unclamped kernels) |
| [within tolerance](glossary.md#within-tolerance) of `format.project` | the ratio from [`ref.compare`](glossary.md#ref-compare) is at most the stated bar | the kernel rounds where `format.project` rounds, but it sums in its own order, and `f32` addition is not associative | decode forms 1 and 3; the dense GEMV chain; the inner products of the [register path](glossary.md#register-path), the fused GEMM and dequantize + matmul; the routed prefill chain; the clamped routed chains (to `ref.moeRowClamped`). The routed chains also sum in an order other than [Sushi](glossary.md#sushi)'s |
| opt-in, its own rounding | within the stated bars | each weight rounds once to the activation dtype, and each output rounds once | the [un-rotated trunk](glossary.md#unrotated-trunk): `turboexl3_unrot_*` at load, then one plain matmul; the `_std` attention kernels |

A byte test compares bit patterns. A tolerance test uses `ref.compare`: the largest |got − ref| over the output,
divided by the root mean square of the reference output.

The chains do not all compute the sigmoid of SwiGLU with the same function:

| chain | sigmoid of SwiGLU |
|---|---|
| the decode forms and the unfused steps | `metal::exp`, whose precision follows the math mode of the compiler |
| the mid of the routed prefill | `metal::precise::exp` |
| the fused shared expert ([Qwen3.8's section 1](../models/qwen38/kernels.md#1-shared-expert-and-router-inside-the-decode-launches-rows-14)) | the host's sigmoid table, in the output dtype |

So the SwiGLU of one row can round differently at [decode rows](glossary.md#decode-prefill-rows) and at prefill rows.

### Reading the diagrams

Each diagram puts the chain that a path replaces (the unfused side) beside turboexl3's launches (the fused side).

| shape | meaning |
|---|---|
| rectangle | a turboexl3 launch, labeled with its MLX kernel name |
| rounded box | an MLX or host operation |
| stadium (a box with round ends) | a tensor |
| arrow | a data dependency |
| dotted arrow | an alternative path |

A diagram leaves out an arrow that a longer path implies. So the longest path in a diagram is its number of dependent
levels.

## Kernel families

| family | entry points | rows | Zig file (under `src/`) |
|---|---|---|---|
| decode GEMV of the [routed experts](glossary.md#routed-expert) | `moe`, `moeClamped` | 1–16 | `core/decode.zig` |
| the same over three expert banks, one routed pair a row-slot | `moeClampedBanked` | 1–64 pairs | `models/deepseek_v41/banked.zig` on `core/decode.zig` |
| shared expert and router in the decode launches | `moeFusedShared`, `moeFusedSharedRouted` | 1–4 (router fold: 2–4) | `models/qwen38/shared.zig` on `core/decode.zig` |
| prefill GEMM of the routed experts | `moe`, `moeClamped` | 17+ | `core/prefill.zig` |
| dense projections | `linear`, `linearSplit`, the `linear` pieces, `trunkUnrot`, `linearBias` | any | `core/dense.zig` |
| attention projections | `attnIn`, `attnPrep4`, `linearPreparedIdx`, `linearGated`, `attnInStd`, `linearGatedStd` | 17+ | `models/qwen38/attn_io.zig` |
| decode-row merges | `decodeMergeGated`, `decodeMergeQmergeGated`, `decodeMergeQPlanes`, `decodeMergeIndexer` | 1–16 | `models/qwen38/decb.zig` |
| grouped (block-diagonal) linear | `linearGrouped` | 1–16 (17+: `linear` of each slice) | `models/deepseek_v41/grouped.zig` |

---

## 1. Routed experts at decode rows (1–16)

For a call of up to `DECODE_ROWS_MAX` (16) rows, `moe` runs the routed experts of an [MoE](glossary.md#moe) layer as a
chain of GEMV launches. Each [row-slot](glossary.md#row-slot) runs its expert's gate and up projections, SwiGLU and the
down projection. A finish then sums the outputs of each row's slots, weighted by their scores.

**What it replaces.** The unfused chain is `Steps` in `unfused.zig`. For each projection it runs a prepare, a raw GEMV
of the [trellis](glossary.md#trellis) and a finish, and then SwiGLU and the score-weighted reduce. That is 11 launches
in 8 [dependent levels](glossary.md#dependent-level). Its kernels read the same lane plan and decoders as the fused
forms, so a comparison of the two measures fusion alone. No test or benchmark runs it today.

**Why it is faster.** Form 9, the served form, merges the steps that share data:

- One launch prepares both the gate and the up inputs.
- One launch runs the mid: the gate and up finishes, SwiGLU and the prepare of the down input.
- One launch runs the down finish and the score-weighted sum.
- At one row, the prepare moves into the gate and up GEMV launch.

So form 9 runs 4 launches in 4 levels at one row, and 5 in 5 from two rows, against 11 in 8. Its split-K planes put
enough threadgroups in flight at one row. From two rows up, two row-slots that chose one expert share one decode of its
weights.

Compare the depths of the two columns. The unfused chain is 8 levels deep, and form 9 is 4 levels deep at one row and
5 from two rows. The dotted arrow is form 9's path at one row, where `turboexl3_gu8` also runs the prepare.

```mermaid
flowchart TB
  subgraph U["unfused: unfused.zig Steps, 11 launches, 8 levels"]
    direction TB
    ux(["x, slots, scores"]) --> upg["turboexl3_prep<br/>gate input"]
    ux --> upu["turboexl3_prep<br/>up input"]
    upg --> ugg["turboexl3_gemv<br/>gate inner product"]
    upu --> ugu["turboexl3_gemv<br/>up inner product"]
    ugg --> ufg["turboexl3_finish<br/>gate finish"]
    ugu --> ufu["turboexl3_finish<br/>up finish"]
    ufg --> usw["turboexl3_swiglu<br/>SwiGLU"]
    ufu --> usw
    usw --> upd["turboexl3_prep<br/>down input"]
    upd --> ugd["turboexl3_gemv<br/>down inner product"]
    ugd --> ufd["turboexl3_finish<br/>down finish"]
    ufd --> ure["turboexl3_reduce<br/>score-weighted sum"]
    ure --> uy(["y"])
  end
  subgraph F["fused: form 9 (served), 5 launches, 4 at one row"]
    direction TB
    fx(["x, slots, scores"]) --> fp2["turboexl3_prep2<br/>gate + up inputs"]
    fp2 --> fgu["turboexl3_gu3<br/>gate + up split-K planes"]
    fx -.->|one row| fg8["turboexl3_gu8<br/>prepare + planes, one launch"]
    fgu --> ffi["turboexl3_fin7<br/>finish, SwiGLU, down input"]
    fg8 -.-> ffi
    ffi --> fdn["turboexl3_down3<br/>down split-K planes"]
    fdn --> fdf["turboexl3_dfin7<br/>down finish + weighted sum"]
    fdf --> fy(["y"])
  end
```

| kernel | what it computes | launches it replaces |
|---|---|---|
| `turboexl3_prep2` | the gate and up inputs, prepared in [lane order](glossary.md#lane-order), with the block sums of each | `turboexl3_prep` × 2 |
| `turboexl3_gu3` | the gate and up inner products as split-K planes, `f32` `[KS, row-slots, inter]`; from two rows up, two row-slots that chose one expert share one decode of its weights | `turboexl3_gemv` × 2 |
| `turboexl3_gu8` | `prep2` and `gu3` in one launch: each threadgroup prepares its own K span in [threadgroup memory](glossary.md#threadgroup-memory) | `turboexl3_prep` × 2, `turboexl3_gemv` × 2 |
| `turboexl3_fin3` | the planes added in slice order, the deferred map, the gate and up finishes, SwiGLU in `f32`, then the down input prepared, with its block sums | `turboexl3_finish` × 2, `turboexl3_swiglu`, `turboexl3_prep` |
| `turboexl3_fin7` | the statements of `fin3`, in the order of `fin3`, with the H128s in registers (no threadgroup exchanges) | as `fin3` |
| `turboexl3_fin7c` | `fin7` with the clamped SwiGLU (`moeClamped`): its SwiGLU line swapped for the clamp, the template int `LIMIT` | as `fin3` |
| `turboexl3_down3` | the down inner product as split-K planes | `turboexl3_gemv` |
| `turboexl3_dfin3` | for each (block of hidden, row): each slot's planes added, the deferred map and the finish, then an `fma` with the slot's score, in slot order; y in x's dtype | `turboexl3_finish`, `turboexl3_reduce` |
| `turboexl3_dfin7` | `dfin3` with the slots finished side by side (one [simdgroup](glossary.md#simdgroup) each), then the same chain of score `fma`s | as `dfin3` |
| `turboexl3_prep2b`, `turboexl3_gu3b`, `turboexl3_fin7cb`, `turboexl3_down3b`, `turboexl3_dfin3b` | the banked decode (`moeClampedBanked`): `prep2`, `gu3`, `fin7c`, `down3` and `dfin3` with each row-slot's arrays read from one of three banks by its packed id (bank << 24 \| slot row) | as their base kernels |
| `turboexl3_gu5` | form 5: the inputs of an expert group (up to 4 row-slots) prepared in threadgroup memory, then the GEMV of `gu3` for every member from one decode | `turboexl3_prep` × 2, `turboexl3_gemv` × 2 |
| `turboexl3_dn5` | form 5: the mid of `fin3` for every member, the down GEMV in the partition of `down3`, then each slot finished up to its score (`f16`) | `turboexl3_finish` × 3, `turboexl3_swiglu`, `turboexl3_prep`, `turboexl3_gemv` |
| `turboexl3_red5` | form 5: the chain of score `fma`s, in slot order | `turboexl3_reduce` |
| `turboexl3_dn6` | form 6: `dn5`, then the slot that arrives last at its (row, block of hidden) runs the chain of `red5`; an integer [counter](glossary.md#arrival-counter) counts the arrivals | as `dn5`, plus `turboexl3_reduce` |
| `turboexl3_gu` | form 1: the inputs prepared in threadgroup memory, the gate and up GEMV, the finishes and SwiGLU; h in `f32` | `turboexl3_prep` × 2, `turboexl3_gemv` × 2, `turboexl3_finish` × 2, `turboexl3_swiglu` |
| `turboexl3_down` | form 1: for each (block of hidden, row), each slot's down prepare, GEMV, finish and score `fma`, in slot order | `turboexl3_prep`, `turboexl3_gemv`, `turboexl3_finish`, `turboexl3_reduce` |
| `turboexl3_prep`, `turboexl3_gemv`, `turboexl3_finish`, `turboexl3_swiglu`, `turboexl3_reduce` | the unfused steps (`unfused.Steps`) | — |

| form | launches | levels | kernels | output bits |
|---|---|---|---|---|
| unfused (`unfused.Steps`) | 11 | 8 | `prep` × 2, `gemv` × 2, `finish` × 2, `swiglu`, `prep`, `gemv`, `finish`, `reduce` | the arithmetic of form 1 as separate steps; no test runs it |
| 1 (`Kernels`) | 2 | 2 | `gu` → `down` | its own order; `moe` cannot select it, but the GPU parity test runs it |
| 3 | 5 | 5 | `prep2` → `gu3` → `fin3` → `down3` → `dfin3` | its own split-K order: the bytes that forms 5–9 keep |
| 5 | 3 | 3 | `gu5` → `dn5` → `red5` | form 3's |
| 6 | 2 | 2 | `gu5` → `dn6` | form 3's |
| 7 | 5 | 5 | `prep2` → `gu3` → `fin7` → `down3` → `dfin7` | form 3's |
| 8 | 4 | 4 | `gu8` → `fin7` → `down3` → `dfin7` (form 7 where `gu8` is not admitted) | form 3's |
| **9 (served)** | 4 at one row, 5 from two | 4 / 5 | one row: `gu8` → `fin7` → `down3` → `dfin7`; 2–16 rows: `prep2` → `gu3` → `fin7` → `down3` → `dfin7` | form 3's |

**The choice of form.** The field `decode_form` of `Config` picks the form, and the default is 9. At one row
(`gu8_rows`), form 9 uses `turboexl3_gu8` when a threadgroup's K span is 128-aligned (`gu8Admitted`). It stops at one
row for a reason: `gu8` rotates its whole K span again in each threadgroup of output tiles. That cost grows once a
threadgroup carries several slices. Form 9 has its own launch shape (`bt1`, `bt9`, `kdiv9`: the output tiles and K
slices per threadgroup), and that launch shape changes no sum.

**Exactness.**

- Forms 1 and 3 are within tolerance of `format.project`. They round where it rounds, but they sum in their own order:
  - Form 1 splits the input dimension four ways in each threadgroup.
  - Form 3 splits it into a partition that is fixed per geometry. Gate and up split into KS slices × 2 parts, and down
    splits into KS2 slices × 4 parts. It adds the planes in slice order, and with MUL1 it applies the deferred map.
  - In the served setting, each partition starts at 4 slices and halves until its parts divide the input tiles
    evenly. Each down part keeps at least 2 tiles. At 2560 × 640, KS is 4 (20 input tiles per part) and KS2 is 2 (5
    tiles per part).
  - The bars: form 1 is within 5e-3 in `f32` ([rates](glossary.md#rate) n 32–128, MUL1 and [MCG](glossary.md#mcg), a
    12-bit window). `root.moe` (form 9, `bf16` x and y) is within 1.5e-2.
- The bytes of a form-3 row do not depend on how many rows share the call. The partition belongs to the geometry
  alone. The row count sets only how many threadgroups share the slices of a row-slot. The GPU test runs each row alone
  and then rows 2–16 together. It groups row-slots by expert and runs both MUL1 decodes (the deferred map on and off).
  This matters for [MTP](glossary.md#mtp): the number of verify rows changes, but a row's output must not.
- Forms 5–9 are byte-identical to form 3. They keep its partition and its order of FMAs per lane (input tiles in order,
  weights m = 0..7). They also keep its quad folds, its part sums and its [rounding points](glossary.md#rounding-point).
  Only the launch shape moves:
  - the tiles and slices per threadgroup;
  - the expert groups;
  - the launch that runs the prepare;
  - H128 in registers or through threadgroup memory.
- The GPU tests compare the bits of forms 5–9 with form 3 at rows 1–5, 8 and 16. They use several settings of tiles,
  K divisors and grouping, and both MUL1 decodes. Forms 5 and 6 run in `f32`; forms 7–9 run in `f32` and `bf16`.
- Every form computes SwiGLU in `f32` with `metal::exp`, as the unfused `turboexl3_swiglu` does. The prefill mid uses
  `metal::precise::exp` ([section 2](#2-routed-experts-at-prefill-rows-17)).

**The clamped SwiGLU.** `moeClamped` is `moe` with DeepSeek-V4's (and GLM's) `swiglu_limit` L: before SwiGLU, each
row-slot's gate is clamped from above at L and its up to [−L, L], in `f32` on the finished `f16` values (an integer L is
exact in `f16`). L is part of the geometry (`decode.Geometry.limit`), so a clamped call has its own cached kernels; only
fin7 changes: `turboexl3_fin7c` is `fin7` with its SwiGLU line (`fin7_swiglu.metal`) swapped for the clamp
(`fin7_swiglu_clamp.metal`), and its launch passes `LIMIT` after fin7's constants. Every other kernel of a clamped call is
the unclamped chain's, and every unclamped kernel's text is unchanged. A clamped call runs forms 7, 8 and 9 as
configured, and form 9 under forms 3, 5 and 6 (their bytes); the fused shared expert ([Qwen3.8's section 1](../models/qwen38/kernels.md#1-shared-expert-and-router-inside-the-decode-launches-rows-14)) has no clamp and refuses
a clamped geometry. The routing weight multiplies the down projection's `f16` output in the `f32` sum, as in `moe`;
DeepSeek's reference scales the activation before w2 instead, the same product up to rounding.

**The banked decode.** `moeClampedBanked` serves an expert streamer whose experts sit in three slot banks (mlx-serve's
streamed DeepSeek-V4.1 bank: base, ext, transient, each its own arrays and slot count). A call is up to 64 routed pairs,
each one [row-slot](glossary.md#row-slot) at top-1 with its id packed (bank << 24 | slot row), and it returns each pair's
output at weight 1 (`f32` `[pairs, hidden]`); the caller weighs and sums them. It runs form 7's launches with form 9's
launch shape at that many row-slots: `turboexl3_prep2b`, `turboexl3_gu3b`, `turboexl3_fin7cb`, `turboexl3_down3b` and
`turboexl3_dfin3b` (top-1, unit scores). Each is its base text with the lines that read a bank's arrays swapped for lines
that read the row-slot's bank: the expert line of `prep2`, `fin7c` and `dfin3` becomes the slot row and the bank's suh /
svh pointers (`prep2b_expert.metal`, `fin7b_expert.metal`, `dfin3b_expert.metal`), and the weight base of `gu3` and
`down3` becomes the bank's trellis at the slot row (`gu3b_w.metal`, `down3b_w.metal`). The GEMVs' members compare the
packed ids, so two row-slots share a decode only on one bank's one slot. Every other statement is the base text's, so a
pair's bytes are `moeClamped`'s on its bank; the GPU test (`banked_test.zig`) holds every pair to them, byte for byte,
and sampled pairs to the clamped reference. Each bank's arrays are separate inputs: at most 14 inputs and 2 outputs a
launch. The trellis may be `I16` (the streamer's code arrays: the kernels read its words). `moeClampedBankedGateUp` and
`moeClampedBankedDown` split the call after `fin7cb` (the down's prepared input and its block sums). `fin7cb` reads each
bank's down suh, so the gate|up half needs the down's suh in place as well as the gate and up arrays: a caller whose
arrays land in parts (an expert streamer's gate and up, then its down) runs the whole call at its down.

## 2. Routed experts at prefill rows (17+)

Above 16 rows, `moe` runs a GEMM chain. The chain sorts the row-slots by expert and prepares the gate and up inputs in
sorted order. Then it runs one GEMM launch for gate and up, a mid launch, the down GEMM and a finish. The finish sums
each row's slots in slot order.

**Why it is faster.** The GEMM decodes the trellis inside its weight loader. For each 64-deep stage of the input
dimension, each simdgroup decodes its share of the stage's 16 trellis [tiles](glossary.md#tile) into threadgroup memory.
Then the [NAX](glossary.md#nax) units multiply 64 or 128 rows by that stage, with `f32` accumulation. No `f16`
[W_hat](glossary.md#w-hat) reaches device memory, so the GEMM reads K bits per weight and never writes or reads a 16-bit
copy. Dequantize + matmul ([section 3](#3-dense-projections)) is the unfused form of one such GEMM.

**The two sides of the diagram.** This repository has no unfused chain for the routed prefill, and it does not hold
Sushi's. So the left side is the base option set (`prefill.Options{}`), and the right side is the served set
(`root.pfOptions`).

Both sides are six levels deep. On the right, one route launch replaces MLX's argsort and the table, and each GEMM gets
a launch of 128-row tiles beside its mixed launch. The two launches of a pair write separate outputs, so they share a
level.

```mermaid
flowchart TB
  subgraph U["unfused: base option set, 7 launches, 6 levels"]
    direction TB
    ui(["x, expert ids"]) --> uas("MLX argsort<br/>sorted order")
    uas --> ut["turboexl3_pf_table<br/>tile table"]
    uas --> up["turboexl3_pf_prep<br/>gate + up inputs, sorted"]
    ut --> ug["turboexl3_pf_gemm2<br/>gate + up GEMM"]
    up --> ug
    ug --> um["turboexl3_pf_mid<br/>finish, SwiGLU, down input"]
    um --> ud["turboexl3_pf_gemm1<br/>down GEMM"]
    ud --> uf["turboexl3_pf_fin<br/>down finish + slot sum"]
    uf --> uy(["y"])
  end
  subgraph F["fused: served option set, 8 launches, 6 levels"]
    direction TB
    fi(["x, expert ids"]) --> fr["turboexl3_pf_t128_route<br/>count, scan, scatter, tables"]
    fr --> fp["turboexl3_pf_prep<br/>gate + up inputs, sorted"]
    fp --> fgm["turboexl3_pf_t128_gum<br/>gate + up, mixed tiles"]
    fp --> fgt["turboexl3_pf_t128_gut<br/>gate + up, 128-row tiles"]
    fgm --> fm["turboexl3_pf_t128_mid<br/>finish, SwiGLU, down input"]
    fgt --> fm
    fm --> fdm["turboexl3_pf_t128_dnm<br/>down, mixed tiles"]
    fm --> fdt["turboexl3_pf_t128_dnt<br/>down, 128-row tiles"]
    fdm --> ff["turboexl3_pf_t128_fin<br/>down finish + slot sum"]
    fdt --> ff
    ff --> fy(["y"])
  end
```

The base chain already fuses. The decode sits in the GEMM's loader, and gate and up share one GEMM launch. `mid` merges
the gate and up finishes, SwiGLU and the down prepare. `fin` merges the down finish and the slot sum. The served set
adds these options:

- 64-row tiles, with remainders of at most 32 rows on the register path in the same launch (`small = .mixed`);
- a double-buffered weight stage in NAX [fragment](glossary.md#mma) order, with unrolled MMA sub-steps;
- the integer-rate reader where n % 16 == 0, in the register path too (`sintr`);
- the deferred MUL1 map;
- four output blocks per threadgroup of the down GEMM (`ncb`);
- the route in one launch (`froute`);
- A layout 2;
- the 128-row tiles (`t128`).

With `froute` and without `t128`, the chain is 6 launches in 6 levels. `root.pfOptions` builds the served set from
constants, except `alay` and `t128`: the fields `pf_alay` and `pf_t128` of `Config` set those two. With `prefill` or
`nax` false, `moe` refuses prefill rows. The third column below compares each kernel with the base chain.

| kernel | what it computes | launches it replaces |
|---|---|---|
| `turboexl3_pf_table`, `turboexl3_pf_table_m` | after MLX's argsort: each expert's run by binary search, an exclusive scan, then the tile table (`_m`: remainders of at most 32 rows as register-path tiles) | — |
| `turboexl3_pf_froute`, `turboexl3_pf_froute_m` | each expert's slots counted (threadgroup atomics), scanned and scattered: `order` and the tile table in one launch | MLX argsort, the table |
| `turboexl3_pf_t128_route` | the count and scatter of `froute`; 128-row tiles for each expert's full runs, and for a remainder of more than `t128_cut` (64) rows; the mixed table over the rest; each expert's first remainder position | MLX argsort, the table |
| `turboexl3_pf_prep` | the gate and up inputs, prepared in sorted order (the A layout of `alay`), `inv` and the block sums | — |
| `turboexl3_pf_gemm2`, `turboexl3_pf_gemm1` | the gate + up GEMM (two operations) and the down GEMM, in the [shared-decode body](glossary.md#shared-decode-body): the weight stage decoded into threadgroup memory, then NAX MMAs; `bk128` and `ncb` reshape the down launch | — |
| `turboexl3_pf_gemm2s`, `turboexl3_pf_gemm1s` | the register-path body (`small = .all`): every tile at most 32 rows; each simdgroup an independent 32 × 32 item that decodes straight into its B fragment registers, with no threadgroup memory | `_gemm2` (`_gemm1`) |
| `turboexl3_pf_gemm2m`, `turboexl3_pf_gemm1m` | the mixed launch: the table's large tiles through the shared-decode body, and its tiles of at most 32 rows through the register path | `_gemm2` and `_gemm2s` (`_gemm1` and `_gemm1s`) as one launch |
| `turboexl3_pf_t128_gum`, `turboexl3_pf_t128_dnm` | the mixed launch over the remainder table | `_gemm2m` (`_gemm1m`), with `_gut` (`_dnt`) beside it |
| `turboexl3_pf_t128_gut`, `turboexl3_pf_t128_dnt` | the shared-decode body at 128 rows over the 128-row tiles, beside the mixed launch; each launch writes its own output | as above |
| `turboexl3_pf_mid`, `turboexl3_pf_t128_mid` | the gate and up finishes, SwiGLU in `f32`, the down input prepared, and its block sums (`t128`: each row from the launch that computed it) | — |
| `turboexl3_pf_fin`, `turboexl3_pf_t128_fin` | the down finish and the score-weighted sum of the slots, in slot order, through `inv` (`t128`: each row from the launch that computed it) | — |
| `turboexl3_pf_prep_v`, `turboexl3_pf_mid_v`, `turboexl3_pf_fin_v` | `aux`: prep, mid and finish with four consecutive elements per lane and several simdgroups per threadgroup | `_prep`, `_mid`, `_fin` |
| `turboexl3_pf_midc`, `turboexl3_pf_t128_midc`, `turboexl3_pf_mid_vc` | the three mids with the clamped SwiGLU (`moeClamped`): each mid's SwiGLU line swapped for the clamp, the template int `LIMIT` | `_mid`, `_t128_mid`, `_mid_v` |

The suffixes:

- `s`: the register path alone;
- `m`: mixed, both bodies in one launch;
- `v`: the vector layout of prep, mid and finish.

The A layouts (`alay`):

- 0: natural order;
- 1: each 32-element block in the slot order of the A fragments, so one 16-byte load per fragment row and 32-deep step;
- 2: each 16-element block in that order, so one 8-byte load per fragment row and 16-deep slice (served).

`half_rows` lets a simdgroup whose rows end within its first 16 multiply only its first fragment row.

**Exactness.**

- The rounding points are those of `format.project`. The GEMMs sum in the order of the NAX (16-deep MMAs, K ascending)
  and apply the deferred map, so the chain is within tolerance. In `f32`, y is within 5e-3 on sampled rows. In `bf16`,
  y is within 1.5 × the error of Sushi's prefill path + 2e-3, where the host exports Sushi.
- Some options only move loads, MMAs or work. Each of them gives the bytes of the same option set without it. They
  are `bk128`, `ncb`, `sintr`, `froute`, `aux` (with the block sums in the strided order), `alay` and `half_rows`. A
  GPU test compares the bits.
- `moeT128` gives the bytes of `moe`, for the whole call and stage by stage. The reason: bm 128 and bm 64 run the same
  K loop per simdgroup.
- A row's bytes do not depend on where its slots land in the sorted order or in the tiles. So the arrival-order scatter
  of `froute` changes no output. The option `aux_lane_sums` (off) sums the block sums in the lane's order, so its
  outputs can differ in the last bits.
- The mid computes SwiGLU in `f32` with `metal::precise::exp`. The decode forms use `metal::exp`, so the SwiGLU of
  one row can round differently at prefill rows.
- `moeClamped` runs this chain with the clamp in the mid alone (`prefill.Geometry.limit`; [section 1](#1-routed-experts-at-decode-rows-116) gives the clamp): the
  clamped mids (`_midc`, `_t128_midc`, `_mid_vc`) are their mids with the one line `mid_swiglu.metal` swapped for
  `mid_swiglu_clamp.metal`, and their launches pass `LIMIT` after the mid's constants. Where no value reaches the limit they give their mids' bytes (a GPU test
  compares the bits at limit 128). A second GPU test holds `moeClamped` at DeepSeek-V4.1-Flash's geometry (K2, K3, K6;
  decode rows 1, 4, 16, prefill rows 32, 1328) to 5e-3 of `ref.moeRowClamped` (`f32` through `ref.compare`; `bf16`
  through `ref.compareRounded`, which first allows half a `bf16` ulp of each output's own rounding), on inputs that
  clamp about a tenth of the gates and a quarter of the ups.

## 3. Dense projections

`linear` picks a path by the row count (`Dense.pathFor`):

| rows | path | launches | levels |
|---|---|---|---|
| 1–4 (DEC-ONE's row cap) | DEC-ONE: one GEMV launch; the threadgroup that arrives last at its (output block, [row chunk](glossary.md#row-chunk)) runs the finish. The prepare runs inside that launch at 1 row, or at up to 1024 outputs, where its buffer fits; otherwise `prep_sc` runs it in a launch of its own (both thresholds are constants) | 2 | 1 with the prepare inside, else 2 |
| 5–16, or `dec_one` false | `prep_s` → multi-row `dgemv` (`dgemv_lean` for chunks of 6 to 8 rows) → `dfinish` | 3 | 3 |
| 17–32 (`small_rows`; 0 when `nax` is false) | `prep` → register-path GEMM `rs_gemm` → `finish` | 3 | 3 |
| 33+ | `prep` and `dequant` → MLX matmul → `finish` | 4 | 3 |
| 33+, resident W_hat | `prep` → MLX matmul → `finish`; no `dequant` after the first call | 3 | 3 |
| 33+, un-rotated | one MLX matmul by W_std | 1 | 1 |
| 17+, the fused GEMM (only the tests select it: `Dense.fused_prefill`) | `prep_s` (`prep` for MCG) → `pf_gemm` → `finish`; a call of more than 512 tiles of 64 rows takes dequantize + matmul | 3 | 3 |

On the right, the GEMV launch also runs the finish, and in the upper box it also runs the prepare.
`turboexl3_dense_dec_zero` reads only `svh`, a weight. So it waits on nothing and runs beside the previous operation.

```mermaid
flowchart TB
  subgraph U["unfused: the three-launch chain, 3 levels"]
    direction TB
    ux(["x"]) --> up["turboexl3_dense_prep_s<br/>prepare + block sums"]
    up --> ug["turboexl3_dense_dgemv<br/>multi-row split-K GEMV"]
    ug --> uf["turboexl3_dense_dfinish<br/>planes, finish"]
    uf --> uy(["y"])
  end
  subgraph F["fused: DEC-ONE, one GEMV launch"]
    direction TB
    subgraph F1["prepare inside"]
      direction TB
      asv(["svh"]) --> az["turboexl3_dense_dec_zero<br/>zero arrival counters"]
      ax(["x"]) --> a1["turboexl3_dense_dec1<br/>prepare, GEMV, last-arriver finish"]
      az --> a1
      a1 --> ay(["y"])
    end
    subgraph F2["prepare in its own launch"]
      direction TB
      bx(["x"]) --> b1["turboexl3_dense_prep_sc<br/>prepare, zero counters"]
      b1 --> b2["turboexl3_dense_dec0<br/>GEMV, last-arriver finish"]
      b2 --> by(["y"])
    end
  end
```

Both fused boxes decode the trellis inside the GEMM, so no `f16` W_hat is written. The register path serves 17–32 rows
where the [NAX gate](glossary.md#nax-gate) holds. Only the tests select the fused GEMM.

```mermaid
flowchart TB
  subgraph U["unfused: dequantize + matmul, 4 launches, 3 levels"]
    direction TB
    ux(["x"]) --> up["turboexl3_dense_prep<br/>prepare, natural order"]
    ut(["trellis"]) --> ud["turboexl3_dense_dequant<br/>f16 W_hat"]
    up --> um("MLX matmul<br/>f16 inner product")
    ud --> um
    um --> uf["turboexl3_dense_finish<br/>finish"]
    uf --> uy(["y"])
  end
  subgraph F["fused"]
    direction TB
    subgraph F1["17-32 rows: register path"]
      direction TB
      ax(["x"]) --> ap["turboexl3_dense_prep"]
      ap --> ar["turboexl3_dense_rs_gemm<br/>decode into B registers"]
      ar --> af["turboexl3_dense_finish"]
      af --> ay(["y"])
    end
    subgraph F2["17+ rows, fused GEMM (tests only)"]
      direction TB
      bx(["x"]) --> bp["turboexl3_dense_prep_s"]
      bp --> bg["turboexl3_dense_pf_gemm<br/>decode per K stage"]
      bg --> bf["turboexl3_dense_finish"]
      bf --> by(["y"])
    end
  end
```

The resident box decodes the trellis once and keeps W_hat, so each later call has three launches. The un-rotated box
runs its decode and both un-rotation passes at load, so each call is one matmul.

```mermaid
flowchart TB
  subgraph U["unfused: dequantize per call, 4 launches per call"]
    direction TB
    ux(["x"]) --> up["turboexl3_dense_prep"]
    ut(["trellis"]) --> ud["turboexl3_dense_dequant<br/>every call"]
    up --> um("MLX matmul")
    ud --> um
    um --> uf["turboexl3_dense_finish"]
    uf --> uy(["y"])
  end
  subgraph F["fused: the trunk kept resident"]
    direction TB
    subgraph F1["resident W_hat, 3 launches per call"]
      direction TB
      at(["trellis"]) --> ad["turboexl3_dense_dequant<br/>first call only"]
      ad --> aw(["kept W_hat"])
      ax(["x"]) --> ap["turboexl3_dense_prep"]
      ap --> am("MLX matmul")
      aw --> am
      am --> af["turboexl3_dense_finish"]
      af --> ay(["y"])
    end
    subgraph F2["un-rotated, opt-in: 1 launch per call"]
      direction TB
      bt(["trellis, suh, svh"]) --> bd["turboexl3_dense_dequant<br/>at load"]
      bd --> bc["turboexl3_unrot_cols<br/>row blocks, f32"]
      bc --> br["turboexl3_unrot_rows<br/>column blocks, scales, bf16"]
      br --> bw(["W_std"])
      bx(["x"]) --> bm("MLX matmul<br/>x W_std")
      bw --> bm
      bm --> by(["y"])
    end
  end
```

| kernel | what it computes | launches it replaces |
|---|---|---|
| `turboexl3_dense_prep_s` | the prepare in lane order, and each 128-block's input sum | — |
| `turboexl3_dense_dgemv` | the multi-row GEMV: each weight decoded once per threadgroup for the rows of one chunk, at most 5 (`max_rb`, a constant), with split-K planes over a K split fixed per shape (`ksFor`) | the row-slot GEMV, which decodes once per row |
| `turboexl3_dense_dgemv_lean` | the lean GEMV: `dgemv` with its codeword loop taken by input element. Each accumulator gets the same FMAs in the same order, and a lane keeps one input per row at a time, so a chunk can have up to 8 rows (`lean_rb`). The chain runs it for chunks of 6 to 8 rows (`Config.dense_lean`) | `dgemv` over two chunks of the same rows, which decodes each weight twice |
| `turboexl3_dense_dfinish` | the planes added in slice order, the deferred map, the finish, and the output dtype | — |
| `turboexl3_dense_dec1` | DEC-ONE: the threadgroup's K slice prepared in threadgroup memory, the planes of `dgemv`, and `dfinish` in the threadgroup that arrives last (an integer counter) | `prep_s`, `dgemv`, `dfinish` |
| `turboexl3_dense_dec0` | DEC-ONE over the rows that `prep_sc` prepared | `dgemv`, `dfinish` |
| `turboexl3_dense_prep_sc` | `prep_s`, and it zeroes the arrival counters of DEC-ONE | `prep_s` |
| `turboexl3_dense_dec_zero` | zeroes the counters for `dec1` | — |
| `turboexl3_dense_prep` | the prepare without block sums: natural order (register path, matmul) or lane order (the row-slot variant) | — |
| `turboexl3_dense_gemv` | the row-slot GEMV (`unfused.text_gemv`; the row-slot variant, which only the tests select): every row decodes the trellis again | — |
| `turboexl3_dense_finish` | the finish of an `f32` or `f16` inner product, then the output dtype | — |
| `turboexl3_dense_rs_gemm` | 17–32 rows: the register-path GEMM of the routed prefill with one operation and 32-row tiles; the trellis decoded straight into the B fragment registers (`f16` weight values), `f32` accumulation, `f16` out | `turboexl3_dense_dequant`, MLX matmul |
| `turboexl3_dense_dequant` | the `f16` W_hat `[in, out]` from the trellis | — |
| `turboexl3_dense_pf_gemm` | tests only: the shared-decode GEMM of the routed prefill over 64-row tiles of the one "expert", with the deferred map | `turboexl3_dense_dequant`, MLX matmul |
| `turboexl3_dense_finish_pieces` | one finish over a call cut into 2–4 row pieces (the host's overlap of the [GDN](glossary.md#gdn) layers) | — |
| `turboexl3_unrot_cols` | at load: the ±1 H128 along the 128-column blocks of each row of W_hat, into `f32` | — |
| `turboexl3_unrot_rows` | at load: the ±1 H128 along the 128-row blocks of each column, × 1/128 × [suh](glossary.md#suh-svh) × svh, and one rounding to `bf16` or `f16`: `W_std = diag(suh) H W_hat H diag(svh)` | for each call: `prep`, `dequant`, MLX matmul, `finish` |

**The row chunks of the GEMV chain.** The multi-row GEMV decodes each weight once per threadgroup, for the rows of one
chunk. The chain cuts a call into the fewest chunks of at most 8 rows, and it spreads the rows evenly over them
(`Dense.chainRb`):

| rows | chunks | rows per chunk | kernel |
|---|---|---|---|
| 1–5 | 1 | all | `dgemv` |
| 6–8 | 1 | all | `dgemv_lean` |
| 9–10 | 2 | 5 | `dgemv` |
| 11–16 | 2 | 6 to 8 (half the rows, rounded up) | `dgemv_lean` |

`dgemv` keeps four input floats per row live through its codeword loop. Past 5 rows a lane needs more than 32 floats
and spills registers, so a chunk on `dgemv` has at most 5 rows. DEC-ONE keeps chunks of at most 5 rows at every row
count (`Dense.rbFor`), because its GEMV loop is `dgemv`'s. The decode-row merges ([Qwen3.8's section 3](../models/qwen38/kernels.md#3-decode-row-merges-rows-116)) take DEC-ONE's chunks where
DEC-ONE takes the call, and the chain's chunks otherwise. With `dense_lean` false, the chain also keeps chunks of at
most 5 rows, all on `dgemv`: 6 rows run as 3 + 3, 7 as 4 + 3, 8 as 4 + 4 and 16 as 4 × 4.

**The pieces of `linear`.** `linearInnerSplit`, `linearPrepared`, `linearFinish` and `linearPrep` serve the host's
fused norm-gate of the GDN layers, which runs between them. `linearPiecesBegin`, `linearPieceInner`,
`linearPiecePrepared` and `linearPiecesFinish` serve row pieces. Each piece is one of `linear`'s own launches with
`linear`'s configs, split where the host puts its own launches in between.

**Joins of the un-rotated trunk.** `trunkUnrot` joins up to 4 projections that read one x into one W_std, with an MLX
concatenate. The members are column views of that W_std. `trunkUnrot` keeps a group rotated when its W_std is not
finite.

**Exactness.**

- The multi-row GEMV chain sums in its own order: K slices fixed per shape, four parts per slice, and planes in slice
  order. It applies the deferred map. It is within tolerance of `format.project` per row:
  - within 5e-3 with `f32` x and y on the [trunk](glossary.md#trunk)'s shapes. The test uses GEMV rows (1–16),
    register-path rows (17, 31, 32) and prefill rows (40, 129, 200);
  - within 2.5e-2 with `bf16` x and y (4 and 40 rows).
- A row's bytes are the same alone and among 2–16 rows, whatever the chunks and the GEMV text.
- `dgemv_lean` is byte-identical to `dgemv`. Each accumulator gets the FMAs of `dgemv`, in the order of `dgemv`, on the
  same operands. A CPU test holds its text to `dgemv`'s, with only the input loads and the codeword loop replaced. The
  GPU tests run the chain with `dense_lean` on and off at 1–9, 11 and 16 rows, and compare each row with the same row
  alone.
- DEC-ONE is byte-identical to the three-launch chain. Its lines that set values are those of `dgemv`, `prep_s` and
  `dfinish`, and a CPU test holds them equal. The planes are still added in slice order. An integer counter counts the
  arrivals; there are no float atomics. The GPU test covers the trunk's shapes at rows 1–9, 11 and 16, and both places
  of the prepare. It also covers the MUL1 deferral on and off, `bf16` and `f32`, one MCG shape, and each row alone.
- The register path, the fused GEMM and MLX's matmul sum in their own MMA orders. The same bars hold the register path
  and dequantize + matmul to `format.project`; the fused GEMM (tests only) is held only to dequantize + matmul, by the
  dense timing test's bars (RMS difference at most 3.9e-3, largest ratio at most 8e-2). With `nax` false, 17–32 rows take dequantize + MLX's matmul, so
  their bytes can differ from those of the register path.
- `linearSplit` records the same launches on two streams, so it gives the bytes of `linear`. The GPU test runs 8192,
  1328 and 40 rows on both inner products.
- The row pieces give the bytes of `linear` over the concatenated rows. This holds wherever MLX's matmul runs the whole
  call's kernel at the row count of every piece. The caller checks that condition.
- The resident W_hat is the output of the per-call `dequant`, kept. Each weight comes from its own tile with no
  reduction, and nothing writes W_hat after that. So every call gives the bytes of the per-call path. The GPU test
  covers every kind of trunk projection at 33–8703 rows, both streams, and ATTN-IN and ATTN-OUT.
- The un-rotated trunk is a different rounding of the same linear map. W_std rounds once per weight to the activation
  dtype, and x · W_std rounds once per output. The chain instead rounds x's rotated copy, the inner product and the
  finish to `f16`. The bars:
  - each element of W_std is within one `bf16` step of an `f64` reference. Where its terms cancel to near zero, the
    bar is wider. It is half a `bf16` step of the element's own value, plus 16 `f32` epsilons of its terms' absolute
    sum;
  - at most 1 element in 1000 differs from the reference's own `bf16` rounding;
  - a call's y is within a relative RMS of 2^-7.5 of the chain. Its largest |difference| is at most 2 `bf16` steps at
    the binade of max |chain|;
  - a CPU test also holds the plain matmul within twice the chain's own distance from the exact linear map, plus 2^-10.
- Calls of up to 32 rows never read the table of the un-rotated trunk, so they keep the bytes of the trellis paths.

**Grouped linears (`linearGrouped`, `grouped.zig`).** DeepSeek-V4.1's attention `wo_a` is block-diagonal: x
`[rows, G, in]` meets G slices, each a linear in → out, and y is `[rows, G, out]` (G 8, 4096 → 1024). The slices come
stacked: trellis `[G, in/16, out/16, n]`, suh `[G, in]`, svh `[G, out]`.

| rows | path | launches | levels |
|---|---|---|---|
| 1–16 | `turboexl3_grouped_prep` → `turboexl3_grouped_gemv_d` (`_gemv_dl` for chunks past 5 rows; the routed `turboexl3_grouped_gemv` when `Grouped.dense_gemv` is false, the tests' reference arm) → `turboexl3_grouped_finish`, every slice in each launch | 3 | 3 |
| 17+ | `linear` of each slice: a copy of its columns of x, then `linear`'s launches; then one MLX concatenate (G copies) | 4 per slice at 17–32, 5 from 33, + G | 5 |

The decode kernels are existing texts with their own template arguments (and `gemv_d`'s head and store pieces):

| kernel | text | what it computes |
|---|---|---|
| `turboexl3_grouped_prep` | `dense.text_prep_s` | the prepare in lane order and the block sums, over each row's G slices as one row of G · in values; every 128-block lies inside one slice, so each slice's values are dense's |
| `turboexl3_grouped_gemv_d`, `_gemv_dl` | `grouped.text_gemv_d`, `_dl` | the default step 2: dense's multi-row GEMV statements (its lean codeword loop for `_dl`) over the slices, one decode per row chunk, into the routed planes |
| `turboexl3_grouped_gemv` | `decode.text_down3` | the gather of the routed down GEMV: row-slot r · G + g reads its own prepared row and slice g's trellis, into split-K planes; from two rows up, the row-slots of one slice share one decode (`GROUP` 2) |
| `turboexl3_grouped_finish` | `decode.text_dfin3` | the routed down finish at top-1 with every score 1, so no sum over slots: the planes in slice order, the deferred map, the finish and the output dtype |

The K split is `Dense.ksFor(in, out)`, dense's split for one slice. `dfin7` would not do: its store covers
32 · TOPK outputs of each 128-block, all of them only from TOPK 4.

**Biases (`linearBias`).** The vision tower's q, k, v and wo, and the aligner's w1 and w2, carry a bias `[out]`.
`linearBias` is `linear`, then one MLX Add of the bias in x's dtype; the host converts the pack's `f16` bias once, at
load. The add is not in the finish: `linear`'s finish is a different kernel on each row path, and the un-rotated
trunk's matmul has none.

- Each slice of `linearGrouped` has the values of `linear` on it. From 17 rows it is `linear`, so the bytes are
  equal. At decode rows each launch runs the dense chain's statements in its order, on the same K split. So the values
  are equal; only the sign of an exact zero can differ, because the finish adds a +0 score product. The GPU test
  checks every slice against `linear` at 1, 4, 16, 32 and 512 rows. It also holds the output to `format.project`
  within the dense bars.
- `linearBias` rounds twice: `linear`'s output, then the sum in x's dtype.

## 4. Building blocks

- **`lane.zig`, the codeword readers.** Lane L of a simdgroup owns code positions 8L to 8L + 7 of a 256-weight tile (the
  [tensor-core order](glossary.md#permutation)). Its eight [codewords](glossary.md#codeword) lie in at most four
  consecutive 32-bit words of the tile's circular bitstream. The readers:
  - the general funnel over four words (`exl3_lane_words`, `exl3_lane_cw`);
  - the fast path for integer rates (`IntPlan`, n % 16 == 0): one select, one shift and one mask per weight;
  - the general fast path (`GenPlan`), for every even n;
  - `exl3_tile_load` and `exl3_tile_cw`. With the 64-bit-window define on (the default), they read one 64-bit window
    (n ≤ 108), or a 96-bit window as three words (n ≤ 128). They rebuild the lane's start bit from the same plan.

  CPU tests hold the host mirrors of the lane plan and of both fast paths to `format.unpackTile`. They test every even
  n from 16 to 128, and the integer path at K1 to K8.
- **The decoders.** The readers mask each codeword to the [window](glossary.md#codeword-window). `exl3_decode<CB>` turns
  it into the `f16` weight, for MUL1 and for MCG. `exl3_mul1_t` and `exl3_mul1_th` give MUL1's byte sum t for the
  deferred map. A Metal define picks one of four exact routes to t. Another define loads the input tiles of the decode
  GEMV in pairs and keeps the sums. The [Wiring](consuming.md#wiring) section of the consuming page names these defines and
  their defaults.
- **Shared Metal headers.** `decode.header` holds the lane helpers, `turboexl3_row_of`, and the GEMV core:
  `turboexl3_members`, `turboexl3_gemv_members`, `turboexl3_fold` and `turboexl3_h128`. Each other header adds helpers:
  - `header5` adds those of form 5 (`turboexl3_group`, `turboexl3_gemv_tg`, `turboexl3_h128r`, ...);
  - `header_s` adds those of the shared heads (`turboexl3_sdgemv`, `turboexl3_sprep`, `turboexl3_sfinish`);
  - `header_t` adds those of the v2 heads;
  - the route, table, prep, mid and finish kernels of the routed prefill read `header_small` (an H128 helper before
    `decode.header`);
  - the prefill GEMMs read `header_nax` and `header_nax_small`: the NAX tiles, the decoders and the bodies' helpers.
- **`src/core/kernels/steel_nax.metal` and `steel.zig`.** The steel headers of MLX 0.32.2, inlined: `steel/defines.h`,
  `steel/utils/type_traits.h`, `steel/utils/integral_constant.h` and `steel/gemm/nax.h`. A comment marks each section
  with its path and its copyright line, and the file is under MIT (see NOTICE). The prefill and dense GEMMs build on
  its `NAXTile` and `tile_matmad_nax` (over `mpp::tensor_ops::matmul2d`). When the consumer's `sdk` declares
  `steel`, `steel.zig` inlines those four headers of the consumer's MLX instead. It drops their `#pragma once` lines
  and their `#include "mlx/..."` lines.
- **`format.zig`, the reference.** It holds the rates (even n from 16 to 128), the windows (8 to 16 bits) and the MUL1
  and MCG [codebooks](glossary.md#codebook). It holds `tensorCorePerm`, `unpackTile`, `hadamard128`, and `prepareInput`
  → `innerGemv` → `finishOutput` (`project`). It also holds `reconstructPublic` and a packer for the tests. It is a port
  of Sushi's `src/exl3/expert_exl3.zig`, with its pins.
- **`ref.zig`.** It holds these functions:
  - `moeRow` and `moeRowDec`: Sushi's `moeSwigluHost` over `format.project` (`moeRow` is MUL1 with a 16-bit window).
    It projects gate and up, runs SwiGLU in `f32` and projects down. Then it sums the slots in slot order, each times
    its score;
  - `synthBank`: random trellis halfwords, with scales that keep the outputs near 1 in size;
  - `compare`: the tolerance measure.
- **Not kernels of the library.** `turboexl3_test_host_swiglu`, `turboexl3_test_router` and `turboexl3_test_qmerge` run
  the host's SwiGLU, router and merge texts as references in the tests. The Metal-source dump (see the
  [tests page](testing.md)) prints the kernels of the routed prefill chain as standalone source files. Its parts
  are in `src/core/kernels/test/dump/`. The dump's
  unit names (`turboexl3_pf_gemm2s_j`, ...) label option variants; they are not MLX kernel names.
