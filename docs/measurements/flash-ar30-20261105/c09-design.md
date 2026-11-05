# C09 — exact SIMD ownership design, source only

2026-11-05. Reserve C09 was preregistered in issue135 and `matrix.md` before
this design. Read the current journal, canonical header/permutation and the
54Q/B05–B08 history first. No implementation, executable change, compiler,
test, prover, model read or GPU experiment was performed. This is an explicit
mapping argument and cost audit, **not** a checked proof or speed result.

Conclusion: removing cross-SIMD storage is impossible with the **current**
work ownership and no replacement communication/recomputation. A different,
bijective ownership can keep all four K-group contributions for each column
inside one SIMD32, with the same128 threads and eight accumulators per output
tile. That makes a barrier-free exact design possible, but creates decoder,
input-shuffle and final-reduction costs. Keep it as a reserve while measuring
the four adaptations already implemented. Neither a gain nor a performance
rejection is justified by this source analysis.

## Current definition of the numerical contract

Sources in the AR30 checkout:

- `native/src/codec.rs::permutation`, and its literal
  `native/shaders/flash_reference_reduction.h`;
- `_qmv_tile_kernel.metal`: `acc`, tile_begin/end loop, codeword extraction,
  literal FMA calls, reference-partials scatter, row→group reduction;
- `flash_projection.metal` / `flash_static_qmv.metal` / `flash_expert_axis.metal`
  select SG4, MB1 and the same canonical header for the target AR domain.

Let `o` be an output tile within NT, `b` the matrix row, `s` the split, `l` the
old SIMD lane0..31, `g` the old SIMD index0..3, and `j` the register0..7.
Inside a16×16 decoded weight tile, the old ownership is:

```text
r(l,j) = 2*(l mod4) + (j mod2) + 8*((j>>1) mod2)
c(l,j) = floor(l/4) + 8*floor(j/4)
t(l,j) = 8*l + j                  // encoded state index
```

Here `r` is the16-element input coordinate inside the K tile, not the matrix
batch row `b`. The inverse is:

```text
old_lane(r,c) = 4*(c mod8) + floor((r mod8)/2)
old_reg(r,c)  = 4*floor(c/8) + 2*floor(r/8) + (r mod2)
```

Define the canonical partial `P[o,b,s,g,r,c]` as follows. Start with positive
float zero; for the exact increasing sequence

```text
tiles_per_split = ceil(TILES_K/N_SPLITS)
begin = s*tiles_per_split
end   = min(begin+tiles_per_split, TILES_K)
tk    = begin+g, begin+g+4, ... < end
P = fma(float(xhat[b*INPUT_DIMS + 16*tk+r]),
        float(canonical_half_codeword(t(r,c), tk, tile_n+o)), P)
```

The first line means the source's
`(TILES_K + N_SPLITS - 1)/N_SPLITS`; SG4 is the K-group count inside each split.
All codebook/Half boundaries and FMA
operand order remain exactly those of the current shader.

For each output column, the final source performs **64 dependent adds**:

```text
S = +0.0f
for r=0..15:
    for g=0..3:
        S = float_add(S, P[o,b,s,g,r,c])
```

Do not pre-sum groups, change the row order, use `simd_sum`, or omit zero partials
from empty groups. Those are different floating-point programs. The later split
finish/Half/H128/Svh stages and the expert route reduction remain unchanged.

## Why shuffles cannot remove the present storage in place

At the end of the current loop, SIMD `g` owns all256 `(r,c)` partials for that
one K group, and owns none of the other three groups' partials. Every final
column depends on contributions from all four groups. A SIMD-local shuffle
cannot obtain the values held in another SIMD. Keeping this ownership thus
requires cross-SIMD storage/communication or recomputing the missing groups;
replacing the scatter/barrier by shuffles alone is not a valid adaptation.

This is a dependency argument for this ownership, not a universal impossibility
theorem for other mappings. 54Q changed TG indexing without changing ownership;
its numeric result therefore does not establish the following design.

## A complete alternative bijection at128 threads

Let `q=simdgroup_index` (0..3), `L=lane` (0..31). Define:

```text
a = L>>3               // local output-column selector 0..3
g = (L>>1)&3           // old K group 0..3, now inside the SIMD
h = L&1                // lower/upper eight input rows
d = a&1
u = a>>1
c = 2*q + d + 8*u      // pair columns to follow EXL3's stream permutation
r = 8*h + j            // new accumulator j=0..7
```

Each lane keeps eight independent partial chains per output tile. For each
`(q,L,j)`, the matching old slot is exactly:

```text
old_group = g
old_lane  = 4*(c mod8) + (j>>1)
old_reg   = 4*u + 2*h + (j&1)
```

Conversely, for any old `(g,old_lane,old_reg)`, derive `r,c` by the old formulas,
then:

```text
q = (c mod8)>>1
a = 2*(c>>3) + (c&1)
L = 8*a + 2*g + (r>>3)
j = r&7
```

All ranges are exact. The inverse recovers the same `g,r,c`; there are
`4*32*8 = 4*256 = 1024` slots per output tile. No partial is duplicated or
dropped. NT2 repeats the same mapping for each `o`, so the live accumulator
array stays16 float scalars/lane, as in the baseline NT2 source. Matrix row,
split, output-tile and expert-record bases are untouched.

## Preserving each FMA chain and tails

K group now varies across lanes of one SIMD. Use a uniform round index
`n=0..ceil((end-begin)/4)-1`, with each lane's
`tk=begin+g+4*n`. For a group whose `tk>=end`, skip its loads and its FMA updates;
keep its initialized positive-zero accumulator. All lanes must nevertheless
participate in any shuffles performed that round, supplying neutral owned data
for inactive groups. A lane-divergent early `continue` around collective
shuffles is invalid. This also covers splits shorter than four K tiles and
empty K groups without changing their partials.

The FMA sequence for each mapped partial is the exact old `begin+g+4*n` sequence.
Each of its two operands must be the same bits, with the same Half→float cast
and the same `fma(x,weight,acc)` argument order. This design does not authorize a
different decoder, a tensor-core dot, or reassociation.

## Exact SIMD-local final reduction

For a SIMD's local column `a`, the source lane of partial `(r,g)` is:

```text
source_lane = 8*a + 2*g + (r>>3)
source_reg  = r&7
```

All lanes execute the same `r=0..15`, `g=0..3` loops. With each lane using its
own `a=L>>3`, a shuffle of **uniformly selected** `acc[o][r&7]` from that source
lane yields the right column's partial. Only `L&7==0` needs to accumulate/store;
other lanes still execute each shuffle. Four leaders per SIMD store the four
columns. For NT2, repeat for each output tile and retain the current destination
index `((b*N_SPLITS+s)*OUTPUT_DIMS + (tile_n+o)*16+c)`.

The register selection must be uniform across the SIMD. Selecting different
`acc[column-dependent-register]` values locally before a shuffle can retrieve
the **source lane's** selected register instead of the destination's intended
register. The proposed mapping deliberately makes `r&7` uniform and keeps
column selection solely in the source-lane index.

No TG reference array/barrier is needed for this reduction. This says nothing
about other barriers in a fused epilogue; the initial proposed domain must be
MB1/SG4/NT1-or2/canonical-reference/non-fused-output AR only. MTP, generic SG8,
other output layouts and unsupported variants require their existing fallback.

## C01 composition trap and a possible subgroup ownership refinement

The existing C01 helper assumes all32 lanes use the same `words` pointer and
that a requested packed-word index identifies a lane owning that word. Here
the same SIMD contains four different `tk` values. Calling that helper
unchanged would shuffle data between **different K tiles**. This is a concrete
incorrect composition, even if the indices remain in bounds.

C09 must either explicitly select a separate backend ineligible for the current
C01 implementation, or implement a new subgroup-aware packed-word ownership.
C01 can remain active on other fallback PSOs, but the two mechanisms' gains
cannot be added or their compatibility assumed.

The column pairing above offers a useful new design fact. For a fixed `q,g`,
the eight lanes cover exactly the64 consecutive state positions of quarter `q`:

```text
t = 64*q + 32*d + 8*(j>>1) + 4*u + 2*h + (j&1)
```

The bits after64q are independent binary fields. For K2/3/4 this quarter needs
the `2*K` core u32 words beginning at `2*K*q`, plus its preceding halo word
(wrapping for q0). Its first state starts `(K-16)` bits before the quarter;
its final state ends exactly at the next quarter boundary. Thus a subgroup
needs exactly5/7/9 consecutive words, not the whole tile's16/24/32.

The eight lanes of a fixed `g` can own those core words with owner ordinal
`v=2*a+h`, source lane `8*(v>>1)+2*g+(v&1)`. One designated lane can additionally
own the halo, broadcast from lane `2*g`. All shuffles remain unconditional;
inactive K groups provide neutral values. This is a **new design refinement**,
not implemented source or a checked oracle. A naive `owned[word_bank]` local
array selection followed by one shuffle has the same register-bank trap as
the accumulator example and must not be used without a correct bank protocol.

## Cost audit and reason to defer a prototype now

| Item | Baseline SG4/NT2 + C01 | C09 design consequence |
| --- | --- | --- |
| Threads / float accumulators |128 /16 per lane | Same source-level counts; compiler allocation unmeasured |
| Reference TG float array |8192 source bytes, scatter/read/barrier | Eliminated in the proposed non-fused domain |
| Global packed-word load requests for all four K groups |32K core u32 per output tile/round | Quarter+halo scheme: `4*(8K+4)`; +25% K2, +16.7% K3, +12.5% K4 requests; cache/DRAM traffic not measured |
| Input data | One half2 lane load +2 half2 shuffles, four values | Cooperative group ownership can retain one half2 load/lane but needs4 half2 shuffles for eight row values |
| Decoding | Two four-state windows/lane | Four two-state windows/lane in this mapping; eight word shuffles plus halo broadcast is a straightforward upper design, versus four word shuffles in C01; metadata/64-bit extraction cost needs inspection |
| Final reduction |64 dependent add steps, one SIMD handles32 NT2 columns in parallel after TG staging |64 shuffle/add steps per output tile, so128 sequential steps per SIMD for NT2; only four column leaders per SIMD, versus32 original active output lanes |

The final reduction is not a free tree: maintaining row→group order requires
all64 dependent steps. Four SIMDs each handling eight NT2 columns execute more
SIMD instructions than the baseline's final single SIMD handling32 columns.
Predicating adds to four leaders avoids redundant scalar sums but leaves most
lanes inactive during those adds. Unconditionally summing in all eight lanes
of each column subgroup repeats the same scalar work eight times. Either form
needs a measured resource/instruction assessment; the saved8KiB and barrier
alone do not establish a win.

An implementation must keep the register index uniform and inspect whether the
fixed16×4 reduction is unrolled into scalar registers. A dynamically indexed
register array that spills would defeat the intended removal of memory traffic.
Equal accumulator counts do not mean equal total registers: input values,
halo/core words, metadata and shuffle intermediates can increase live state.

A different one-SIMD-per-tile mapping could place all groups in that SIMD with
many more partial accumulators per lane, trading the cross-SIMD storage for
register pressure and less K parallelism. It is not a zero-cost alternative,
and thread-count/register variants would be one C09 mechanism, not additional
distinct optimizations.

For now, measure C01/C02/C03/C04 first. C09 is a plausible exact mapping reserve,
with a concrete C01 incompatibility and nontrivial extra decoder/reduction
work. If selected later, amend the matrix before coding with its narrow domain,
subgroup ownership, fallback, exact-loop oracle, and resource budget. Required
gates would compare every mapped partial/column/split to independent canonical
partials (including empty/odd split tails and distinct experts), then frozen
resident/routed complete outputs and the common model/continuation/CLI path.
Only after those pass should representative component timing inform whether
this adaptation saves time. No gain or performance rejection is asserted here.
