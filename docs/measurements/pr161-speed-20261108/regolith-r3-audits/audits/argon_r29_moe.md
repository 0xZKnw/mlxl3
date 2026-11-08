# Round 29 Track C: Decode `moe_route` Unrolls + `moe_gu` / `deq<T_IQ2_S / T_IQ2_XXS / T_IQ3_XXS / T_IQ2_XS>` Optimization

## 1. Summary of Changes
- **`metal/kernels/quant.metal` (lines 145–201):**
  - `deq<T_IQ2_S>` (34 of 48 routed-expert layers in `k512-iq2_xs-metal.gguf`): Replace the four per-iteration 8-bit `uchar` loads and variable shift `(qh << (8 - 2 * l))` with two 2-byte-aligned 16-bit `ushort` loads (`q2 = *(device const ushort*)(b + 2 + 2 * ui)` and `s2 = *(device const ushort*)(b + 34 + 2 * ui)`), pre-shift `qh = uint(b[66 + ib]) >> (4 * h)` once outside the loop, and add `#pragma clang loop unroll(full)` on `l2 = 0, 1`.
  - `deq<T_IQ2_XXS>` (12 of 48 routed-expert layers): Eliminate `a0 = rd32(b + 2 + 8 * ib)` (two `ushort` loads + shift + OR + variable shifts by `8 * l`) in favor of direct byte loads `qs = b + 2 + 4 * ib + 2 * ui` (`qs[0]`, `qs[1]`), pre-shift `s0 = a1 >> (14 * h)` once before the loop, and add `#pragma clang loop unroll(full)`.
  - `deq<T_IQ3_XXS>`: Pre-shift `s0 = aux >> (14 * h)` once before the loop and add `#pragma clang loop unroll(full)`.
  - `deq<T_IQ2_XS>`: Add `#pragma clang loop unroll(full)` on `l2 = 0, 1`.
  - `deq<T_IQ3_S>`: Kept baseline (unrolling `l2` across two `g_iq3s` table lookups per iteration was `+0.9%` slower on `moe_gu` `32,1,4` due to register pressure).
- **`metal/kernels/moe.metal` (`moe_route` lines 163–334, `gu_body` lines 381–410):**
  - Add `e512 = (a.n_expert == 512u)` fast path and `#pragma clang loop unroll(full)` across all 16-element per-lane loops and SIMD xor reduction loops in `moe_route`.
  - Hoist `has_r1`, `has_moce`, and `min_scaled_rank` outside the 10-iteration `rank` loop, split the `!has_moce` vs `has_moce` rank loops, and preload the 16-bit residency mask `res_mask` from `smap` once before the `rank` loop when `has_moce` is active (removing 16 conditional device loads from inside the 10-iteration `rank` loop).
  - Replace the dynamically-indexed array write `if ((ex & 31) == lane) v[ex / 32] = -INFINITY;` with a branchless unrolled select `#pragma clang loop unroll(full) for (int i = 0; i < 16; ++i) v[i] = (ex == lane + 32 * i) ? -INFINITY : v[i];` outside the divergent `if ((ex & 31) == lane)` block, keeping `v[0..15]` in pure scalar registers without dynamic indexing or divergent predication.
  - In `gu_body`, hoist `etok[0]` and advance `x` and `xs16` in place before the `u` loop when `NE == 1` (zero additional live registers; hoisting `int tok[NE]` or preloading `xs[2][16]` for `NE >= 2` was tested and rejected due to `+1..+3%` occupancy regression on `32,1,4`).
- **`metal/tests/expert_parity.cpp` (lines 220–233):**
  - Extend `--ab` to accept a baseline kernel directory path (compiling `quant.metal`, `units.metal`, and `moe.metal` from that directory) as well as a single `.metal` file path.

## 2. Verification & Paired A/B Benchmarks (Apple M5)

### Bitwise Parity
- `route_parity`: 200 windows `K=512` + 200 windows `K=256 + MoCE=0.85`: **0 mismatches (`PASS`)**
- `route_parity --ab`: 200 trials (`moe_route` + `moe_plan`, `split = 0, 1`): **0 fails (`ALL PASS`, 100% bitwise identical)**
- `expert_parity --ab`: `IQ2_S`, `IQ2_XXS`, `IQ3_S`, `IQ2_XS`, `IQ3_XXS` (`NE=1..4,8` + random windows): **0 fails (`ALL PASS`, 100% bitwise identical)**
- `gemv_parity`: All formats, shapes `2560/640/10240`, `ntok = 1, 3, 8`: **`ALL PASS`**

### Paired A/B Timing: `moe_route` (`route_parity --ab --bench`, 192 reps x 9 interleaved rounds)
| Mode | `ntok` | Base (us) | Cand (us) | Ratio | 48-Layer Savings |
|---|---:|---:|---:|---:|---|
| `split=0` (MoCE off) | 1 | 4.00 | 3.49 | **0.8722 (-12.8%)** | 0.192 -> 0.167 ms (-0.025 ms) |
| `split=0` (MoCE off) | 2 | 4.58 | 4.03 | **0.8775 (-12.3%)** | 0.220 -> 0.194 ms (-0.026 ms) |
| `split=0` (MoCE off) | 4 | 6.05 | 5.55 | **0.9174 (-8.3%)** | 0.291 -> 0.267 ms (-0.024 ms) |
| `split=0` (MoCE off) | 8 | 8.80 | 8.41 | **0.9556 (-4.4%)** | 0.423 -> 0.404 ms (-0.019 ms) |
| `split=1` (Streaming + MoCE) | 1 | 5.96 | 4.39 | **0.7363 (-26.4%)** | 0.286 -> 0.211 ms (**-0.075 ms**) |
| `split=1` (Streaming + MoCE) | 2 | 6.43 | 4.85 | **0.7563 (-24.4%)** | 0.309 -> 0.233 ms (**-0.076 ms**) |
| `split=1` (Streaming + MoCE) | 4 | 7.62 | 6.05 | **0.7943 (-20.6%)** | 0.366 -> 0.290 ms (**-0.076 ms**) |
| `split=1` (Streaming + MoCE) | 8 | 10.47 | 9.18 | **0.8763 (-12.4%)** | 0.503 -> 0.441 ms (**-0.062 ms**) |

### Paired A/B Timing: `moe_gu` (`expert_parity --ab --bench`, `QUANT_CTAB=1`)
| Format (Layers in K512) | Tile | `NE=1` Ratio | `NE=2` Ratio | `NE=3` Ratio | `NE=4` Ratio | `ntok=1` Ratio | `ntok=2` Ratio | `ntok=4` Ratio | `ntok=8` Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `IQ2_S` (34 layers) | `32,1,4` | **0.9861** (-1.4%) | **0.9684** (-3.2%) | **0.9422** (-5.8%) | **0.9912** (-0.9%) | **0.9879** (-1.2%) | **0.9817** (-1.8%) | **0.9792** (-2.1%) | **0.9756** (-2.4%) |
| `IQ2_S` (34 layers) | `16,1,4` | **0.9852** (-1.5%) | **0.9665** (-3.4%) | **0.9298** (-7.0%) | **0.9832** (-1.7%) | **0.9856** (-1.4%) | **0.9824** (-1.8%) | **0.9768** (-2.3%) | **0.9766** (-2.3%) |
| `IQ2_XXS` (12 layers) | `16,1,4` | **0.9491** (-5.1%) | **0.9409** (-5.9%) | **0.9691** (-3.1%) | **0.9772** (-2.3%) | **0.9480** (-5.2%) | **0.9475** (-5.3%) | **0.9467** (-5.3%) | **0.9475** (-5.3%) |
| `IQ3_XXS` | `16,1,4` | **0.9929** (-0.7%) | **0.9713** (-2.9%) | **0.9921** (-0.8%) | **0.9756** (-2.4%) | **0.9934** (-0.7%) | **0.9886** (-1.1%) | **0.9849** (-1.5%) | **0.9809** (-1.9%) |
| `IQ2_XS` | `32,1,4` | **0.9972** (-0.3%) | 1.0029 | 1.0018 | **0.9999** | **0.9971** (-0.3%) | **0.9982** (-0.2%) | **0.9991** (-0.1%) | **0.9981** (-0.2%) |

Across the 48 routed-expert layers of `k512-iq2_xs-metal.gguf` (34 `IQ2_S` + 12 `IQ2_XXS` + 2 `IQ3_S`), this saves **~110 us/tok at `ntok=1`**, **~230 us/tok at `ntok=2`**, and **~470 us/tok at `ntok=4`** in `moe_gu` alone, plus **~75 us/tok** in `moe_route` (`split=1`).
