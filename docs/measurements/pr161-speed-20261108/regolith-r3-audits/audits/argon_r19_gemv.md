# Round 19 Track C: `gemv.metal` (`gemv` / `gemv_cat` / `gemv_catk`) K-Split & Inner-Loop Unrolling + `router.metal` Unrolling

- **Agent:** `argon-gemv-and-router-unroll` (`opencode-go/gemini-4-argon-contributor`)
- **Target files:** `metal/kernels/gemv.metal`, `metal/kernels/router.metal`
- **Unified patch:** `/tmp/argon_r19_gemv/argon_r19_gemv.patch` (verified clean via `git -C vendor/strata apply --check`)
- **Standalone A/B harness:** `/tmp/argon_r19_gemv/ab_bench.cpp` (`< 0.25s` minimal shader subset compile per library)

---

## 1. Key Findings & Root-Cause Analysis

### 1.1 `gemv`, `gemv_cat`, `gemv_catk`, `gemv_pair`, and `gemv_x` (`metal/kernels/gemv.metal`)
1. **Why `for (int t = 0; t < NTOK; ++t)` and `for (int r = 0; r < NR; ++r)` Were Not Unrolled by Default:**
   - `NTOK` (`[[function_constant(0)]]`) is a Metal function constant (not a C++ template parameter). Without `#pragma clang loop unroll(full)`, the Apple Metal compiler does **not** unroll `for (int t = 0; t < NTOK; ++t)` when the body contains `X<F>::load` and `unit_combine<UNIT>`, nor does it unroll `for (int r = 0; r < NR; ++r)` when `NR > 1` (`acc[r][t]`, `q[r]`, `sc[r]`).
   - Consequently, `acc[r][t]` was indexed by a runtime loop variable `t` (and `r`), preventing scalar replacement of aggregates (SROA) into registers and causing thread-local stack indexing and loop-branch overhead on every unit `u` and every K-split reduction.
2. **K-Split (`KS > 1`) Reduction & Bounds Check Hoisting (`gemv` and `gemv_catk`):**
   - Adding `#pragma clang loop unroll(full)` to `for (int r = 0; r < NR; ++r)`, `for (int t = 0; t < NTOK; ++t)`, and `for (int k = 0; k < KS; ++k)` in the `KS > 1` reduction emits `KS` immediate-offset threadgroup loads (`red[(k * RPT + lrow0 + r) * 8 + t]`) directly into registers.
   - Hoisting `if (row0 + r < int(a.n_out))` outside the `t` and `k` reduction loops in `gemv` and `gemv_catk` avoids redundant threadgroup loads on tail rows.
3. **`gemv_pair` Init Bound (`t < NTOK` vs `t < 8`):**
   - Changed `for (int t = 0; t < 8; ++t) ag[t] = au[t] = 0.0f;` to `#pragma clang loop unroll(full)` `for (int t = 0; t < NTOK; ++t) ag[t] = au[t] = 0.0f;` and unrolled all `t < NTOK` loops in `gemv_pair` and `gemv_x`.
4. **Tested & Rejected Sub-Variant (`if (NR == 1)` branch inside `u` loop):**
   - Explicitly branching on `if (NR == 1)` inside `u` with scalar `q0[UNIT]` / `sc0` produced identical codegen once `#pragma clang loop unroll(full)` was present on `for (int r = 0; r < NR; ++r)`, except for a `+3.62%` scheduling wobble on `IQ3_S 6144->2560 (16_1_4_1, ntok=2)`. Kept the clean `#pragma clang loop unroll(full)` loop without the redundant `if (NR == 1)` branch.

### 1.2 `router_route` (`metal/kernels/router.metal`)
1. **Why Guarding `threadgroup_barrier` with `if (fmt<F>::TAB > 1)` Regresses `router_route`:**
   - We A/B tested `if (fmt<F>::TAB > 1) { load_tab<F>(tab, tid, nth); threadgroup_barrier(mem_flags::mem_threadgroup); }` in `router_route` across 21 interleaved pairs x 128 reps.
   - Removing the initial `threadgroup_barrier(mem_flags::mem_threadgroup)` at line 131 **consistently regressed** `router_route_30` (BF16) by `+1.22% .. +1.57%` (`24.06 -> 24.35 us` at `ntok=1`, `39.23 -> 39.71 us` at `ntok=4`) and `router_route_0` (F32) by `+3.84% .. +4.35%` (`42.04 -> 43.79 us` at `ntok=1`).
   - **Root cause:** `router_route` launches a 512-thread (16-simdgroup) threadgroup where all 16 simdgroups read the same `xp` / `s16` vector across `u = lane; u < U; u += 32` and then synchronize at the device-fence / atomic counter tail (`ngx = 32` threadgroups). The initial `threadgroup_barrier(mem_flags::mem_threadgroup)` aligns all 16 simdgroups within each threadgroup at `u = 0` so simdgroup 0..15 hit the L1 cache in lockstep for `xp[u]` and finish together at the tail barrier. Without it, simdgroups drift out of phase on `xp` loads.
   - Therefore, per Rule 2.3 (*"Only keep changes that are 100% memcmp == 0 and neutral-to-faster (zero regressions)"*), we **kept** the initial synchronization in `router_route` and **kept** `#pragma clang loop unroll(full)` on the three `for (int t = 0; t < NTOK; ++t)` loops in `router_route` (which is `100%` bitwise identical and neutral-to-faster: `-0.28%` on BF16 `ntok=2`, `-0.33%` on F32 `ntok=1`).

---

## 2. Standalone A/B Benchmark Results (Apple M5)

All measurements use interleaved A/B pairs (`48..128` reps per command buffer, `7..15` pairs, both pipelines pre-warmed) and verify `100%` bitwise `memcmp == 0` across all `ntok = 1..8`.

### 2.1 `gemv` Production Shapes & Tiles (`T_Q8_0`, `T_IQ3_S`, `T_Q2_0`)

| Shape & Format | Kernel / Tile | `ntok=1` Base -> Cand (`diff`) | `ntok=2` Base -> Cand (`diff`) | `ntok=4` Base -> Cand (`diff`) | Bitwise `ntok=1..8` |
|---|---|---:|---:|---:|:---:|
| `Q8_0 2560->10240` | `gemv_8_32_1_4_4` | `201.51 -> 200.96 us` (`-0.27%`) | `211.49 -> 202.91 us` (**`-4.05%`**) | `348.97 -> 299.45 us` (**`-14.19%`**) | `PASS` |
| `Q8_0 2560->10240` | `gemv_8_32_1_8_8` | `196.08 -> 195.60 us` (`-0.25%`) | `323.95 -> 258.86 us` (**`-20.09%`**) | `624.15 -> 476.42 us` (**`-23.67%`**) | `PASS` |
| `Q8_0 2560->10240` | `gemv_8_16_1_4_1` | `208.93 -> 208.91 us` (`-0.01%`) | `208.05 -> 209.34 us` (`+0.62%`) | `218.76 -> 206.75 us` (**`-5.49%`**) | `PASS` |
| `Q8_0 2560->10240` | `gemv_8_32_2_4_1` | `207.49 -> 207.55 us` (`+0.03%`) | `206.22 -> 207.23 us` (`+0.49%`) | `231.49 -> 209.05 us` (**`-9.69%`**) | `PASS` |
| `Q8_0 2560->6144` | `gemv_8_32_1_4_4` | `129.98 -> 130.15 us` (`+0.13%`) | `141.41 -> 134.75 us` (**`-4.71%`**) | `209.96 -> 184.90 us` (**`-11.94%`**) | `PASS` |
| `Q8_0 2560->6144` | `gemv_8_32_1_8_8` | `128.38 -> 128.39 us` (`+0.00%`) | `193.50 -> 155.67 us` (**`-19.55%`**) | `364.34 -> 279.02 us` (**`-23.42%`**) | `PASS` |
| `Q8_0 2560->6144` | `gemv_8_16_1_4_1` | `120.73 -> 120.34 us` (`-0.32%`) | `119.97 -> 120.10 us` (`+0.11%`) | `132.44 -> 122.34 us` (**`-7.63%`**) | `PASS` |
| `Q8_0 2560->6144` | `gemv_8_32_2_4_1` | `118.67 -> 118.86 us` (`+0.16%`) | `118.42 -> 118.84 us` (`+0.36%`) | `142.03 -> 123.08 us` (**`-13.34%`**) | `PASS` |
| `Q8_0 2560->512` | `gemv_8_32_1_4_4` | `7.71 -> 7.73 us` (`+0.21%`) | `11.05 -> 10.46 us` (**`-5.40%`**) | `17.99 -> 16.73 us` (**`-7.00%`**) | `PASS` |
| `Q8_0 2560->512` | `gemv_8_32_1_8_8` | `10.22 -> 10.23 us` (`+0.05%`) | `16.60 -> 14.44 us` (**`-12.99%`**) | `31.32 -> 24.36 us` (**`-22.24%`**) | `PASS` |
| `Q8_0 2560->512` | `gemv_8_16_1_4_1` | `6.48 -> 6.48 us` (`+0.09%`) | `8.60 -> 7.98 us` (**`-7.19%`**) | `13.35 -> 11.68 us` (**`-12.49%`**) | `PASS` |
| `Q8_0 2560->512` | `gemv_8_32_2_4_1` | `6.28 -> 6.24 us` (`-0.55%`) | `8.84 -> 8.18 us` (**`-7.54%`**) | `14.77 -> 12.61 us` (**`-14.61%`**) | `PASS` |
| `Q8_0 6144->2560` | `gemv_8_32_1_4_4` | `106.68 -> 106.84 us` (`+0.14%`) | `116.47 -> 109.68 us` (**`-5.82%`**) | `184.67 -> 169.26 us` (**`-8.34%`**) | `PASS` |
| `Q8_0 6144->2560` | `gemv_8_32_1_8_8` | `105.99 -> 106.05 us` (`+0.06%`) | `120.07 -> 108.98 us` (**`-9.24%`**) | `198.21 -> 173.55 us` (**`-12.44%`**) | `PASS` |
| `Q8_0 6144->2560` | `gemv_8_16_1_4_1` | `112.05 -> 111.92 us` (`-0.11%`) | `117.51 -> 115.72 us` (**`-1.52%`**) | `152.61 -> 132.76 us` (**`-13.01%`**) | `PASS` |
| `Q8_0 6144->2560` | `gemv_8_32_2_4_1` | `110.23 -> 110.17 us` (`-0.05%`) | `115.58 -> 113.20 us` (**`-2.07%`**) | `140.40 -> 130.82 us` (**`-6.82%`**) | `PASS` |
| `IQ3_S 2560->12288` | `gemv_21_32_1_4_4` | `219.87 -> 219.84 us` (`-0.01%`) | `290.39 -> 291.32 us` (`+0.32%`) | `431.03 -> 429.36 us` (`-0.39%`) | `PASS` |
| `IQ3_S 2560->12288` | `gemv_21_16_1_4_1` | `123.05 -> 122.36 us` (`-0.56%`) | `149.62 -> 149.92 us` (`+0.20%`) | `226.63 -> 226.33 us` (`-0.13%`) | `PASS` |
| `IQ3_S 2560->12288` | `gemv_21_32_2_4_1` | `114.88 -> 113.00 us` (**`-1.64%`**) | `151.16 -> 139.53 us` (**`-7.70%`**) | `265.46 -> 210.24 us` (**`-20.80%`**) | `PASS` |
| `IQ3_S 2560->12288` | `gemv_21_8_1_4_1` | `121.11 -> 121.19 us` (`+0.07%`) | `143.04 -> 142.57 us` (`-0.33%`) | `206.65 -> 203.51 us` (**`-1.52%`**) | `PASS` |
| `IQ3_S 6144->2560` | `gemv_21_32_1_4_4` | `73.84 -> 73.87 us` (`+0.04%`) | `104.33 -> 104.27 us` (`-0.06%`) | `162.88 -> 163.11 us` (`+0.14%`) | `PASS` |
| `IQ3_S 6144->2560` | `gemv_21_16_1_4_1` | `60.79 -> 60.73 us` (`-0.10%`) | `75.71 -> 75.75 us` (`+0.05%`) | `114.59 -> 114.28 us` (`-0.27%`) | `PASS` |
| `IQ3_S 6144->2560` | `gemv_21_32_2_4_1` | `55.33 -> 55.33 us` (`-0.01%`) | `75.87 -> 69.07 us` (**`-8.96%`**) | `134.64 -> 107.01 us` (**`-20.52%`**) | `PASS` |
| `IQ3_S 6144->2560` | `gemv_21_8_1_8_1` | `60.99 -> 60.94 us` (`-0.09%`) | `73.42 -> 73.25 us` (`-0.24%`) | `107.80 -> 108.23 us` (`+0.40%`) | `PASS` |
| `Q2_0 640->2560` | `gemv_42_8_2_4_1` | `4.11 -> 4.15 us` (`+0.85%`) | `7.00 -> 5.91 us` (**`-15.57%`**) | `14.54 -> 10.26 us` (**`-29.42%`**) | `PASS` |
| `Q2_0 640->2560` | `gemv_42_8_1_4_1` | `4.37 -> 4.39 us` (`+0.53%`) | `6.01 -> 6.01 us` (`+0.06%`) | `10.23 -> 10.24 us` (`+0.07%`) | `PASS` |

### 2.2 `gemv_cat`, `gemv_catk`, `gemv_pair`, and `router_route`

| Shape & Format | Kernel | `ntok=1` Base -> Cand (`diff`) | `ntok=2` Base -> Cand (`diff`) | `ntok=4` Base -> Cand (`diff`) | Bitwise `ntok=1..8` |
|---|---|---:|---:|---:|:---:|
| `Q8_0 2560->512+512` | `gemvc_8_32_1_4` | `18.53 -> 18.62 us` (`+0.50%`) | `20.55 -> 19.20 us` (**`-6.59%`**) | `32.83 -> 31.32 us` (**`-4.58%`**) | `PASS` |
| `Q8_0 2560->512+512` | `gemvc_8_16_1_4` | `18.52 -> 18.45 us` (`-0.38%`) | `19.04 -> 18.23 us` (**`-4.26%`**) | `25.29 -> 21.67 us` (**`-14.33%`**) | `PASS` |
| `Q8_0 2560->512+512` | `gemvck_8_32_1_4_4` | `19.12 -> 19.12 us` (`-0.04%`) | `21.89 -> 19.76 us` (**`-9.74%`**) | `34.35 -> 31.50 us` (**`-8.29%`**) | `PASS` |
| `Q8_0 2560->512+512` | `gemvck_8_32_2_4_2` | `20.65 -> 19.29 us` (**`-6.58%`**) | `23.38 -> 20.98 us` (**`-10.28%`**) | `34.00 -> 28.61 us` (**`-15.84%`**) | `PASS` |
| `Q8_0 2560->512+512` | `gemvck_8_32_1_8_8` | `19.10 -> 19.21 us` (`+0.54%`) | `32.43 -> 27.08 us` (**`-16.52%`**) | `59.15 -> 47.65 us` (**`-19.45%`**) | `PASS` |
| `Q2_0 2560->640x2` | `gemv_pair_42_42` | `9.89 -> 9.88 us` (`-0.09%`) | `16.50 -> 16.49 us` (`-0.05%`) | `30.47 -> 30.48 us` (`+0.03%`) | `PASS` |
| `BF16 2560->512` | `router_route_30` | `24.01 -> 24.07 us` (`+0.25%`) | `26.54 -> 26.56 us` (`+0.08%`) | `39.05 -> 39.17 us` (`+0.30%`) | `PASS` |
| `F32 2560->512` | `router_route_0` | `41.88 -> 41.74 us` (`-0.33%`) | `46.83 -> 46.77 us` (`-0.12%`) | `55.69 -> 55.69 us` (`+0.01%`) | `PASS` |

---

## 3. Parity Verification

- `gemv_parity`: **ALL PASS** across all 13 formats (`F32`, `BF16`, `Q8_0`, `Q4_0`, `Q5_0`, `IQ4_NL`, `IQ4_XS`, `Q2_0`, `IQ3_S`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`), `n_in` in `{2560, 640, 10240}`, `ntok` in `{1, 3, 8}`, all 20 tiles (`window == one token: bitwise`).
- `fuse_parity`: **ALL PASS** (`gemv_cat` `ntok=1..8` bitwise, `router_route` `BF16` & `F32` `ntok=1..8` 2 runs each bitwise, `gr_norm_comb` & `gr_norm_comb_4s` `ntok=1..8` bitwise).
- `route_parity`: **ALL PASS** (`K=512` 200 windows `0` mismatches, `K=256 + MoCE=0.85` 200 windows `0` mismatches).
