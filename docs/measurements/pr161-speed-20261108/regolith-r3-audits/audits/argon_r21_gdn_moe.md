# Round 21 Track C: `gdn_step_t` 32-Bit State Offset `st_off` + `float4` `conv_w` + Program-Scope `constant` `gdn_q2s`/`gdn_p4s` & `moe.metal` `plan_body` `topk10` Fast-Division + `unroll_count(10)`

## Summary of Changes

1. **`metal/kernels/gdn.metal` (`gdn_step_t`):**
   - Replaced the 64-bit per-lane pointer `device float* st_base = state + (i0 * hv + h) * S + j` kept live across the 4-barrier `for (int t = 0; t < GDN_NTOK; ++t)` loop with a 32-bit integer offset `const int st_off = (i0 * hv + h) * S + j`.
   - Moved `q2s[8]` and `p4s[4]` from function-local arrays inside `if (a.plane)` to program-scope `constant float gdn_q2s[8]` and `constant float gdn_p4s[4]`.
   - Replaced the 4 scalar `conv_w[ch * 4 + k]` loads with a single 128-bit `float4` load `*(device const float4*)(conv_w + ch * 4)`.
2. **`metal/kernels/moe.metal` (`plan_body`):**
   - Replaced the runtime 32-bit integer division `i / int(a.topk)` in `plan_body` with `topk10 ? ((i * 205) >> 11) : (i / int(a.topk))` (exact for all `0 <= i < 1024`) and added `#pragma clang loop unroll_count(10)` on both `i < ne` loops in `plan_body`.

## Verification & Paired A/B Benchmarks (Apple M5)

### `gdn_parity --ab --bench`
```
device Apple M5
gdn_step pipeline GDN_NTOK 1: max 512 threads per threadgroup (the kernel needs 512)
gdn_step pipeline GDN_NTOK 2: max 512 threads per threadgroup (the kernel needs 512)
gdn_step pipeline GDN_NTOK 3: max 512 threads per threadgroup (the kernel needs 512)
gdn_step pipeline GDN_NTOK 4: max 512 threads per threadgroup (the kernel needs 512)
gdn_step pipeline GDN_NTOK 5: max 512 threads per threadgroup (the kernel needs 512)
gdn_step pipeline GDN_NTOK 6: max 512 threads per threadgroup (the kernel needs 512)
gdn_step pipeline GDN_NTOK 7: max 512 threads per threadgroup (the kernel needs 512)
gdn_step pipeline GDN_NTOK 8: max 512 threads per threadgroup (the kernel needs 512)
ab ntok 1 commit 0x0 plane 0: y ok state ok hist ok  PASS
ab ntok 1 commit 0x1 plane 0: y ok state ok hist ok  PASS
ab ntok 1 commit 0x0 plane 49152: y ok state ok hist ok  PASS
ab ntok 1 commit 0x1 plane 49152: y ok state ok hist ok  PASS
ab ntok 2 commit 0x0 plane 0: y ok state ok hist ok  PASS
ab ntok 2 commit 0x1 plane 0: y ok state ok hist ok  PASS
ab ntok 2 commit 0x101 plane 0: y ok state ok hist ok  PASS
ab ntok 2 commit 0x0 plane 49152: y ok state ok hist ok  PASS
ab ntok 2 commit 0x1 plane 49152: y ok state ok hist ok  PASS
ab ntok 2 commit 0x101 plane 49152: y ok state ok hist ok  PASS
ab ntok 3 commit 0x0 plane 0: y ok state ok hist ok  PASS
ab ntok 3 commit 0x1 plane 0: y ok state ok hist ok  PASS
ab ntok 3 commit 0x101 plane 0: y ok state ok hist ok  PASS
ab ntok 3 commit 0x102 plane 0: y ok state ok hist ok  PASS
ab ntok 3 commit 0x0 plane 49152: y ok state ok hist ok  PASS
ab ntok 3 commit 0x1 plane 49152: y ok state ok hist ok  PASS
ab ntok 3 commit 0x101 plane 49152: y ok state ok hist ok  PASS
ab ntok 3 commit 0x102 plane 49152: y ok state ok hist ok  PASS
ab ntok 4 commit 0x0 plane 0: y ok state ok hist ok  PASS
ab ntok 4 commit 0x1 plane 0: y ok state ok hist ok  PASS
ab ntok 4 commit 0x101 plane 0: y ok state ok hist ok  PASS
ab ntok 4 commit 0x102 plane 0: y ok state ok hist ok  PASS
ab ntok 4 commit 0x103 plane 0: y ok state ok hist ok  PASS
ab ntok 4 commit 0x0 plane 49152: y ok state ok hist ok  PASS
ab ntok 4 commit 0x1 plane 49152: y ok state ok hist ok  PASS
ab ntok 4 commit 0x101 plane 49152: y ok state ok hist ok  PASS
ab ntok 4 commit 0x102 plane 49152: y ok state ok hist ok  PASS
ab ntok 4 commit 0x103 plane 49152: y ok state ok hist ok  PASS
ab bitwise summary: 0 fail(s)  ALL PASS
ab bench plane     0 ntok 1 commit  base   55.14 us  cand   55.36 us  ratio 1.0044 (x36: 1.985 -> 1.993 ms)
ab bench plane     0 ntok 1 verify  base   27.53 us  cand   27.47 us  ratio 0.9954 (x36: 0.991 -> 0.989 ms)
ab bench plane     0 ntok 2 commit  base   58.39 us  cand   58.54 us  ratio 1.0025 (x36: 2.102 -> 2.107 ms)
ab bench plane     0 ntok 2 verify  base   33.07 us  cand   33.08 us  ratio 0.9998 (x36: 1.190 -> 1.191 ms)
ab bench plane     0 ntok 3 commit  base   60.46 us  cand   60.82 us  ratio 1.0096 (x36: 2.177 -> 2.190 ms)
ab bench plane     0 ntok 3 verify  base   38.57 us  cand   38.78 us  ratio 1.0054 (x36: 1.389 -> 1.396 ms)
ab bench plane     0 ntok 4 commit  base   62.50 us  cand   62.61 us  ratio 1.0040 (x36: 2.250 -> 2.254 ms)
ab bench plane     0 ntok 4 verify  base   44.33 us  cand   44.55 us  ratio 1.0064 (x36: 1.596 -> 1.604 ms)
ab bench plane 49152 ntok 1 commit  base   55.04 us  cand   55.08 us  ratio 1.0011 (x36: 1.981 -> 1.983 ms)
ab bench plane 49152 ntok 1 verify  base   27.28 us  cand   27.32 us  ratio 0.9997 (x36: 0.982 -> 0.984 ms)
ab bench plane 49152 ntok 2 commit  base   60.02 us  cand   59.81 us  ratio 0.9965 (x36: 2.161 -> 2.153 ms)
ab bench plane 49152 ntok 2 verify  base   33.59 us  cand   33.58 us  ratio 1.0013 (x36: 1.209 -> 1.209 ms)
ab bench plane 49152 ntok 3 commit  base   61.29 us  cand   61.27 us  ratio 1.0008 (x36: 2.206 -> 2.206 ms)
ab bench plane 49152 ntok 3 verify  base   39.19 us  cand   39.38 us  ratio 1.0065 (x36: 1.411 -> 1.418 ms)
ab bench plane 49152 ntok 4 commit  base   63.54 us  cand   63.20 us  ratio 0.9938 (x36: 2.288 -> 2.275 ms)
ab bench plane 49152 ntok 4 verify  base   44.76 us  cand   45.08 us  ratio 1.0046 (x36: 1.611 -> 1.623 ms)
```

### `route_parity --ab --bench`
```
route_parity --ab bitwise (moe_route + moe_plan, split 0/1, 200 trials): 0 fail(s)  ALL PASS
ab bench moe_route split 0 ntok 1: base  23.94 us  cand  25.82 us  ratio 1.0713 (x48: 1.149 -> 1.239 ms)
ab bench moe_plan  split 0 ntok 1: base  12.92 us  cand  13.82 us  ratio 1.0798 (x48: 0.620 -> 0.663 ms)
ab bench moe_route split 0 ntok 2: base   6.18 us  cand   5.61 us  ratio 0.9066 (x48: 0.297 -> 0.269 ms)
ab bench moe_plan  split 0 ntok 2: base   4.14 us  cand   3.64 us  ratio 0.8799 (x48: 0.199 -> 0.175 ms)
ab bench moe_route split 0 ntok 4: base   8.23 us  cand   7.07 us  ratio 0.8591 (x48: 0.395 -> 0.339 ms)
ab bench moe_plan  split 0 ntok 4: base   6.15 us  cand   5.02 us  ratio 0.8187 (x48: 0.295 -> 0.241 ms)
ab bench moe_route split 0 ntok 8: base  12.35 us  cand   9.82 us  ratio 0.7962 (x48: 0.593 -> 0.471 ms)
ab bench moe_plan  split 0 ntok 8: base   9.87 us  cand   7.24 us  ratio 0.7336 (x48: 0.474 -> 0.348 ms)
ab bench moe_route split 1 ntok 1: base   7.20 us  cand   6.90 us  ratio 0.9598 (x48: 0.346 -> 0.331 ms)
ab bench moe_route split 1 ntok 2: base   8.02 us  cand   7.35 us  ratio 0.9161 (x48: 0.385 -> 0.353 ms)
ab bench moe_route split 1 ntok 4: base   9.85 us  cand   8.59 us  ratio 0.8703 (x48: 0.473 -> 0.412 ms)
ab bench moe_route split 1 ntok 8: base  14.12 us  cand  11.48 us  ratio 0.8134 (x48: 0.678 -> 0.551 ms)
```
