# round 7 track b: hc-up register epilogue

## decision

ship the cooperative-destination epilogue on the existing `PromptGemm::k4` path. use 64x64 below `T=1024`, 64x128 at/above it. keep the original 64x64 threadgroup epilogue for other layouts. no full-model speedup claimed.

- patch: `/tmp/sol_r7_hcup/sol_r7_hcup.patch`
- isolated source: `/tmp/sol_r7_hcup/strata`
- evidence, source snapshots, commands, hashes: `/tmp/sol_r7_hcup/`
- `git -C vendor/strata apply --check /tmp/sol_r7_hcup/sol_r7_hcup.patch`: exit 0.
- vendor base head: `296b59d2f73c8f04a805df29c2fa97655adace78`; the actual working-tree baseline is preserved under `baseline/`, because head alone does not identify local changes.

## what changed

`prompt.metal` adds a compile-time register destination to the existing templated kernel, retaining its original fallback epilogue. the register branch is specifically `hc=4`: a lane computes the four sigmoid/fma terms in exactly c=0,1,2,3 order, then divides by four and writes fp16 plus optional fp32. it never reduces already-rounded products or reassociates the fmas.

`window.cpp` changes only the original hc_read tail at lines 3357–3360: tile name/grid selection from the already-probed `p.k4`; no new per-read gpu probe, no other prompt dispatch changes. `HC` is a fixed four in this engine. overriding `COOP_K4` incorrectly remains unsafe, as it already is for other layout-sensitive kernels.

`comb_fold_parity.cpp` adds a focused `T hcup [bench|confirm]` mode. it compiles only the final hc-up section of prompt.metal, compares all four tile shapes with both output bindings, and offers five rotating-order confirmation batches. the full original test mode remains intact. register tests skip on the existing probe's non-k4 path.

## destination layout: direct measurement

`layout.log` records the 64x64, four-simdgroup destination's full lane/element coordinates on apple m5. each lane owns four consecutive n columns at the same m, not two columns requiring a neighbor shuffle. examples:

- lane 0, capacity 32: `(0,0),(1,0),(2,0),(3,0)`, then `(0,8)..(3,8)`, then `(32,0)..(35,0)`.
- lane 1 starts `(4,0)..(7,0)`.
- lane 2 starts `(0,1)..(3,1)`.

so no shuffle or small shared buffer is needed. the measured static threadgroup allocation falls from 16,384 bytes at baseline 64x64 (32,768 at larger threadgroup tiles) to **zero** on every register tile. all four register shapes pass bitwise parity, independently validating their assumed four-element grouping. the exploratory layout probe and generated tiny probe library are scratch-only, not in the patch.

## tile sweep

apple m5, safe math, `M=T, N=10240, K=320`, seeded fp16 tensors, `mixed` unbound. milliseconds per 48 dispatches; best of three, 10ms warm-up per case. these are exploratory selection results, not confirmation medians.

| tile / destination | T64 | T128 | T512 | T1024 | T2048 |
|---|---:|---:|---:|---:|---:|
| 64x64 threadgroup | 2.954 | 4.136 | 16.170 | 31.409 | 62.401 |
| 64x64 register | 2.591 | 4.028 | 14.974 | 29.813 | 59.328 |
| 128x64 threadgroup | 4.803 | 6.410 | 27.043 | 55.775 | 106.629 |
| 128x64 register | 3.815 | 4.822 | 17.458 | 34.321 | 67.416 |
| 64x128 threadgroup | 3.727 | 6.410 | 28.130 | 54.890 | 107.091 |
| 64x128 register | 3.086 | 4.445 | 15.058 | 29.179 | 57.356 |
| 64x32 threadgroup | 3.415 | 6.607 | 25.902 | 52.582 | 106.821 |
| 64x32 register | 2.535 | 4.510 | 18.664 | 37.307 | 73.741 |

both output bindings were measured; bound results are in `candidate1_T*.log`. all sweep processes finished below 5s; the longest was 4.35s at T2048.

## paired confirmation

five batches, rotating baseline/reg64/reg128 order each batch, same process/data, 48 dispatches per batch. table entries are medians over five batches. `confirm_T*.log` preserves every batch, including a cold/noisy first T64 unbound sample (baseline 9.061ms); it is not silently removed.

| T | baseline, unbound | selected register, unbound | reduction | baseline, bound | selected register, bound | reduction |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 2.988 | 2.598 | 13.1% | 3.013 | 2.619 | 13.1% |
| 128 | 4.327 | 3.993 | 7.7% | 4.389 | 4.090 | 6.8% |
| 512 | 16.183 | 14.930 | 7.7% | 17.040 | 15.630 | 8.3% |
| 1024 | 31.444 | 28.968 | 7.9% | 35.565 | 29.801 | 16.2% |
| 2048 | 62.901 | 57.274 | 8.9% | 68.618 | 59.018 | 14.0% |

at T2048, unbound baseline range is 62.729–63.065ms versus 57.236–57.394ms for reg64x128. larger 128x64 loses even without shared memory; 64x32 only narrowly wins at T64 but loses elsewhere, so it is not selected. 64x128 is not selected at T512 because it is slightly slower unbound than reg64x64.

a later `final_confirm_T2048.log` run is **excluded/inconclusive**: baseline jumped to 332–414ms and the process hit its 4.9s cap. a separate `gdn_prompt_parity` process was present in the contemporaneous process snapshot; contention is plausible, not proven. no result from that incomplete run is used above. earlier complete paired logs remain the evidence for the recommendation. parent should recheck in a serialized gpu window before attributing any end-to-end gain.

## correctness / build results

- focused final evaluator: exit 0 at **T=1,7,64,512,1024,1983,2048**, four threadgroup + four register tiles, bound/unbound mixed; exact `memcmp==0` on fp16 mixh and fp32 mixed when bound. sentinel-filled candidate buffers catch unwritten outputs. logs: `final_T*_parity.log`.
- independent comparison against the **literal copied original kernel source**, separately compiled (not just the retained template fallback): exit 0 for all the same T values, all eight shapes, both bindings. `compare_original.cpp` and `original_compare_T*.log` preserve this check.
- original full `comb_fold_parity 64` mode: exit 0, all existing tests pass at T=1,7,64,64; 4.13s process duration. `final_full_T64.log`.
- forced non-k4 focused test (`COOP_K4=0`), T64: exit 0; register cases skip and all threadgroup variants pass. this checks override/skip behavior on m5, **not actual m4 hardware parity**.
- c++ test executable built successfully with clang++ c++20/O2 and existing libmtl_rt.a.
- scratch `window.cpp` independently compiled to `/tmp/sol_r7_hcup/window.o` with the existing engine's actual ninja flags: exit 0. `engine_build.log`, exact argv in `metadata.json`.
- full prompt shader library compiled successfully through the existing full test mode.
- final visual diff/context review completed; only the three owned paths appear in the patch.

## reproduction

from the workspace root:

```text
clang++ -std=c++20 -O2 /tmp/sol_r7_hcup/strata/metal/tests/comb_fold_parity.cpp -I/tmp/sol_r7_hcup/strata/metal/src -I/tmp/sol_r7_hcup/strata/metal/tests vendor/strata/build-metal/libmtl_rt.a -framework Metal -framework Foundation -o /tmp/sol_r7_hcup/comb_fold_parity
STRATA_METAL_KERNELS=/tmp/sol_r7_hcup/strata/metal/kernels /tmp/sol_r7_hcup/comb_fold_parity 2048 hcup
STRATA_METAL_KERNELS=/tmp/sol_r7_hcup/strata/metal/kernels /tmp/sol_r7_hcup/comb_fold_parity 2048 hcup bench
STRATA_METAL_KERNELS=/tmp/sol_r7_hcup/strata/metal/kernels /tmp/sol_r7_hcup/comb_fold_parity 2048 hcup confirm
```

invoke gpu processes with a <5s timeout and no competing gpu benchmark. safe math is the runtime default; no fast-math override was supplied. `metadata.json` records exact build argv and final source/patch hashes.

## limits / excluded attempts

no edits to tracked vendor sources, no :8080 access, no full-model benchmarks, and no child agents. all gpu jobs used a 4.9s deadline. the original unmodified `comb_fold_parity 2048 bench` attempt hit that deadline with buffered stdout; it cannot establish which stage consumed the time and is not a baseline measurement (`baseline_2048.log`). the focused evaluator repairs the duration problem without changing inputs/comparison mathematics. compilation-only jobs may exceed 5s; engine object compilation was cpu-only.

performance evidence is standalone, not kprof or full-model validation. non-m5 hardware was not available; other layouts retain the original kernel. production sizes beyond the measured range and architecture revisions need revalidation.
