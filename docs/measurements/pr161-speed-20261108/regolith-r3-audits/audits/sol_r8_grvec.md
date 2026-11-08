# round 8 track b: gr address/memory experiment

plan: compare current gr kernels with precomputed 32-bit expert offsets, ordered slot compaction, and residual reread versus register caching. owner: prompt.metal gr helpers only; no host dispatch/interface changes. fma and reduction order must remain unchanged, dropped slots must skip fma entirely. uint offsets are safe for the supported chunk geometry (maximum 230,686,720 elements).

evaluator: comb_fold_parity short gr-only mode, fixed seed 23+T, N=2560 hc=4 topk=10 with dropped slots, T=1,7,64,512,1024,2048; memcmp residual and half normalization output against original dispatch chains. timing four dispatches per sample, three samples; confirmation interleaves baseline/candidate processes. select >3% repeatable wins; reject any bit mismatch. each standalone process capped at 5s, no full model or server use. archive source variants and logs under /tmp/sol_r8_grvec. no tracked vendor edits.

## implementation and evidence

scratch input snapshot: git `81b76e4` in `/tmp/sol_r8_grvec/strata` (only the two owned files tracked). this snapshots the actual dirty working inputs, not vendor HEAD. source variants are archived as `/tmp/sol_r8_grvec/{offset,compact,reread,both,stream,stream_orig,unroll,unrolld,candidate}.metal`; baseline is `/tmp/sol_r8_grvec/baseline/prompt.metal`. all measurements use apple m5, runtime safe-math default, existing `build-metal/libmtl_rt.a` and clang++ -std=c++20 -O2.

candidate keeps `bv[10]` for bo, then updates/reduces/emits one residual stream at a time with `rv[10]`: nominal live cached float payload 20 rather than 40, plus fewer stream accumulators/gates. this is a source-level register-pressure argument, not a measured hardware register count. reduction order within each stream and per-element fma order stay unchanged. the shared new tail avoids repeating the stream normalization in both kernels. `moe_comb`, `moe_comb_slots`, `gr_apply_moec`, `gr_norm_h_moe`, host dispatch and gemm_hc_up are unchanged in the candidate.

### rejected or inconclusive variants

- precomputed uint row offsets: initial separate-process apparent ~5% gain disappeared after warm-up. same-process paired T2048 median ms4: 13.8539 candidate vs13.8124 baseline. not retained.
- order-preserving compaction via first-simdgroup exclusive prefix and sum: median 13.8535 vs13.8479; no win, not retained. dropped slots removed without adding zero-weight fmas.
- residual reread instead of rv: moec essentially flat, h4 apply1 ~1% slower, apply0 ~5% slower. not retained.
- topk10 unroll: no clear improvement over stream candidate, not retained.
- fully unrolled d traversal: contention and no demonstrated incremental win, not retained.

### exploratory stream result before final helper cleanup

paired timing format is candidate/baseline milliseconds for four dispatches; three samples alternate AB/BA. median values below, not kprof or whole-model performance:

| source/log | T | moec | h4 apply1 | h4 apply0 |
|---|---:|---:|---:|---:|
| paired_stream_2048.log (stream+uint) |2048|13.5182/13.8329 (-2.3%)|6.9600/7.1963 (-3.3%)|3.7201/4.0299 (-7.7%)|
| confirm_stream_1024.log (stream+uint)|1024|6.8074/6.9979 (-2.7%)|3.5236/3.6321 (-3.0%)|1.8177/2.0026 (-9.2%)|
| paired_stream_orig_512.log (stream, no uint)|512|4.2696/4.2976 (-0.7%)|1.7997/1.8695 (-3.7%)|0.8299/1.0031 (-17.3%)|

these are leads, not isolated confirmation. the moec change does not meet the planned >3% threshold; h4 does in these exploratory runs. candidate intentionally drops the neutral uint/compaction change. `gr_apply_moec` shows no stable improvement and is unchanged.

### correctness and process budget

final source: `candidate.metal`, current scratch `metal/kernels/prompt.metal`. `/tmp/sol_r8_grvec/final_{1,7,64,512,1024,2048}.log`: all six processes exited 0, all four comparisons bitwise PASS (R and fp16 output, h4 apply0 and apply1), memcmp against original dispatch chains; random scattered routes include pos=-1. final process durations 0.084–1.022s, below5s. every standalone run in this investigation was capped at4.9s and completed below2s. no server access or full-model command used.

all final timing samples are contaminated: unchanged `gr_apply_moec` jumps from ~12 to ~300ms4 across the investigation and final1024/final512 show broad slowdowns. do not use final timing logs to claim a speedup. parent and the two active sibling tracks were asked for a short exclusive slot; clean final confirmation pending.

reproduction:

```
clang++ -std=c++20 -O2 /tmp/sol_r8_grvec/strata/metal/tests/comb_fold_parity.cpp \
  -I/tmp/sol_r8_grvec/strata/metal/src -I/tmp/sol_r8_grvec/strata/metal/tests \
  vendor/strata/build-metal/libmtl_rt.a -framework Metal -framework Foundation \
  -o /tmp/sol_r8_grvec/comb_fold_parity
STRATA_METAL_KERNELS=/tmp/sol_r8_grvec/strata/metal/kernels \
  /tmp/sol_r8_grvec/comb_fold_parity 2048 grvec /tmp/sol_r8_grvec/baseline/prompt.metal
```

patch: `/tmp/sol_r8_grvec/sol_r8_grvec.patch`; `git -C vendor/strata apply --check` passed. scratch `git diff --check` passed. final visual diff/context review completed; removed redundant braces and duplicate comments and centralized the normalization tail. no tracked vendor files edited.

## final short confirmation (after parent authorization)

`confirmation_2048.log` and `confirmation_1024.log`, exit0, wall0.689s and0.323s; all bitwise checks pass again. opus track explicitly paused. final T1024 paired medians (candidate/baseline ms4):

- moec: 6.7921/6.9860 = -2.8% (below planned3% acceptance threshold; marginal).
- h4 apply1: 3.4993/3.6128 = -3.1%.
- h4 apply0: 1.8099/2.0053 = -9.7%.
- unchanged apply_moec: 6.0877/6.1005 = -0.2% (noise control).

T2048 h4 apply1 6.9608/7.1618 = -2.8%, apply0 3.7630/4.0195 = -6.4%; moec has warm-up/contention drift (17.7 ->14.1ms4) and is inconclusive there. no model-wide extrapolation claimed. parent plans fully serialized isolated confirmation once all patches arrive. recommend testing this candidate, not declaring a completed production speedup; h4 is the stronger lead, moec remains marginal and may be dropped independently.
