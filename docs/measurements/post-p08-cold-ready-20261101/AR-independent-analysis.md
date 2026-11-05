# Independent AR128 aggregation

All four preregistered pilots A1/B2/B3/A4 are included. Each arm has four measured responses and four retained warmup responses. Each response generates 128 tokens and times 127 decode transitions. Rates below divide total tokens by total time; they are not averages of reported tok/s.

| Measured metric | A pooled | B pooled | B versus A |
|---|---:|---:|---:|
| Decode tok/s | 11.209891 | 11.663980 | +4.051% |
| Prefill tok/s | 11.114494 | 11.142839 | +0.255% |
| Decode time, 508 transitions | 45.317123 s | 43.552886 s | 1.764237 s less |
| Mean TTFT | 7.603032 s | 7.583768 s | -0.253% |

The raw decode difference is **+4.05%**, but final A throughput is **4.74% lower** than initial A. Comparing pooled B only with the initial A gives **+1.53%** decode and **+0.218%** prefill. The larger pooled contrast is therefore amplified by the slower last A. No causal drift correction, thermal explanation or uncertainty bound is invented.

| Measured process | Decode tok/s | Prefill tok/s | MAIN compressed MiB | MAIN decompressions |
|---|---:|---:|---:|---:|
| 1-A | 11.488580 | 11.118627 | 0.00 | 0 |
| 2-B | 11.683867 | 11.217047 | 413.53 | 815 |
| 3-B | 11.644161 | 11.069606 | 925.42 | 39,005 |
| 4-A | 10.944403 | 11.110363 | 1011.55 | 47,037 |

| Measured prompt | A decode | B decode | Decode change | Prefill change |
|---|---:|---:|---:|---:|
| MAIN | 11.436863 | 11.729016 | +2.554% | -0.538% |
| HOLD | 10.991752 | 11.599662 | +5.531% | +1.117% |

Warmups are retained separately: pooled A 11.317248, B 11.385653 tok/s (+0.604%). All 16 raw rows, warmup-inclusive totals, per-prompt initial-A comparisons and exact timing sums are in the JSON; no rows were discarded.

All four external inspections passed. Report hashes match their inspection receipts; all outputs match the independent full 128-token goldens. The same final CLI, model, context limit, 3 GiB cache and four readers are used. The only reported environment difference is Cold readiness 0/1; GDN, F02 and dynamic readers remain zero. AR has MTP/DFlash inactive and null/zero acceptance counters. Native numerical/state parity belongs to the separately completed quality campaign, not this arithmetic analysis.

Fresh verification checks 735 source identities (7,515,091 bytes), all three binary identities, and hashes every report/inspection/supervisor used. Three large MLX runtime artifacts are deliberately not reread during ongoing GPU timings; their frozen expected hashes and the already-passed preflight context are retained in the JSON. No actual runtime artifact is declared freshly rehashed when it was not.

Native footprint maximum over warmup and measured rows: A 6787.01 MiB, B 6815.96 MiB. MLX active bytes stay near 6.128 GiB. Native compression increases strongly through the sequence, including final A, so the footprint difference is not attributed to the candidate alone. All native ledger swapin counters are zero; system-wide swapins are a different measurement.

During these four AR processes, system Swapout totals 192 KiB (B3); system Swapin totals 90.33 MiB. Cumulative GPU1 Swapin reaches 229.14 MiB by A4. Both endpoints show AC power and no recorded pmset warning, but no GPU clock trace is available. This does not establish thermal causality.

The AR signal is promising and small against the initial A; the pooled +4.05% must retain the observed drift and memory-history limits. Only two processes per arm were measured. MTP/DF and the adaptation audit remain separate; this report makes no global integration decision. No GPU, compilation, prover, or controller change was performed by this analysis.
