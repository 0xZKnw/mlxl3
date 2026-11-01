# Independent P1 results: AR, MTP and DFlash

All 18 plans completed and all 18 external inspection receipts passed. The controller was joined by the parent. The primary timings below are the 12 pilot128 processes: four measured responses and four retained warmups per arm in each mode, with MAIN=87 and HOLD=82 prompt tokens. Each response generates 128 tokens and times 127 decode transitions. All 72 campaign response rows remain in the JSON: 48 pilot128 plus 24 bridge17.

Every rate is total tokens divided by total time. Modes are compared separately; percentages are not added and there is no combined causal speedup estimate.

| Mode | A decode tok/s | B decode tok/s | B/A raw | B/initial A | Final A/initial A | Prefill B/A |
|---|---:|---:|---:|---:|---:|---:|
| AR | 11.209891 | 11.663980 | +4.051% | +1.527% | -4.737% | +0.255% |
| MTP | 11.438365 | 11.865311 | +3.733% | +4.699% | +1.881% | +4.193% |
| DF | 12.197625 | 12.312687 | +0.943% | +1.486% | +1.082% | +1.258% |

AR is amplified by the slower final A: its +4.05% pooled contrast becomes +1.53% against the first A. MTP and DFlash have faster final As, so their pooled contrasts are smaller than their contrasts with initial A. These are descriptive endpoints, not bounds on the true effect.

| Mode / prompt | A decode | B decode | Decode B/A | Prefill B/A |
|---|---:|---:|---:|---:|
| AR / MAIN | 11.436863 | 11.729016 | +2.554% | -0.538% |
| AR / HOLD | 10.991752 | 11.599662 | +5.531% | +1.117% |
| MTP / MAIN | 11.710449 | 12.289354 | +4.943% | +4.482% |
| MTP / HOLD | 11.178638 | 11.469555 | +2.602% | +3.889% |
| DF / MAIN | 12.443017 | 12.774684 | +2.665% | +2.395% |
| DF / HOLD | 11.961724 | 11.882939 | -0.659% | +0.068% |

## Speculative work and logical reads

Acceptance counters are exactly equal per prompt across both arms and repeated processes. MTP MAIN accepts 54/72 proposals (75%); HOLD accepts 48/78 (61.538%). DFlash MAIN accepts 58/69 (84.058%); HOLD accepts 56/70 (80%). Thus measured totals per arm are MTP 204 accepted / 300 proposed, and DFlash 228 / 278. Depth/width remains one; all AR speculative counters are inactive.

DFlash reports identical logical expert-demand reads: MAIN 23,889,445,704 bytes, HOLD 25,808,888,160 bytes per measured response, or 99,396,667,728 bytes for each arm’s four measured responses. Prefetch counters are zero. The counter measures successful expert-job demand bytes in the target decode path; it does not distinguish macOS page-cache service from physical NAND. AR and MTP do not expose this byte counter in these reports, so their read volumes are not inferred.

DFlash reported verification time sums are A 38.699221 s and B 38.272893 s (-1.102% time). Draft time sums are 1.956179 → 1.988973 s, commit/cache 0.813268 → 0.818880 s. These stage totals are not independent speedups to add.

## Memory and drift

| Mode | MAIN compressed MiB A1 / B2 / B3 / A4 | MAIN decompressions A1 / B2 / B3 / A4 | Max footprint MiB A / B |
|---|---|---|---:|
| AR | 0.0 / 413.5 / 925.4 / 1011.5 | 0 / 815 / 39005 / 47037 | 6787.01 / 6815.96 |
| MTP | 1960.0 / 975.4 / 1189.8 / 1663.9 | 126616 / 55260 / 105885 / 113841 | 7414.74 / 7428.29 |
| DF | 1015.6 / 1168.9 / 1215.3 / 1230.5 | 71121 / 78001 / 89560 / 110089 | 7379.24 / 7399.43 |

Compression and decompressions differ substantially between runs; in particular initial MTP A has more compressed memory than either B. These histories limit clean causal attribution, including the apparent MTP prefill improvement. AC power and no recorded pmset thermal/performance warning are present at process boundaries; no GPU-frequency trace establishes a thermal cause. System-wide swap and process ledger values remain separately preserved in the JSON.

## Identity and interpretation

Before the parent was told it could patch the separate diagnostic, all 738 frozen P1 source identities (213,364,307 bytes, including the MLX runtime libraries/metallib) and all three binaries were freshly rehashed and matched. The receipt is `all-modes-source-recheck.json`; this statement does not describe sources after a diagnostic edit. All report hashes match their external inspection receipts. Same CLI/model, 3 GiB cache, four readers and environment controls are retained; only Cold readiness differs between arms. All full 128-token outputs match their independent goldens.

The numerical/state qualification remains the parent’s separate six completed quality tests, including 175 natural waves. This aggregation executes no model or GPU and cannot establish detailed wave coverage within each timed CLI. The observed gains are positive across pooled modes, with per-prompt variation and drift retained. This supports continuing the adaptation audit; it does not itself decide global activation or promise a fixed percentage. No row was excluded, no source or original report was edited, and no test was rerun to improve a number.
