# Native Flash Next AR composition — 2026-11-04

The final K/codebook specialization plus HC command chain is retained. This is a direct comparison of the whole ensemble against a fixed reference, not a sum of isolated gains.

| Prompt | Reference median | Ensemble median | Change | Paired changes |
| --- | ---: | ---: | ---: | --- |
| MAIN | 5.127148 tok/s | 5.941885 tok/s | +15.891% | +15.096%, +16.696% |
| HOLD | 5.136340 tok/s | 5.662650 tok/s | +10.247% | +11.547%, +8.927% |

MAIN reference drift is−1.408% and ensemble drift−0.037%. HOLD reference drift is−1.478%; its two ensemble samples are5.772072 and5.553228tok/s (−3.791% drift). Both ensemble samples exceed both references for each prompt. Every sample is retained. Median TTFT is17.0703→14.8392s on MAIN (−13.070%) and16.1043→14.0217s on HOLD (−12.932%).

The same native release SHA256 `942dbeee8ccb2c5907c2b071ae071707139e17f20fbeda4ecf326603f8d68d90` is used in all four independent processes, with macOS deployment target26.2 and SDK27.2. Reference: NT2=1, STATIC_K=0, HC_CHAIN=0. Ensemble: NT2=1, STATIC_K=1, HC_CHAIN=1. Order is reference/ensemble/ensemble/reference. Each process has one warmup and one measured response per MAIN/HOLD prompt:32 generated/31 decode tokens, context4096, T0/top-k1/penalty1, no prefix reuse,3GiB requested expert cache (3,174,942,528B effective;53 slots across48 layers),4readers, lookahead0, dynamic readers0.

All8 warmups and8 measured responses have identical full text and hashes per prompt. Every child and benchmark reader is joined. Native Metal allocated bytes remain6,608,617,472B in all8 measurements. Process footprint, lifetime peak and compression are separate raw measures in [the summary](composition-summary.json). AC power remains100%; thermal state is not measured. Background desktop activity was observed, with no compiler/prover/other inference during the pilot.

The combined defaults were physically qualified before timing, with HC/STATIC_K variables absent. The actual archived-model context reports HC initially enabled,192 real joint pairs and static submissions `[5760,0,0,480,723]`. Full AR checks cover prefixes0/2/9,86,409,216 finite F32 state words, full logits/hidden, rollback and exactly48 fewer helper commands per row with matched QSA ownership. The independent target oracle covers84 states/28,803,072 finite F32 words and496,640 logits. Real AR golden responses, invalid messages, context/delta cancellations, recovery, budget and memory-saver behavior pass. Candidate images load no MLX. Unchanged independent component oracles are reused.

Final CPU/direct/MLX-compatibility suites pass333/342/378 tests, with9/41/259 ignored. Format, three strict Clippy configurations, source inventory and Python checks pass. After activation markers, direct library checks pass253 tests/37 ignored. Kani on the actual HC policy helper passes237 checks and7 covers for arbitrary hardware/presence booleans, valid UTF-8 of0–4bytes and unwind10, with no assumptions;7 unreachable obligations are standard-library guards. Earlier GPU-source Kani ICEs, the old90s parser timeout and the initial failed HC command-count fixture remain preserved. These bounded host checks do not prove Metal, allocation or concurrency.

HC defaults ON only for exact`Apple M5`; other devices keep the previous default. Explicit`MLXL3_NATIVE_AR_HC_CHAIN=0` restores the unchained path. Static-K retains its separately qualified scope and`MLXL3_NATIVE_FLASH_STATIC_K=0` rollback. No quantization, routing or canonical shader arithmetic is changed.

[Preregistered protocol](composition-protocol.json), [all four process outcomes](composition-screen.json), [paired results and memory values](composition-summary.json). Detailed source hashes, actual CPU/proof commands and physical logs accompany the composition PR.

Limits: only32-token MAIN/HOLD and two measured samples per variant/prompt. HOLD variation is visible above. Sustained throughput, long-context performance and MTP/DFlash generation are unqualified. The installed app/distribution was not replaced. CI must be read for the final published commit; local checks do not imply a globally green workflow.
