# Issue165 — Q01 parallel reduction pilot

Optional `MLXL3_NATIVE_PROFILE=parallel-v1`, exact Apple M5 only. Default remains `exact`. Profile selection is immutable per engine process and is reported in bridge ready metadata; model-owner checks isolate snapshots and the process-local prefix cache. The profile changes Flash reduction order, including Hot, static and MB2/expert-consumer kernels. Weights, decoding Half rounding, H128 and recurrent F32 are unchanged. Both prefill and decode use this first candidate, explicitly permitted under the user's relaxed experimental restrictions. No global fast-math.

## Result

Same-binary A1/B1/B2/A2, each AR+MTP, one32-token warmup+one measured response per mode, MAIN87/context4096/greedy/cache3GiB/53slots; no process ceiling, serial GPU. Measured source/binary identities: [identity](q01-identity.json). AR/MTP have different draft residency. Do not compare these32-token scores directly with previous64-token pilots.

| Mode | Exact A1/A2 tok/s | Parallel B1/B2 tok/s | Mean change | Control drift |
|---|---|---|---|---|
| AR | 9.1106 / 9.2317 | 9.7916 / 9.6592 | +6.04% | +1.33% |
| MTP | 10.9349 / 10.7601 | 10.9030 / 10.8889 | +0.45% | -1.60% |

All measured32-token outputs match hash `bfa49090cb33d4fe`; all processes/readers joined and actual bridge profile metadata checked. AR improvement is promising in this short pilot; **MTP gain is not established**, smaller than control drift. No20tok/s claim. Retain as an optional experimental AR candidate; no default promotion. Per-arm raw reports retain prefill/TTFT/total/decode and actual memory separately. Thermal/power were not captured for this new pilot; no strong noise/generalization claim.

## Quality and correctness

384teacher-forced tokens, six authored screening documents,16prefix+64scored tokens each, uninterrupted per-document history. Same checkpoint artifact key/corpus/token IDs; full248320Half logits with shape/finite/EOF checks; stable F64 full-vocabulary NLL/KL. Aggregate PPL ratio **0.99650434 (−0.350%)**; worst domain numbers/negation **+1.312%**; raw top1agreement ranges93.75–100%. [Per-domain results](q01-quality-summary.json). Practical preregistered review thresholds under the user's relaxation: aggregate+1%, each domain+2%; they pass. This is not a task success test or independent holdout; improved mean perplexity does not establish improved overall quality. Original full logit payloads remain local, hashes archived.

Fast-profile B2/B3/B4 all-prefix logits/state/hidden/continuation/drop/late-error oracle passed, including actual temporal/Hot/MB2 dispatches. This establishes tested coherence **within the new target**, not equality with the old target. CPU library368pass/135ignored; strict Clippy/fmt and Metal+chat CLI build pass. Kani619obligations,12covers, domain up to12arbitrary bytes/unwind13;12compiler-unreachable checks. This bounded CPU verification does not prove Metal, allocator, concurrency or model quality. Metric unit tests use analytic uniform/two-class oracles and empty/missing/invalid/NaN/Inf failures.

Earlier Clippy/caller failures, UTF8-verifier180s timeout, genuine unwind10 failure, and static-header byte regression are retained. The legacy header was restored byte-for-byte; the variant macro is injected at runtime shader-source assembly. Final affected checks pass; no hidden discarded GPU measurement. CBMC/ESBMC are unavailable and no Metal source proof is claimed.

## Scope and sources

Baseline diagnostic with full reproduction source: [issue166](https://github.com/0xZKnw/mlxl3/issues/166), after merged PR161/main0fd9fbd1. Host inclusive spans are diagnostic and cannot be summed as exclusive GPU cost.

Direct source review: [Sushi904b0f6](https://github.com/beamivalice/sushi/tree/904b0f6e4819b5dea627806f90d75d0b5c2ae7ba) keeps exact geometry independent of row width and excludes MUL1 from certain expert fast paths; [TurboEXL3c7ce66c](https://github.com/davidtai/turboexl3/tree/c7ce66c12f6992eabda5db1d11934a6fe5efec33) defers affine/weight rounding, a separate Q02 experiment not copied into Q01; [Regolith92a523f](https://tangled.org/ptr.pet/regolith/commit/92a523f036092df5a072d24db1cb4823863ca862) exposes suffix-row routing and probability unscaling traps in its audit, which are not imported. Upstream gains are not attributed to MLXL3.

Only Q01 is implemented here. Q02–Q15 remain pending/conditional; no new routing, training overlay, codec or relaxed acceptance was implemented. Work was limited to this usable checkpoint at the user's explicit rate-limit request. No push, hosted CI, merge, release or installed-app replacement for this new branch.
