# AR30 candidate preregistration — 2026-11-05

Issue [135](https://github.com/0xZKnw/mlxl3/issues/135) was created before any new test, profile or prototype. Baseline engine is `2035348319a76bd505ad6991ccdd510345225568`: accepted AR10 defaults ON, AR20's twelve options OFF. Historical 10.303/10.020 tok/s are short observations, not fresh controls. The three research reports pin primary implementations and distinguish mechanism evidence from measured speedups.

| ID | Hypothesis / concrete change | Independent correctness gate before speed | State |
|---|---|---|---|
| AR30-PROFILE-01 | Reuse qualified final native-libtest `4577d8cd…`, unchanged native8 diagnostic, current default flags; find the shifted hot path. This repeats AR20's intrusive profile because current accepted defaults/quiet application conditions differ, not to manufacture a boost. | Existing nonempty independent archive/continuation gate, positive event and ancestry checks, exact compiled source/artifact hashes. | Preregistered; diagnostic, not throughput measurement. |
| C01 | ExLlamaV3 K2/3/4 cooperative packed-word ownership and SIMD shuffles in native EXL3 QMV. Preserve exact codeword/FMA/reduction and fallback. Distinct from AR20 reader8/B07 preload. | Exhaustive windows vs canonical state, resident/routed frozen projection fixtures, real path counters, complete model/bridge trajectory. | Preregistered before prototype. |
| C02 | MLX multi-output GEMV mechanism for router: two output rows per SIMD, shared input load, original per-output FMA/shuffle order. | Every router logit/probability/ID/score including strides and tails, then complete trajectory. | Preregistered before prototype; no automatic TM sweep. |
| C03 | TUFF-style scalar GPU greedy epilogue adapted to native **normalized Half** tie/error semantics, avoiding whole-vocabulary CPU conversion/readback where possible. Preserve transaction and full-logit API; eligibility before advancement, other sampling/speculation unchanged. | CPU normalized oracle, adversarial Half ties/nonfinite/first-max, rollback/cancel/fallback, actual bridge/CLI counters and full trajectory. | Preregistered before prototype; raw argmax explicitly forbidden. |
| C04 | MLX/TUFF lifetime-scoped temporary reuse: first investigate resident prepared/partial scratch, bounded budget and completion plus owner leases. Cache and fixed arena are alternatives, not additive boosts. | Alias/retained-view/pending-CB/failure/trim/poison and shape guards plus actual allocation counts; independent projection and state gates. | Design investigation first; implementation only if ownership has a small safe solution. |
| C05 | Cache immutable native PSO capability/geometry checks in active resident/GDN paths. | Same qualified geometry and fallback across shapes/devices; actual current-default caller counts. | Conditional after profile, no old AR20-OFF checks counted as savings. |
| C06 | Aligned QSA vector loads/stores and genuinely dense gated QSA specialization; same scalar arithmetic/order and explicit tail fallback. | Independent dense/tail/stride/all-value parity plus trajectory. | Conditional after profile. |
| C07 | Resident-only shared compute encoder across projection stages, explicit dependency barriers and same CB boundaries. New resident-only domain, excludes canceled P04/routed same-expert encoder trial. | Dependency/retained-view/error/encoder-count gates, component and trajectory parity. | Conditional; historical cancellation respected. |
| C08 | Packed EXL3 extraction metadata (six 5-bit fields/u32). | Exhaustive field identity, actual projection parity. | Reserve; compiler may already do this. |
| C09 | Exact SIMD ownership removes reference partial TG roundtrip without reassociation. | Explicit bijection and original eight-chain/reduction order; reject design if not preservable. | Reserve; upstream attests ownership, not this native arithmetic adaptation. |

Other attested reserves and rejected/inapplicable mechanisms stay in the research reports: bounded tracked MTLHeap only after allocation evidence, validated CPU encode replay for stable segments (no CB recommit), scoped QoS, uniform metadata; A08 producer KV append is an explicit historical extension. BaseRT publishes no engine kernels, wrappers are not independent mechanisms, GPU-only graph loops cannot cross SSD CPU routing readbacks. No inflated count of ten proven local boosts.

Protocol: correctness before timing; one GPU process/model at a time, no compilers/provers/other inference during GPU, no swap abort or artificial cooling wait. Retain all accepted defaults. Few coherent groups, actual CLI MAIN32 with one warm and one measured request (32 generated /31 decode), context4096, cache3GiB, readers4, dynamic readers0, lookahead0, prefix cache0, unpenalized greedy, AR only. Fresh controls and alternation if needed; strong drift makes speed nonconclusive. No sum of incompatible gains; compatible small gains may be composed. Record source/binary/controller/checkpoint hashes, environment and process cleanup. CPU proofs do not prove Metal or whole inference. Qualified increments are authorized for PR publication; no merge/app replacement/release.

## Dated outcome — 2026-11-05

All four implemented prototypes passed independent component/model/CLI numeric
qualification; all remain OFF. C01 standalone9.384326tok/s versus10.409299/10.253031
controls is rejected for performance. C02router10.323522, C03greedy10.247368 and
C04scratch10.355911 establish no gain outside control variation. Their compatible
composition9.911238 lies between10.253031/9.345471 controls and is nonconclusive.
All36 warm/measured outputs match the complete golden text/hash. No new default
or11–12tok/s claim. Source audits found no missing adaptation or arithmetic fault;
extra shuffle/hazard/host costs remain hypotheses rather than measured causes.
C05–C09 remain conditional/source-only, with no execution or speed claimed. The
new user request prioritizes detailed production profiling before further work.
See [evaluation](evaluation.json), [qualification](quality/report.json),
[report](README.md) and [C09 design](c09-design.md). Earlier hypotheses and failures
are intentionally retained above.
