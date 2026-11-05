# AR33 T04/T05: cycle replay and native scheduling opportunity

2026-11-05. Preregistered in issue138/opti/matrix before prototype and replay. Native source remains82fff; this increment adds an **offline Python analyzer**, deterministic policy/error tests and this report. No model, GPU, Instruments, Cargo compile or heavy prover was started. Brave and Deezer remain open. Cancelled profiling is not resumed.

## T04 provenance, protocol and reconstruction

Input: [current-capture.json](../issue58-20261029/current-capture.json),9,958,454bytes, SHA256 **fd8cee874b73379db02a6f2f7e22fe8426870331fba79ac167e166e36f4eb884**. Its [source sidecar](../issue58-20261029/current-capture.json.sources.json) pins the Oct27 MAIN/HOLD traces, complete/gold evidence and compressed/decoded hashes. This is the qualified historical workload, **not a new native32-token capture**. AR20 previously matched only the first336 MAIN decode vectors to its short native diagnostic; that limited check does not establish today's complete cache contents.

Reused the schema, initialization and exact physical reconstruction contract from native/src/flash_cache_profile/shared/current.rs:48layers,512experts,10distinct routes,53slots/layer. MAIN has87prompt tokens and HOLD82; each phase has127decode forwards. Each policy/case starts empty, consumes two complete first-phase generations, then retains its own state through trace_1/2/3. Every one of **609,120 observed(slot,hit) pairs** and all48layer cumulative before/prefill/after counters match the native decayed-LFU reference before candidate scoring. Warmup observations are used for initialization, not falsely asserted to have observed slots. This also reproduces the earlier [SLRU reference totals](../flash-ar20-20261105/slru-audit/README.md).

Three arms, no parameter/capacity sweep:

1. Actual native-reference policy,48independent LFU53 pools.
2. Global LFU control,2544slots, preserving native per-layer count aging every16visits, oldest-use/smallest-slot ties.
3. Same global pool, **cyc64 armed only during decode**. Per-layer past-route EMA alpha1/64; victim maximizes d+48(1-p)/p, p=0 gives infinity; ties oldest-use then smallest-slot. Prefill deliberately uses arm2's native decayed-LFU fallback, not a claim to reproduce JitEngine's different decay32 fallback.

The candidate borrows the exact cyclic rule from [JitEngine victims76–185](https://github.com/skydiscover-ai/jitengine/blob/2c34508d617800ab7448a69e14a6dd551aec156e/engines/qwen3.8-flash-next-m4pro/src/jitengine_qwen3_8_flash_next_m4pro/moe/offload_cache/victims.py#L76-L185); it does not read future routes. All selected IDs are protected before the first miss. Leased/pending slots are excluded in policy tests. The historical workload is fully drained demand-only, so it contains **no timed in-flight GPU/IO leases**; that concurrency is not simulated or proved.

Per-layer victim heads are an acceleration of a full candidate scan. Tests compare them against independent full-slot enumeration over720deterministic visits including mode changes, leases and pending slots. Baseline physical observations provide a separate native oracle; corrupt expected slots are detected. For this bounded workload every layer is visited635decode times, without adjacent same-layer visits. The scale-rescaling branch is not reached; our1e-100 threshold differs from upstream1e-200, and our helper does not implement upstream's repeated-same-layer suppression. Neither affects this fixed trace, but **general unbounded JitEngine policy parity is not claimed**.

Command actually executed once:

```text
/Users/justin/Documents/mix-stq1_0/.venv/bin/python scripts/analyze_flash_cycle_cache.py \
  --capture docs/measurements/issue58-20261029/current-capture.json \
  --out docs/measurements/flash-ar33-trials-20261105/cycle-replay.json --seconds 45
```

Completed15.6174815s, no timeout. [Full receipt](cycle-replay.json) contains per-phase/per-layer counts, hit histograms, final per-layer residency and source SHA. [Exact source executed](cycle-replay-source.py.txt) is preserved. Subsequent source changes are only five dict-constructor→literal lint repairs; no policy calculation changed and no second replay was run. The final lint-only source and newly added deadline test await root's coordinated focused check before GPU grant.

## T04 result: no opportunity to promote this global cycle adaptation

Totals over three observed phases; each decode cell contains182,880expert lookups. These phases repeat the same deterministic routes with differing policy history; they are **not three independent statistical trials**.

| Case / part | Native LFU53 misses | Global LFU misses | Global cyc64 misses |
|---|---:|---:|---:|
| MAIN prefill |40,828|41,505|41,199|
| MAIN decode |54,884|54,690|55,867|
| HOLD prefill |38,522|39,095|38,999|
| HOLD decode |59,005|59,600|60,619|

Cycle increases decode reads by983on MAIN and1614on HOLD: approximately **+1.79%/+2.74%**, with hit rates69.45%/66.85% versus69.99%/67.74%. Global LFU alone is mixed:194fewer MAIN misses,595more HOLD misses. No capacity was added: all arms have2544slots,3,174,942,528expert bytes. The final cycle decode distribution is24–71slots/layer on MAIN and39–69on HOLD; exact48-entry distributions are in the receipt. Prefill shares the same fallback implementation but starts from each arm's preceding decode state, so different prefill miss totals are expected.

Audit found no wrong cache initialization, missing prefill, changed route order, ignored cohort pin, disabled EMA or favorable budget comparison explaining these counts. Each cycle case records30,480actual decode EMA observations (5×127×48), zero prefill observations. The cyclic rule is distinct from old ARC/CLOCK/SLRU, but this equal-budget historical workload is negative. **T04 stops at the offline opportunity gate; no global runtime pool is justified.** Counts predict neither tok/s nor physical NAND traffic, and omit host-policy overhead, cache/page faults and numerical-model qualification.

## T05 partial-hit opportunity: not disproved by the all-hot rate

Native-reference decode hit histogram, summed across three phases:

| Case | Per-expert hits | Entirely hot cohorts | Cohorts with at least one hit | Source-eligible regular-GDN successors |
|---|---:|---:|---:|---:|
| MAIN |127,996/182,880 (69.99%)|2,149/18,288 (11.75%)|18,225/18,288|1,751all-hot/12,954boundaries|
| HOLD |123,875/182,880 (67.74%)|1,883/18,288 (10.30%)|18,179/18,288|1,638all-hot/12,954boundaries|

Eligibility here is a **source structural count**, not an actual dispatch counter: successor regular GDN, no successor PLE(index2)/QSA, no layer beyond47. Source: flash_config.rs layer types, gpu/flash.rs::hidden and gpu/gdn.rs deferred boundaries. It differs from the old Sushi eligibility protocol and is not substituted for that measured diagnostic.

Low full-hot acceptance does not make partial repair useless: roughly68–70% of expert contributions could in principle be retained. But native already avoids reading resident experts after the router wait. New benefit must come from executing those hits **before** host route consumption, and possibly anticipating a successor—not from counting all retained hits as newly saved compute. Missing contributions must be repaired in original route positions and the entire canonical mix recomputed; adding corrections to an existing sum changes rounding. Native flash_moe.metal:41–53 uses8stripes+last2withHalf contributions, unlike Jit's sequential affine-Q4 combine.

### Concrete native scope contradiction

1. **A slot map is not an addressable expert bank.** Native53slots are53separate Shared record buffers. flash_expert_axis.metal binds only ten selected records explicitly at buffer1..10, prepared0/partial11/params12. Current kernels cannot use a device-selected slot ID to dereference any of53records. An argument/pointer table with explicit residency/lifetime, or multiple smaller bank dispatches, is required. This is not a flag on the existing A06 kernel, and expanding bindings must be checked against real Metal/device limits.
2. **Current route producer returns synchronously.** hc.rs::read_after_write_with_gate uses gpu.rs::command, which commits and waits before returning. Adding Hot work to that same CB extends the wait instead of overlapping host consumption. A useful prototype needs an owned pending producer and a separately identifiable route-completion boundary; error/status/retained-input behavior must survive deferred return. Existing accepted joint HC/GDN already removes some boundaries, so old inclusive route waits are not a budget of recoverable milliseconds.
3. **Allocation ownership does not protect contents.** A11 can reuse old Shared expert backings after drain/drop. A provisional device-map consumer may still read a slot when the CPU wants to overwrite it. Buffer clones retain allocation, not immutable bytes. Require generation-qualified maps and leases protecting every possible consumer until its read set and completion are established; reserve-only speculative destinations cannot steal a leased hit slot. EOF/timeout/cancel/map replacement must drain ownership without stranded GPU waits.
4. **Repair can invalidate the successor.** Exact current-layer repair changes its output. A previously executed successor requires snapshot/rollback of GDN recurrent/conv/init and HC state; QSA/PLE add distinct history/position state. The smallest source domain would restrict successors to regular GDN and preserve canonical kernels/mix. No successor state or cache-policy update can be published from provisional data.

**T05 decision:** partial repair remains a real conditional opportunity, not a failed12%-acceptance trick. The current direct-buffer and synchronous-helper architecture does not permit a small safe drop-in implementation. Gate it out of the present bounded kernel/guard batch; a larger owned-producer/device-address/lease transaction needs its own design and independent numerical/continuation/error tests. No speculative runtime, throughput gain or model validation is claimed by this source gate.

## Verification and handoff

- Before replay:7focused unittest cases PASS in0.008s, including720optimized/full-scan comparisons, explicit physical-observation mutation, mode/EMA examples, pin/pending/no-victim cases, capacity/generation and invalid input/output preservation.
- First Ruff run:5C408 dict-style errors; repaired as literals without changing computation. Final lint/format/test check is pending root coordination, not silently claimed PASS.
- A real nonempty-cohort deadline regression was added after the initial7-test run; it is not yet executed. No extra replay follows the parent's freeze instruction.
- CrossHair command attempted on the actual Python script; module absent in the existing venv. Nothing installed, no source-level proof claimed. No Rust executable change, so no new Kani run.
- Root must wire the new script/test into the existing Python CI lint/test lists before publication; this child did not modify shared workflows, opti or matrix. No commit/push was performed here.

Files handed off: scripts/analyze_flash_cycle_cache.py, tests/test_flash_cycle_cache.py, this report, cycle-replay.json and its exact source snapshot. Root owns final gates/status/CI and the exclusive GPU schedule.


## Dated final validation revision — 2026-11-05

Root completed all8focused cases (including deadline), lint/format and the final-source replay after the documented C408 changes. The second replay took16.7446793s and its complete variant/case/phase counts equal the preserved first result. [Final receipt](cycle-replay-final.json) and [portable checks](cpu/final-portable.json) supersede the earlier pending descriptions, which remain as history. No new optimization or GPU speed result is inferred.
