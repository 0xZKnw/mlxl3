# Issue125 L2 — bounded SSD wave: inconclusive screen

The experimental schedule passes its four essential correctness gates. Its AR32 screen is **inconclusive**: reference drift reaches7.32% and candidate drift reaches7.66%. L2 remains a local, default-OFF prototype. This publication contains results and a passive patch archive; the active runtime, tests, proof manifest and defaults are unchanged.

Both measured binaries are based on qualified P13, before L1. The publication branch includes merged L1; it is not the measured candidate. No composed L1/L2/L5 performance is claimed. Refs issue125, L2 only; the issue remains open.

## Matched short screen

Apple M5/24GiB, Flash Next checkpoint65c89531/EXL3 average2.05bpw, context4096, cache3GiB/48×53slots,4readers, depth0/dynamic0/prefixoff, T0/topk1/penalty1. A/B/B/A, one warmup and one measured32-token response per MAIN/HOLD prompt per process;31decode transitions. AC100%, no thermal/performance warning recorded; temperature and GPU frequency unmeasured. Compilers/provers/other inference were drained; background desktop activity was not eliminated.

| Prompt | P13 median tok/s | L2 median tok/s | Median change | Reference drift | Candidate drift |
|---|---:|---:|---:|---:|---:|
| MAIN |6.389712|6.192056|−3.0933%|+7.3178%|+0.4018%|
| HOLD |6.126813|6.321533|+3.1782%|−4.0895%|+7.6646%|

All16 warm/measured texts and token hashes match; all processes/readers were joined. Metal allocated bytes are6,608,617,472 for all8 measured responses; this is not process RAM. MAIN total latency/TTFT change+5.1536%/+5.8903%; HOLD−0.7271%/+0.1146%. Neither prompt has every candidate above both controls. These data establish neither a throughput gain nor an isolated causal regression. There was no automatic second AR32 campaign or default activation.

[Protocol](performance/protocol.json), [four-process campaign](performance/campaign.json), [summary and every sample](performance/summary.json), reports [A1](performance/1-reference/report.json), [B1](performance/2-candidate/report.json), [B2](performance/3-candidate/report.json), [A2](performance/4-reference/report.json).

## What was exercised

`MLXL3_NATIVE_SSD_EXTRA_WAVE=1` was experimental and exact-M5-only, depth0/no-prefetch. After W1, if at least3 results remained unreceived, it gathered at least2 completions, encoded them and allowed at most1 extra command buffer before the full tail. The original absolute deadline, reads/workers, immutable record owners, expert math and canonical `plan.routes` reduction were retained. Literal0/absence used the old schedule.

[Four essential gates](quality/report.json) pass before speed:

- Forced1/2/7 record cohorts, F16/F32,4OFF/ON cases and10,240finite words; exactly2 actual extra submissions, cold3→4CB and hot1CB.
- Same Metal/model, context4096, cold caches in both arms,4forced untraced tokens:191 actual extra submissions47/48/48/48 versus0,1,986,560logits and115,212,288continuation-state words exact, hidden exact and independent initial84-state/logit archive.
- Partial-record error/recovery passes; cold4CB and hot1CB.
- Actual native CLI at context4096 passes the independent2-token golden`9f850af1cbba074b`, noMLX and joined processes.

Long CLI reject/cancel/budget extensions were deferred until after the screen and remain unexecuted for this prototype. They are not claimed from another engine's qualification. [Initial plan](quality-plan.json), [executed essential plan](quality/actual-plan.json), [30 rejected validator mutations](quality/validator-selfchecks.json).

## Short adaptation diagnostic

A source audit found a gap in the first review: `remaining()` counts unreceived completions, not active workers. The queue can fill while the middle experts are encoded, without being repolled before `extra_submit`. The earlier claim about an already-complete queue applied only when `ready_min` itself drained it. This correction is preserved in the journal.

Two existing native8 selections, OFF then ON, reused the same frozen libtest; no rebuild, new prototype or AR32 repeat. Both pass independent initial checks, complete finite continuation comparison and intrusive-profile checks. All336 row/layer pairs have identical tokens/routes/hits/misses/bytes:1,097records/1,369,069,164bytes.

| Diagnostic observation | OFF | ON |
|---|---:|---:|
| MoE command buffers |886|1026|
| Extra submissions |0|141|
| Extra tags after all worker read ends |0|6(4.26% of extras)|
| Extra tags before the last worker read end |0|135|

Extra cohorts contain2records138times,3records twice and4records once. Six tags occur after all worker read ends, at most0.0325ms later. For the other135, the CPU tag precedes the last read end by median0.3245ms/p950.90325ms. The tag is taken **before** the commit call: these135 cases do not prove GPU execution overlapped SSD work. The six late tags rule out remaining reads from this batch at that instant, but the extra CB can still overlap CPU tail encoding. The missing repoll exists; this trace does not establish it as the dominant explanation.

Ordered MoE labels were matched to completed commands on the same Metal clock. OFF W1→Final:249pairs, median gap0.398958ms/p951.206625ms. ON W1→Extra:141pairs, signed median0.210250ms/p950.628958ms,8overlaps; Extra→Final:141pairs, median0.261625ms/p950.884208ms, no overlap. These are application-CB observations, not devicewide idle, CPU/GPU clock alignment, exclusive costs or recoverable milliseconds.

[Diagnostic protocol](diagnostic/protocol.json), [two runs](diagnostic/report.json), [all cohort records and clock-safe analysis](diagnostic/analysis.json), profiles [OFF](diagnostic/off/profile.json.gz), [ON](diagnostic/on/profile.json.gz). An initial offline cohort-bookkeeping assertion double-counted Hot+Final cases; [failure and correction](diagnostic/analysis-failure-01.json) are retained. No GPU run was repeated for it.

## CPU evidence, identities and limits

[Final CPU receipt](cpu/final-report.json), [earlier partial receipt](cpu/partial-report.json), [frozen artifacts](freeze.json). CPU340passed/8ignored, DIRECT356/54, MLX392/273; three strict Clippy configurations, fmt, normal inventory252 and Python6pass. The final ignored model fixture was added after the DIRECT suite and compiled by final Clippy/release; affected SSD12tests were rerun after its thread-cleanup correction. No ignored physical case is counted from these suites.

Kani0.68/CBMC6.11 checks the actual three pure helpers:255checks/13covers, seven unreachable standard-library guards inspected. Policy/minimum inputs use unrestricted `usize` and symbolic booleans; flag verification covers absence and valid UTF8 from0..4symbolic bytes, unwind10, no assume/stub. This is bounded CPU checking, not a proof of channels, worker interleavings, OS IO, Metal lifetimes or complete inference. Python controllers have finite adversarial checks and reused process helpers; no source-level Python proof is claimed.

Initial fixture compilation failed on moving leases out of a `Drop` type; corrected by cloning test leases. A review found early test-error returns could bypass thread joining; corrected to disconnect/join before propagating results, then affected tests and threshold/minimum mutations were rerun. Both mutations fail101 and restored tests pass. Old failure logs remain in the CPU receipts. A metadata append had a pre-writeSyntaxError; no runtime changed or GPU rerun. An inherited environment snapshot was reduced before any publication; the archived controller uses an explicit variable whitelist. [Metadata revision](controllers/metadata-revision.json) contains no ambient values.

The local prototype source is`b37903a2c64b626cc187dce5b65da634971715a2` on base`65b50fa5413747c709729707fde5cd033d8b022e`; CLI SHA`4c5f1f7d28e12e2727fceca5ddc032019a94d26c38426c4fadf05654fb29880a`, libtest SHA`4bb5b11fffb2503f91efd2019039458ee48c21cf22bdd02a95c08378081eccbf`, both minos26.2/sdk27.2. Local commits/binaries are preserved; they are not added as active source here. [Archived patch](prototype/experimental.patch.gz) applies only to an isolated checkout of its stated P13 base; [identity and SHA-256](prototype/identity.json). Do not apply it to the current L1 runtime without a new trial.

[Payload manifest](manifest.json), [raw-log identities](raw-log-manifest.json). Controllers are passive text under`controllers/`; the quality controller's environment-serialization fix occurred after its numerical run and is explicitly recorded. No app replacement, release, merge or delivered L2 gain is claimed.
