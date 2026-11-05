# Flash Next AR: pipeline final MoE completion

**The final engine enables MoE completion deferral on exact Apple M5, above accepted L1.** On the short MAIN32 screen the actual default binary reaches **7.394552 tok/s**, versus current OFF controls 6.713478/6.650899: +10.14/+11.18%. This is a short-screen result, not a sustained-performance claim.

All five research candidates were implemented and exercised. Four additional paths remain **promising retained opt-in candidates**, with qualified correctness and inconclusive isolated throughput. The measured all-five composition reaches 7.564627 tok/s; its +0.42% versus the earlier MoE-only point is not an independently established contribution of the other four. No percentages are added. L3/L5 HC experiments, MTP/DFlash and cancelled P04/expert-encoder work were not retested.

## Behavior and defaults

The previous path waited for every final MoE command before encoding the next eligible L1 consumer. The new path carries its submitted producer commands and values in the existing bounded pending HC write, submits the consumer, joins/checks **all** producers and checks the consumer before publishing GDN state. Abandonment and encoding failures also join producers. Router readback, demanded SSD launch, reduction order, shaders and model snapshots keep their contracts. Only the34 existing L1 consumer boundaries are eligible; export/public/depth fallbacks stay synchronous.

| Setting | Exact-M5 default | Meaning |
|---|---|---|
| MLXL3_NATIVE_MOE_DEFERRED | ON | Set0 to restore the previous completion boundary |
| MLXL3_NATIVE_GDN_HALF_ELISION | OFF | Avoid private half→F32 materialization where HC can consume half |
| MLXL3_NATIVE_GDN_NORM_GATE | OFF | Fuse canonical F32 normalization and precise sigmoid gate |
| MLXL3_SSD_DYNAMIC_READERS | OFF in this native backend | One bounded common queue for the same whole-record reads |
| MLXL3_NATIVE_FIXED_RESIDENCY | OFF | Explicit residency for admitted immutable fixed weights only |

L1 remains ON on exact M5. The four earlier HC/workspace/extra-wave options remain OFF. Strict0/1 parsing and hardware/capability fallbacks remain in place. Residency enrolled1644 actual SDK allocations/3,272,810,112B, exactly the fixed-weight ledger; those are existing weights, not new payload capacity or process RSS. Its idle detach/resume path was tested. No installed app was replaced and no release/merge was performed.

## Fast speed screen

Apple M5/24GiB; Qwen3.8-Flash-Next EXL3 average2.05bpw, checkpoint65c89531; context4096/cache3GiB/48×53slots/four readers/depth0/prefixOFF/T0K1/penalty1. One warmup plus one measured response per process,32 generated/31 decode tokens. Twenty complete warm/measured outputs match the same full text/token hash bfa49090cb33d4fe. All owned processes/readers joined. AC100%; no recorded thermal/performance warning; temperature not measured; swap observed, never an abort condition.

| Arm | Decode tok/s | Scope |
|---|---:|---|
| published | 6.317694 | previous published b489; slower than both current controls |
| opening | 6.713478 | same exploration binary, all five OFF |
| moe | 7.532705 | MoE only |
| half | 6.681595 | private GDN cast elision only |
| normgate | 6.543291 | GDN norm/gate only |
| queue | 6.526605 | common native SSD queue only |
| residency | 6.687134 | fixed-weight residency only |
| closing | 6.650899 | same exploration binary, all five OFF |
| all-five | 7.564627 | same exploration binary, all five ON |
| actual final default | 7.394552 | new final binary, MoE variable **absent**, other four OFF |

The exploration OFF controls drift−0.93%. The older published control is retained and was not used to inflate the claimed improvement. Final-default TTFT is11.248914s, response time15.441205s and Metal allocation6,608,617,472B. These isolated points do not establish individual small gains, long-context behavior or sustained throughput. [Raw screens](speed/report.json), [composition](composition/report.json), [actual default](default-speed/report.json), [summary](summary.json).

Before retaining the four optional candidates, the adaptation was audited: actual36cast/36norm paths; unchanged F32/T17 rounding and public/F16/trace contracts; common queue1631 actual model submissions with unchanged whole-record admission; and real residency count/bytes/lifecycle. Norm fusion processes four gates per lane and the common queue adds a mutex: potential costs, not diagnosed bugs or attributed timings. No further parameter matrix was run.

## Essential verification

A single DIRECT/chat full suite passed **369 tests,61 ignored**, plus14 focused SSD tests, two manifest tests, strict Clippy, fmt, reviewed258-file dependency inventory and real release CLI/libtest builds. The ignored tests remain unexecuted except the named physical gates below. Both previous failure modes were detected by temporary mutations: AND→OR eligibility and static scheduling under the held-worker oracle; restored implementations pass. CPU logs include the initial large-enum Clippy rejection and its narrow stack-ownership rationale; no per-boundary Box allocation was introduced.

Four initial physical/CLI gates passed, then only two affected gates were rerun on the actual final default:
1. MoE/L1: four steps,115,212,288 state words/1,986,560 logits, independent84-state archive, nonzero34 boundaries, encode failure, injected deferred-producer failure before state publication, retained prefix and late rollback/recovery.
2. GDN composition: two sequential tokens×four half/norm modes;230,424,576 state words/3,973,120 logits/81,920 hidden words; finite independent archive/continuation, actual36 counters, public/F16/trace fallback,6144-value norm/gate corpus and T17's03f2 subnormal case. All five new options were enabled for this composed model check.
3. Existing SSD model gate extended with residency OFF/ON, exact SDK/ledger counts, full states/logits and idle/drop/resume replay.
4. Real4096 CLI short golden9f850af1cbba074b,48×53slots, no MLX library, clean shutdown.
5–6. The MoE model/error gate and real CLI again with the **MoE variable absent** in the final binary.

Kani0.68/CBMC6.11 bounded checks on actual CPU helpers passed:
| Property | Result/domain |
|---|---|
| MoE deferred-consumer eligibility |1 assertion,2 covers; two symbolic bools, no assumptions |
| Private GDN eligibility |2 assertions,4 covers; three symbolic bools, no assumptions |
| Native common-queue flag |232 checks,4 covers; UTF8≤4/unwind10;7 standard UTF8 guards unreachable |
| Residency admission guard |2 assertions,6 covers; symbolic bools/u64, no assumptions |
| Reused qualified-default parser |240 checks,7 covers; UTF8≤4/unwind10;7 standard UTF8 guards unreachable |

These do not prove Metal, IO/concurrency, Objective-C lifetime or complete generation. Real GPU Failed/OOM was not injected; the producer failure is a controlled actual completion-path error. Other chips, the MLX-feature build and installed app were not exercised. The parent's profiling-ancestry Kani ICE remains documented; no intrusive profile was repeated here. CrossHair is unavailable.

## Source and receipts

Exploration source635c8a1af721085d43e6c5f342949a3548d50bb8: CLI487d15a69ff2ec2b5b6233c4d46eb42e5b711dccb73c8b4f44305c7eb646b342, libc6facf3257e512f47939bf3ba81f606529eaa89013ab41194d7899a355a63ac2. Final source4771b20d052ad727594419da32c4f473896b2eb3: CLIdc8bb065514550d8cedb86009b426eb910399330525df27742330b23597a3a32, lib543db49ef2d0b40a15315e0244be534bf1ec2892860c1812bcbdf4924daceb23. All522 source fingerprints match; the sole executable-source delta is the MoE default initialization in gpu.rs. Actual final minos26.2/SDK27.2. [Exploration freeze](freeze.json), [selected final freeze](freeze-default.json), [conditions](conditions.json).

The two controller preflight mistakes were blocked before GPU measurement: composition output-path substitution, then a final-screen pointer to the old freeze. Both remain in opti.md; the refused final controller is archived. The successful exact default gates were reused after correcting the screen's freeze path; none was rerun unnecessarily. Reports never mark partial/failed attempts as validated.

CPU logs and physical logs are preserved losslessly as gzip with raw SHA/size in [log-archive.json](log-archive.json); controller snapshots are the exact used .py.txt files. The benchmark driver and validators are the existing qualified versions, not a new framework. Commands and bounds are in the CPU receipts and each phase report. Hosted CI must be inspected for the published PR head; no global CI-green claim is made here.

The original commands/imports use this Mac’s absolute worktree paths. [controller-manifest.json](controller-manifest.json) also preserves the imported legacy validator/process helpers; adapt paths to these byte-identical snapshots when reproducing elsewhere.
