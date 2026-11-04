# Flash Next AR above accepted L1: qualified opt-in candidates

**Accepted L1 remains the only enabled new default. All four additional candidates are opt-in and default OFF.** Their numerical paths are qualified, but the final throughput confirmation was flat. The original bounded SSD wave uses remaining≥3/gather≥2 with at most one extra CB. The uncompiled SSD5/3 idea is excluded.

| Option | Exact-M5 default | Rollback / opt-in |
|---|---|---|
| Accepted cross-layer L1 | ON | `MLXL3_NATIVE_CROSS_LAYER=0` |
| HC workspace | OFF | `MLXL3_NATIVE_HC_WORKSPACE=1` |
| HC up/mix fusion | OFF | `MLXL3_NATIVE_HC_UP_MIX=1` |
| SSD extra wave | OFF | `MLXL3_NATIVE_SSD_EXTRA_WAVE=1` |
| HC down/injection fusion | OFF | `MLXL3_NATIVE_HC_DOWN_INJECT=1` |

Flags accept only literal0/1; hardware qualification is preserved even for explicit1. F16/component trace retains the canonical kernels. No MTP/DFlash inference, installed-app replacement or release was performed.

When both options are enabled, the workspace retains only normalized/down/active; mixed/gate outputs remain fresh. Leases end only after the enclosing Metal command completes. Busy/dtype/trace and uncommitted/error paths retain safe fallbacks. Admission and the memory ledger include retained buffers. Compared with L1, this opt-in composition removes97up materializations/dispatches and reuses the remaining291scratch buffers per row after warmup:388scratch allocation requests removed in total, not388+97. Logical retained capacity is4,221,440B across97HC instances. The observed current-Metal increase in the exploratory composed screen was5,029,888B; this is distinct from logical capacity and RSS.

## Measurements use L1 as the accepted baseline

Earlier isolated P13 screens remain historical. After the user's correction, every current screen compares accepted L1 CLI `9e2c538c` with the common engine plus the selected option. Old percentages are not added. The common engine also includes inactive experimental branches; these are end-to-end comparisons, not a pure same-binary ablation of a single kernel.

Apple M5/24GiB, Qwen3.8-Flash-Next checkpoint65c89531, EXL3 average2.05bpw, context4096, cache3GiB/48×53slots, four readers, dynamic0/lookahead0/prefixoff, temperature0/topk1/penalty1. Following the request for faster feedback: MAIN32,31decode, one warmup and one measured response per process. One sample is exploratory; no HOLD, sustained or long-context gain is established by these screens.

| Current option | Decode tok/s | Reference and limit |
|---|---:|---|
| L1 opening control |6.561890|Accepted frozen CLI9e2 |
| L1 + SSD wave |6.524039|−0.577% vs opening control; no gain established |
| L1 + workspace |6.696875|+2.057% vs opening control, dated previous control |
| L1 + up/mix |6.653878|+1.402% vs opening control |
| L1 + down/injection |6.524045|−0.577% vs opening control; no gain established |
| L1 closing control |6.614966|+0.809% control drift |
| L1 + workspace + up/mix |6.760968|+2.207% vs closing control; measured composition, not summed gains |

All warmup/measured complete text and token hashes match,32generated/31decode, and owned process groups/readers are joined. The selected combination also reduced complete-response latency1.698% and TTFT1.533% versus the closing control on this short sample. Individual source/protocol/quality receipts are in [L2](fast-main32/), [workspace](fast-workspace-main32/), [kernels](fast-hc-main32/) and [composition](fast-w-up-main32/).

The final A/B MAIN32 comparison of the actual default binary versus accepted L1 is **6.630084 → 6.632714 tok/s (+0.03966%)**: effectively flat, with no throughput gain established. TTFT13.070135→12.968365s (−0.77864%); complete response17.745813→17.642189s (−0.58393%). Four complete warm/measured outputs match. [Final confirmation](final-default-main32/summary.json) preserves this result independently from the earlier favorable exploratory points. This default-ON trial is superseded: all four new options are kept OFF in the published engine. Numerical parity and structural scratch-allocation/dispatch reduction do not establish a speed gain. The two earlier standalone workspace campaigns, including the explicit Deezer-closed retest and its outlier, remain in the parent workspace report. Deezer observations do not establish a causal explanation for the old variance.

## Selected source and superseded trial binaries

The **superseded default-ON trial** [freeze](freeze-defaults.json): CLI `ae3730adedcceca5d03e7bda7e41b04a199dfa4139e0e82cc63162535b730402`, libtest `a69d5f53f0f2ea76b5d7378321e1312b29ad3cccef4d2537fec40618a27bf066`, actual minos26.2/SDK27.2. The measured opt-in prototype CLI `b4899081` and libtest `7382bddb`, plus earlier freezes, remain preserved. [Pre-default freeze](freeze.json), [intermediate freeze](freeze-pre-profile.json).

For publication the default-only edit was reverted to commit3a584944, selecting the already frozen common-engine CLI `b4899081478369ae575dff4863d65570efb9909b384fdbbf6330a93a2e660110` and libtest `7382bddbfb23538485bdb654e63319ab1513494189e67fd112e18f1e50df1296`. No numerical/kernel changes were made for this rollback. Selected executable source identities are checked against freeze.json before publication. The original personal checkout/model weights are untouched; targets are private APFS clones, package-cleaned before use.

## Enabled-path qualification and superseded default trial

Eight gates passed for the superseded W/UP default-ON trial, with those variables absent from the environment (DOWN1 is used only to load both PSOs for the kernel matrix):

1. HC model16cases: finite bit-exact full logits/hidden/84states,97UP/96DOWN when selected,97workspace uses, actual34L1triples in each untraced control/candidate,0triples in exported continuation.
2. Workspace model4steps: actual97reuses and34L1triples per row, full finite logits/hidden/84states and retained prefix.
3. Memory ledger/admission/idle cleanup with three-buffer workspace4,221,440B.
4. Real CLI: four goldens, rejection, cancellation/recovery and native/noMLX evidence.
5. Frozen independent bridge-budget fixture explicitly bound to the **new final CLI**.
6. AR prefix replay: two goldens and one real prefix hit.
7. Native8 diagnostic:679workspace uses,679up/mix uses,0down/injection uses,0warm workspace allocations,238L1triples+14P13doubles; ancestry rejects MoE/residual under a triple.
8. Original L2 model with the new W/UP defaults: cold OFF/ON extra waves, actual L1 work, late layer20 failure, prefix retention and complete recovery.

The first final runner attempt was interrupted by an incorrectly written parser selfcheck, not a numerical engine failure. Three successful exact final gates were revalidated offline and reused; only remaining gates ran again in a separate02directory. The selfcheck and preflight path errors, interruption and process-cleanup diagnostics remain archived. All owned groups were verified empty before resuming. No partial report is treated as full validation. [Initial receipt](final-default-quality/), [resumed receipt](final-default-quality-02/).

## CPU, verification and remaining limits

Full suites: CPU347pass/8ignored; DIRECT363/60; MLX399/278 before the diagnostic-only/default-policy edits. Relevant follow-up tests, three strict Clippy variants, fmt, normal reviewed inventory256, manifest/partition checks, actual final release builds and default-flag mutation checks pass. Zero-test filters and Linux-only skips are explicitly retained in the CPU receipts.

Actual source-level Kani receipts cover CPU eligibility, grid/index bounds, flags, lease state and admission arithmetic. Unchanged proof bodies are linked to their checked parents in [proof-reuse.json](proof-reuse.json). The changed workspace default proof passes240checks and5covers over symbolic hardware bools and UTF8≤4/unwind10, without assumptions/stubs; seven standard UTF8 guards are unreachable. The default-OFF mutation fails and restored source passes. [Defaults CPU receipts](defaults-cpu/).

The new actual two-event profiling ancestry harness, symbolic bools/Option<usize> parents/unwind64, was attempted and hit Kani0.68 compiler ICE at intrinsics.rs243 before obligations. It is **not formally verified**, and its explicit test-only core CI invocation is retained without suppressing failures. CPU regression and physical native8 ancestry checks pass. This is not a proof of Metal, IO concurrency, allocation or complete generation. Real GPU Failed/OOM was not injected; Python CrossHair is unavailable.

Exact-head hosted CI must be inspected after publication. Existing parent failures/timeouts and unexecuted GPU tests remain separate from successful named physical gates. Source research and old conclusions are preserved in the [initial CPU report](history/pre-default-cpu-report.md), `opti.md`, PR127 and parent measurement directories. No GPU speed is inferred from intrusive profile gaps, allocator API time or saved bytes.

## Publication decision

L1 remains enabled. Workspace, up/mix, down/injection and the original SSD wave remain opt-in on exact M5. The rejected default-selection trial and all favorable, flat, outlier and administrative-failure records remain archived. The selected default-OFF artifact passes its final native8/CLI check:238triples+14doubles, correct ancestry,2716workspace allocations, full finite independent oracles; real CLI golden/noMLX/48×53slots with no workspace residency. [Selected-artifact receipt](selected-off-quality/report.json). All owned processes are joined. No throughput boost is delivered or claimed by these optional paths. The profiling-scope/counting repair is retained.
