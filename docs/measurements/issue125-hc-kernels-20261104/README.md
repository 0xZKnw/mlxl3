# Issue125 HC kernel trials L3/L4

Two independently selectable native HC variants, initially off. Based on65b50fa5, after qualified P13; the existing research and rejected HC22/HC24 experiments remain in `opti.md`. No GPU numerical qualification or speed measurement has run yet.

- `MLXL3_NATIVE_HC_UP_MIX=1`: F32/untraced exact M5 only,320groups×256threads. Keep the original K320/BN1 FMA and shuffle order per up output, convert to source float before ordinary sigmoid, then mix four streams in order with contraction off. Up is no longer a materialized buffer; mixed remains separate. Actual cfgtest counter `hc_up_mix_uses` is incremented at the selected encode, expected97per model row.
- `MLXL3_NATIVE_HC_DOWN_INJECT=1`: F32/untraced exact M5 only,81groups×256threads. Reuse the narrow GEMV source body, with group80 selecting the separate four-row injection matrix and separate zero-offset output. Never pass a324-wide view into legacy activation. Actual counter `hc_down_inject_uses`, expected96per row.

Flags accept exactly absent/`0`/`1`, are sampled once in `Metal::new`, default off, and only select kernels on exact `Apple M5`. Other hardware, half inputs, component traces and explicit zero use the existing implementation. New PSOs are requested only when selected. No weight conversion, SSD/cache/routing, public API, HC-write, model state, installed app or release change is made.

The structural removals are97dispatches and97up allocations per row for L3,96dispatches for L4. These are not measured latency or throughput gains. L4 does not reduce matrix bytes, multiply-adds or allocations. L3 removes7,946,240logical temporary write/read bytes per row; this is not measured physical bandwidth.

## Current verification

[CPU progress](cpu-progress.json) gives exact argv, environment and log identities. Initial full DIRECT/chat suite:354passed/55ignored. Strict all-target Clippy passed. The complete shader compiled with `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`; default xcrun first failed to find `metal`, retained as a negative setup result. Twenty warnings arise in existing other shaders.

Kani0.68/CBMC6.11: two real CPU helpers verified,301checks,12covers. Shapes are arbitrary u32 triples; selection guards arbitrary booleans; index bounds cover valid groups/streams/output/K. Flag literals cover UTF8 of length0..4, both hardware states and absence, unwind10 with assertions retained. Seven standard UTF8 guards are unreachable. This is bounded source verification of CPU contracts, not a proof of Metal arithmetic, allocation, OS or concurrent execution. A wrong319-group mutation fails the Kani assertion and unit test; production320 was restored and Kani rerun passed.

The final full-suite invocation accidentally overlapped that temporary mutation and correctly failed on319groups. Its failed log remains preserved. After the parent GPU window, the restored full suite passed354tests/55ignored. Both release builds passed (CLI35.96s, libtest61s), with minos26.2/sdk27.2 verified. Exact artifacts are in [freeze.json](freeze.json). Ruff passed for the updated proof-manifest regression; pytest skipped44Linux-only tests on macOS, and the two actual baseline/required-set checks were executed directly and passed. CrossHair is absent; no Python formal claim is made.

[Physical plan](physical-plan.json) lists exact ignored test names, fixed nonempty counts, environment, finite checks and deadlines. Tests execute one GPU process/model at a time after explicit coordination. Component48cases cover three HC instances, four variants, F32/half, strided inputs, trace fallback and a dropped encoded command followed by recovery. Model16cases cover four tokens×four variants, true97/96path counts, finite logits/hidden and full84-state continuation against the legacy implementation plus historical independent initial SHA oracles. The true untraced candidate runs before explicit legacy continuation; exporting model traces alone is not an activation oracle.

CPU qualification is complete and all owned compiler/prover/test processes are joined. Physical GPU qualification remains pending an exclusive grant. No push, speed claim, app replacement, merge or release.
