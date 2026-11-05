# Flash Next AR — twelve upstream-inspired candidates

Research and prototypes for [issue131](https://github.com/0xZKnw/mlxl3/issues/131), stacked on PR130 source8c13e922. These are twelve distinct mechanisms, not twelve promised large gains. Native AR only: no MTP/DFlash, excluded P04/expert-encoder experiment, or HC L3/L5 rerun. The original dirty workspace, model weights and installed app were untouched.

## Sources and registration

The [preregistered matrix](matrix.md) maps A01–A12 to source opportunities, history and oracles. Primary-source notes: [Strata](strata.md), [oMLX/MLX-LM](gdn-attention.md), [llama.cpp/TensorFold](kernels-banks.md). Official revisions were pinned before prototypes. Upstream affine/BF16/GGUF arithmetic was not imported as an EXL3 numerical oracle. [Adaptation audit](audit.md) records costs and limits before interpreting weak results.

## Initial short screens

M5/24GiB, Qwen3.8-Flash-Next EXL3 average2.05bpw/checkpoint65c89531, macOS27.2, Xcode/SDK27.2, deployment26.2. Context4096, cache3GiB/48×53slots, four readers, depth0, prefixOFF, temperature0/top-k1/repetition1. One warmup and one measured MAIN32 response per arm:32 generated/31 decode tokens,87 prompt tokens. L1+MoE deferred remain ON, the earlier eight optional paths remain OFF. No swap abort. [Conditions](conditions.json) include AC power and desktop/background activity; this was not an isolated machine.

| Arm | Decode tok/s |
|---|---:|
| Published PR130, fresh |7.401531|
| Same-binary OFF start |7.589933|
| A02 routed NT2 |7.413658|
| A03–A06 experts |7.492045|
| A07+A10 GDN |7.481796|
| A01+A08+A09 HC/QSA |7.570562|
| A11+A12 IO |7.915628|
| All twelve |**8.399583**|
| Same-binary OFF end |7.684634|

All eighteen complete warm/measured outputs match the fixed MAIN32 text and token hash; all children joined. The composed arm is +9.30/+10.67% versus same-binary controls, with +1.25% endpoint drift. The fresh published point is retained separately, not substituted for the current control. Group screens do not establish individual contributions and percentages must not be added. NT2 component wall-time samples are noisy and are not per-kernel GPU times. Raw reports: [components](components-01/report.json), [model/CLI quality](quality/report.json), [initial speed](speed/report.json), [initial freeze](freeze.json).

## Correction and integration status

Every candidate passed its physical component oracle. A01–A10 also pass a shared full-model OFF/ON gate and a legacy continuation with all new arithmetic options OFF:57,606,144 finite state words,993,280 logits,81,920 hidden bytes, plus the independent84-state initial archive. Actual per-row uses are HC82; routed NT2 GU960/down480; expert paths480/480/48/480; whole GDN36; QSA prepare12/gate12; joint a/b36. The expert-axis path issues144 QMV dispatches versus1,440; this dispatch reduction is not a throughput forecast. A04's distinct synthetic Suh swap is detected and restored. Retained frames, reordered/duplicate routes, negative strides, F16 fallback and uncommitted error recovery are covered by the components.

A11/A12 component oracles verify explicit reuse and full-record split bytes, EOF/deadline/drop/partial-enqueue recovery and no publication after only one part. The model/CLI gate runs with both IO options ON but does not instrument their real model activity; it is not an IO OFF/ON numerical comparison. The17-byte reuse fixture is distinct from the1,248,012-byte split fixture. Positive partial reads/EINTR rely on std read_exact_at and were not separately fault-injected. The IO group result cannot be attributed to either member alone.

After the initial matrix, the audit found two unnecessary OFF-path costs: six expert PSOs built unconditionally and a host Vec of absent record owners per Shared batch. The final source loads those PSOs only when initially enabled and uses an exact-size iterator for fresh destinations. No shader/math/shape/cache policy changed. Final source35c7c976 and the [final freeze](freeze-final.json) are separate from the initial component binary; only affected OFF/ON model/CLI and short speed checks are repeated. All twelve candidate options remain **opt-in/OFF by default**, including on M5. L1+MoE remain the accepted defaults. There is no installed-app update, merge or release.

| ID | Candidate | Correction | Individual performance / integration |
|---|---|---|---|
| A01 | HC write→norm |Validated component + model|Promising retained, group only; opt-in|
| A02 | Routed fixed-K2 NT2 |Validated stock + model|Inconclusive/no isolated boost established; opt-in|
| A03 | Up finish→Half SwiGLU |Validated stock + model|Promising retained, expert group; opt-in|
| A04 | Distinct dual GU preparation |Validated prefixes/swap + model|Promising retained, expert group; opt-in|
| A05 | Canonical route output slab |Validated prefixes/mix + model|Promising retained, expert group; opt-in|
| A06 | Ready-cohort EXL3 expert axis |Validated stock/ownership + model|Promising retained, expert group; opt-in|
| A07 | Whole GDN CTA |Validated every state/history word + model|Promising retained, GDN group; QK duplication/CTA costs; opt-in|
| A08 | QSA norm→RoPE |Validated positions/strides + model|Promising retained, HC/QSA group; opt-in|
| A09 | Canonical attention gate store |Validated short contexts + model|Promising retained, HC/QSA group; opt-in|
| A10 | Joint a/b GEMV |Validated independent a/b + model|Promising retained, GDN group; opt-in|
| A11 | Quiescent Shared record reuse |Validated byte/error component; model enabled|IO group favorable short screen, activity unattributed; opt-in|
| A12 | Two reads for one cold miss |Validated exact physical split/error component; model enabled|IO group favorable short screen, activity unattributed; opt-in|

## Verification

- `cargo fmt --check`, `cargo clippy --locked --all-targets --features chat -- -D warnings`, `cargo test --locked --features chat`, release CLI and libtest builds; final full suite **379 passed/73 ignored**. Physical tests are selected explicitly and run alone, not inferred from ignored tests.
- `xcrun -sdk macosx metal -c native/shaders/exl3.metal`: success,37 warnings retained in the log, including canonical include warnings. No shader source changed after this check.
- Eleven real-source CPU Kani0.68/CBMC6.11 harnesses: **734 checks/44 covers**, no failed properties. [Seven guard commands/log](proofs/report.json), [four expert proof receipts](experts/proofs/README.md). Symbolic booleans and full integer types cover exact shape/format admission, prefix arithmetic, record ranges and unique claims; parser UTF8 length≤4/unwind10 and array comparison unwind40 are explicit in the receipts. No added proof assumptions/stubs. Thirteen expert standard-library unreachable checks are explained; no business assertion/cover was unreachable there.
- The global Kani inventory attempt hits the already observed compiler ICE at intrinsics.rs243/expectedI32; it remains **unverified**, not a passing full proof suite. Two pure manifest checks pass;44 pytest orchestration cases skip because Linux prlimit is absent. Hosted CI must be inspected independently.
- CrossHair is absent (`No module named crosshair`), so Python source verification remains unexecuted. Controllers reuse the established strict JSON/process cleanup/golden validators; plan selfchecks reject six malformed plans. Failed compiler attempts and the refused inventory remain in the evidence.

The bounded CPU checks do not prove Metal arithmetic, GPU/CPU concurrency, ObjC lifetime, IO correctness, the allocator or complete generation. Physical numerical tests are sampled evidence. No HOLD/long-context/sustained throughput, other GPU, MLX-feature FFI build, installed app or release qualification is claimed. Final validation results and exact publication head are appended after the final short screen.

## Final retained source check

Final source `35c7c976743b4840b0213059d2142dbf46ff4ae5`, all531 recorded source hashes checked against the retained files. The CLI embeds `83e118ccf74a-dirty` because the final idle fix was built before its commit; the complete source hashes and binary SHA identify the measured implementation. Final CLI SHA-256 `a99c9cb432f7b79d93b4e9da095fd516d6cc722d3cce494dc32a9cf7546421db`; libtest `44d543faabb0431352b7fa52a09b4eff92b074b88900d9e3eec438dfe05cf4c8`.

The affected final idle/model/CLI gates pass3/3, including six expert PSOs absent when OFF, every selected ON counter and complete state/logit parity. [Final quality](quality-final/report.json), [final MAIN32](speed-final/report.json):

| Final arm | Decode tok/s |
|---|---:|
| OFF start |7.685043|
| All twelve opt-ins |**8.349829**|
| OFF end |7.591638|

The final composition is **+8.65/+9.99%** versus its endpoint controls (endpoint drift−1.22%). Six further complete warm/measured outputs match the same text/hash, for24/24 speed outputs overall; all processes/readers joined. Reported native Metal allocation is6,608,617,472 bytes for each final arm; this is not process RAM, peak transient allocation or a per-candidate memory saving. All12 trials are closed: correction validated, individual gain inconclusive/group only, compatible composition favorable short screen, code retained opt-in and no new default activated. The idle fix is in the retained source and its affected paths were rechecked; the earlier component evidence remains explicitly tied to its earlier binary.

To reproduce the all-candidate composition, set every flag listed in `plan-final.json` arm `all` to1. The default engine does not enable these experiments. Existing L1/MoE defaults remain ON; no MTP/DFlash is used. Controllers and plans are preserved as `.py.txt` snapshots, including the initial and final versions. Raw logs are losslessly compressed with raw/compressed SHA-256 in [log-archive.json](log-archive.json); independent routed fixtures are copied byte-identically with their provenance manifest. No temporary proof log is missing.
