# AR33 — exact codeword table, cached capabilities and remaining source gates

2026-11-05. [Issue138](https://github.com/0xZKnw/mlxl3/issues/138) and `opti.md` preregistered the [matrix](matrix.md) before implementation. This is a qualified **increment**, not completion of every proposed runtime port. No Instruments profiling was resumed; Brave and Deezer stayed open.

**T01 and T02 pass numerical qualification and remain opt-in OFF. Their performance is nonconclusive.** MAIN32 showed a possible T02 signal; MAIN128 did not confirm it and controls drifted9.56%. No default, installed app, sustained speed or11–12tok/s achievement is claimed. The final old-binary MAIN128 confirmation is still running at this documentation checkpoint; its receipt must be reviewed separately before publication closes.

## Retained code and exact artifacts

Base `01da8f013c4eeb1a8b1fdeb2ec0fab0e347620d4`, qualified native baseline `82fff5a54c91efd2a3f3d34f75a83aa5251785cc`, plus the recorded AR33 source changes. [Freeze](freeze.json) pins279source/helper/model-metadata/fixture fields and the executed artifacts:

| Artifact | SHA-256 | Bytes |
| --- | --- | ---: |
| New direct CLI | `d00476ff609137914164de5ddf9ed660dd98aa8eaee1a88e7a48fca4fd9ee4de` |10,364,416|
| New native libtest | `cfc3e52c543e4bdf412b393f4ead71733bf2e7957bdbe67d1bfdde7b736dfb68` |10,339,472|
| Old qualified direct CLI | `f24201e3e5ca145f94397a90cf9edaa746d81c86102c67d2d51350a7d3e734f0` |10,346,944|

- **T01** `MLXL3_NATIVE_CODEWORD_TABLE`: exact1021-entry/2042-byte Half table replaces only the final Half FMA after the unchanged cb2 hash and four-byte sum. The static bits come from exact rational values rounded once to nearest-even. Codebooks0/1,16-bit state masking, EXL3 interpretation, Half boundaries and canonical downstream FMA/reduction stay unchanged. This is historical K7; it does not repeat PERF89's invalid8-entry K3 table. Strict qualified Apple M5 selection, defaultOFF and explicit0 override remain.
- **T02** `MLXL3_NATIVE_CACHED_CAPS`: each wrapped PSO owns its immutable width/maximum snapshot. Existing dynamic guards, order, error paths and underlying PSO remain at their callers. The narrow domain is HC/GDN/QSA/PLE/model-ops plus shared MoE switch/logsumexp, not every projection or Metal property. No global registry, address cache, shared encoder or numerical change. [Design, scope and proof](capabilities-design.md).

The thirteen accepted AR10 settings remain ON. All twelve AR20 and four AR30 experiments remain OFF; no MTP/DFlash, cancelled P04 or HC L3/L5 option is enabled. The two AR33 settings remain OFF by default.

## Independent numerical qualification

[Quality report](quality/report.json): **5/5phases pass, all children joined** — four explicitly selected physical tests followed by the real two-token CLI journey.

| Gate | Nonvacuous evidence |
| --- | --- |
| Actual GPU codeword oracle |592,137finite state words across all65,536states, three codebooks, three high-bit masks and truncated dispatch tails; all1021table entries vs canonical GPU Half FMA; six shifted-index mutations detected. |
| Resident projections |48frozen stock cases,738,304finite Half words, full binary comparison, zero unequal words;44actual table-path resident dispatches. |
| Routed projections |20,480finite Half words vs independent frozen stock;24actual table-path routed dispatches. |
| Full model/continuation | Independent archived initial84states and cross-process oracle:57,606,144state words,993,280logits,81,920hidden bytes; exact equality and finitude, two forced token cases and restored continuation. Actual resident/routed table totals1,565/1,701. |
| Production CLI | Two-token greedy golden, context4096, no MLX library loaded, actual bridge shutdown and joined child. |

T02's wrapper-domain counters are observed on the model path: each measured forced row adds621cached width accesses and549cached maximum accesses, with **zero new live queries in that wrapper domain** after load. The actual selected flags and counter deltas are in the quality report; this does not count unwrapped PSOs as cached or turn query counts into a speedup.

The model oracle is the independently captured all-new-OFF archive, SHA `171bea949c7c16f20c1e1affa26b70602676c73620fcbb380d430a7587835351`. Its full-state gates cover the specified forced cases, not every state at every benchmark token. Benchmark output identity is checked separately below.

## MAIN32 and MAIN128

Same Apple M5/24GiB, Qwen3.8Flash Next EXL3avg2.05 checkpoint `65c89531`; context4096, expert cache3GiB (48×53slots), four readers, dynamic readers0, lookahead0, prefix cache0, temperature0/top-k1/repetition penalty1. Each process performs one warm generation and one measured generation. All five arms use the same new frozen CLI and differ only in T01/T02 selection. Exactly32generated/31decode or128generated/127decode tokens; complete text and token hash match the independent golden in both warm/measured outputs.

| Arm, fixed order | MAIN32 tok/s | MAIN128 tok/s |
| --- | ---: | ---: |
| OFF start |8.151001|8.674594|
| T01 table |8.274966|8.553075|
| T02 capabilities |8.688592|8.322255|
| T01+T02 |8.345306|8.082439|
| OFF end |8.370118|7.845044|

[MAIN32](speed32/report.json):5/5arms,10/10warm/measured golden outputs, all joined. T02 is above both short controls; table and composition lie between them. This isolated observation is not a promoted default or an established component gain.

[MAIN128](speed128/report.json):5/5arms,10/10warm/measured golden outputs, all joined. OFF falls8.674594→7.845044tok/s (−9.56%); every candidate is between the controls. **Performance nonconclusive** for the two components and their combination. Do not add the32-token deltas or attribute the decline to heat, Brave or Deezer.

The [independent128reference](reference128/receipt.json), made with old qualified CLI `f242…`, observed about9.207tok/s **before** these groups. It supplies a full128-token golden, not a current causal speed control. The final [old-OFF128 confirmation](confirm-old128/receipt.json) is8.807120846tok/s, after new-OFF7.845044375. This short sequence does not separate OFF-path overhead from changing conditions; no blanket absence-of-regression or gain is claimed. Both new flags remain experimental OFF, with default integration unqualified.

## Other trials and verification limits

T03's smallest stable dispatch-description port would remove no demonstrated Metal calls or allocations: stack bindings, inline parameters and encoder work still occur. Its source gate closes without a runtime prototype, not as a universal rejection of replay. T04's same-budget historical route replay reconstructs609,120physical(slot,hit) observations before comparison; cyc64 increases decode misses983/1614on MAIN/HOLD (+1.79%/+2.74%). This is a negative **offline opportunity** result, not numerical GPU parity or tok/s. [Source gate/provenance](cycle-opportunity.md), [final replay](cycle-replay-final.json).

T05 remains **not implemented** and continues in the next issue phase: native53separate buffers need GPU-addressable argument-buffer bindings plus retained read leases; the synchronous route producer also needs an owned pending boundary. The narrow candidate can compute partial resident contributions before CPU route consumption, then repair cold contributions and retain the complete canonical sum; no successor rollback is required for that first intralayer variant. Source feasibility is not compile/quality/speed evidence. T07's larger asynchronous AR pipeline likewise remains unimplemented. T06 PLE prefetch has no warm MAIN32 read opportunity because reset preserves the populated n-gram cache; only cold/new prompts remain a scope reserve. T08 durable prefixes concern TTFT, not prefix0 decode.

Root-reviewed CPU/build results: **403CPU tests passed,93ignored**, three strict Clippy modes, format and final release builds pass. Ignored physical tests are not represented as passing; the four selected physical gates above were run separately. [Commands/process receipts](cpu/checks.json), [final portable checks](cpu/final-portable.json). Final portable checks cover10controller tests,8offline cache-policy tests and10portable checks, with the affected inventory rerun6tests plus2subtests. Offline Metal compilation checks tableOFF and tableON; compiler warnings remain visible.

[Kani0.68.0/CBMC6.11.0](cpu/kani-caps-flags.json): two actual CPU harnesses pass in20.26s,346reported obligations =339SUCCESS+7std-internalUNREACHABLE,11/11covers and no failures. Capability inputs range over full64-bit values without assumptions; flag inputs cover presence/hardware and valid UTF-8 byte lengths0..4, unwind10. These bounded CPU checks do not prove Metal, FFI immutability, GPU indexing, concurrency or the complete model. CrossHair was attempted but unavailable; no symbolic Python proof is claimed.

Initial missing-module E0433, inventory mismatch, wrong Metal-tool path and controller closure lint failures were fixed; their original diagnostics are preserved, not replaced by the passing reruns. Raw evidence still awaits root's post-GPU publication archive. Numerical quality, exploratory performance, default selection and publication are separate statuses. No installed app or release is changed by these results.
