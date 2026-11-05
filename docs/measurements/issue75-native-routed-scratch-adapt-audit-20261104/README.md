# Scratch WAR audit after the combined pilot

**New dependency edges are present; a lost overlap cost has not been measured.** There is no basis here to assign the flat decode result to scratch, or to reject the three changes. Read-only audit, preregistered as `I75-AR-ROUTED-SCRATCH-ADAPT-AUDIT-20261104` before investigation. No executable change, prototype, build, test, proof or GPU process.

The ABBA pilot compares CLI `789aa28d…` with qualified main `942dbeee…`, Flash Next EXL3,3GiB,32tokens, one measured completion per prompt/process. Arithmetic means of the two process rates: MAIN5.877078→5.861515tok/s(-0.2648%), HOLD5.949796→5.936114(-0.2300%). MAIN candidates5.668175/6.054855 and HOLD6.051717/5.820511 vary substantially; reference drift is-2.14/-1.35%. TTFT means improve7.359/4.019%, total means5.373/2.863%; decode gain is not established. These are combined results, not additive isolated gains. The root physical component report has16 nonempty parity cases, actual allocations10OFF/6ON, poison/recovery and three duplicate/multiple encodes. Exact metrics, source/artifact identities and input hashes are in `audit.json`.

## What the native source guarantees

Composition `native/src/gpu.rs:662` creates a fresh default compute encoder, binds one kernel and ends it for every dispatch. The metal-rs0.33 binding invokes Objective-C `computeCommandEncoder`; it does not request concurrent dispatch. Apple documents [default serial dispatch within the compute pass](https://developer.apple.com/documentation/metal/mtlcommandbuffer/makecomputecommandencoder()). Each native pass contains only one dispatch, so this fact alone cannot establish that independent commands in different passes had zero overlap. No trace from the measured build establishes gate/up overlap or its absence. Apple’s [GPU performance guidance](https://developer.apple.com/documentation/xcode/optimizing-gpu-performance) also distinguishes an observed concurrent timeline from an artificial serial profiling replay; isolated pass timings cannot answer that question.

The native buffers are device allocations bound directly to legacy `MTLCommandEncoder` APIs, with default tracked hazards. The [Metal resource synchronization rules](https://developer.apple.com/documentation/metal/resource-synchronization) order conflicting access across passes; their separate Metal4 no-tracking rule does not apply to this implementation. These API guarantees establish safe ordering, not the duration of that ordering or a promise of concurrent compute passes on this device.

`FlashExpert::encode` issues gate prepare/QMV/finish, up prepare/QMV/finish, SwiGLU, then down prepare/QMV/finish. In the original path gate and up have independent scratch and destinations, although their input x is shared read-only. Their SUH factors differ, so prepared values cannot be shared.

| Shared temporary | New edge between otherwise independent gate/up work |
| --- | --- |
| Prepared P | Gate QMV reads P before up prepare overwrites P |
| Partial T | Gate finish reads T before up QMV overwrites T |

Thus reuse can force up preparation to wait for gate QMV, and up QMV to wait for gate finish. This is a concrete possible loss of scheduling freedom, not observed lost execution time. Down already depends on SwiGLU, which depends on both finished gate/up destinations. The new down scratch reuse is therefore transitively ordered by real dependencies. Inter-expert/Hot/W1/W2 sharing was not introduced.

## Legacy comparison and one bounded adaptation

The pinned [MLX0.32.2 command encoder](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp#L539-L546) keeps a concurrent encoder and [inserts barriers from recorded input/output conflicts](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp#L367-L394). Local `prepared_metal.h` and `flash_prepare.h` use that encoder and retain arrays. They do not turn the native one-dispatch encoders into the same scheduling policy. T01B/R1–R5 retained metadata/pipelines, not these live scratches; POOL04 altered persistent expert banks/cache policy, and54Q altered layout/decode. None is evidence for the cost of these two new edges. The earlier audit’s “principal risk” wording is refined here: it remains an unmeasured hypothesis.

**Conditional variant only: one pair for gate and one pair for up/down.** Gate keeps prepared5120B+partial2560B; up/down use prepared5120B+partial10240B. Keep the four destinations,10dispatches, shaders, rounding and all offsets unchanged. This predicts10→8 allocations and35840→32000 requested bytes/expert. Gate/up no longer share temporary resources; down can reuse up’s pair after SwiGLU because up QMV/finish are already completed before that dependency. It is local to one expert, with no persistent pool or arena. This variant has not been implemented or qualified.

Before allocating a new prototype, an explicitly preregistered comparison of the existing frozen789a binary with scratch0/1 and the other two flags held fixed could isolate whether scratch has any repeatable effect in this context. A timeline showing baseline gate/up overlap could then support testing the two-pair variant. If the GPU already serializes those passes or the decode critical path is elsewhere, restoring independent scratch can merely increase allocation cost. No new trial is authorized or launched by this note.

A second attribution limit is visible in the current source: every reused projection performs two `device()` and four `registry_id()` Objective-C calls, plus two buffer-length queries. That repeated validation has no isolated timing either. Allocation count10→6 therefore does not prove CPU preparation time fell proportionally, and the pilot does not isolate WAR from host/SSD/GPU costs. This audit selects no additional code change around those checks.
