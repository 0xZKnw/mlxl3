# Issue125 HC kernels + workspace composition

Status: source preparation only. No formatting command, inventory checker, build, proof, test, GPU inference or timing has run for this composed source. The individual L3/L4 and L5 CPU receipts remain independent; their frozen binaries do not qualify this merge. Published L1/PR126 is now included with its qualified exact-M5 default ON; L2 remains excluded.

Parents are L3/L4 fdf99c6d91f0b5d04258ac5fa0b9fb9b03555162 and L5 source3d47c4f4d2e01b23182e13c439b7acd684c47fc5, with its separate artifact receipt6feb262d. The preregistration was committed asab104157 before cherry-picking. L5 source merged asf5285b84 after resolving four files (`gpu.rs`, both proof manifests, native inventory); its artifact receipt merged as32a06530. Both optimization histories and all six original proof obligations are preserved. The later L1 extension was separately preregistered ase5d187a4, after importing L5 OFF/ON QA as61aa2afe; L1 runtime merged asd74adb33 and release receipt as581f4270, adding its seventh required guard proof. The native inventory is a reviewed union pending regeneration/check after a CPU grant.

The semantic adaptations are limited to:

- Determine L3 eligibility before acquiring a lease. `with_up` is false only for selected F32 untraced fusion. F16 retains the fourth buffer even when the L3 flag is1; component traces retain the original fresh buffers.
- Let the selected L4 joint GEMV write the leased down buffer. Raw injection, mixed and activated gate remain separate fresh outputs. There is no324-wide output or offset subview.
- Retain the named `_workspace` lease in `ReadBuffers::finish` through completion/readback. Cache Key/State, command ownership, non-Completed invalidation, busy/dtype fallback and idle release are unchanged from L5.
- Reserve97×43520B =4221440B for the actual F32 model when L3 is active, otherwise8194560B. The workspace remains disabled by default and is accounted only when selected.

All HC shaders and the workspace state/cache policy are byte-identical to their respective parents. No arithmetic, FMA/shuffle, typed load, stream order, SSD, routing, public output lifetime or model state changes are introduced by the merge.

Physical assertions are prepared for exact combined behavior. The workspace component checks optional owner counts3/4, matching poison/acquire mode, fresh requests6(F32/L3) or7(F16/legacy), reused3, true busy/dtype fallback and all dropped-command/late-error/recovery cases. Its marker reports `fresh_requests_by_dtype_f16_f32` (expected `[7,6]` with L3 enabled); the standalone marker's scalar warm count is not reused. The model kernel matrix forces workspaceOFF for the reference and legacy continuation, restores the initial workspace policy for each candidate, and requires97actual acquisitions per selected untraced row alongside97L3/96L4 encodes. Ledger and admission use the exact selected capacity, with complete state/logit and retained-snapshot checks preserved.

The future CPU batch must format, regenerate/check the inventory, run strict lint, relevant full suites and builds against a private cleaned target. Existing seven Kani properties apply only to their unchanged CPU helpers and stated bounds, never to Metal or this composition's resource lifetime. The changed factory and admission wiring still require physical tests after freezing fresh artifacts. A wrong `with_up=true` or an L4 fresh down destination should fail the exact allocation assertions; these mutations have not been executed on the composition.

After CPU qualification and an exclusive GPU grant, use the existing bounded process owner with the prepared workspace component, memory ledger, HC component/model oracles, then a real CLI golden/ledger/error/cancel path on the final binary. Never infer composed throughput by adding standalone gains. No push, activation, app installation, merge or release is performed here.

## Published L1 extension, still no composed validation

[PR126](https://github.com/0xZKnw/mlxl3/pull/126) is included exactly at runtimef99b0576 plus artifact receiptc5286b90. Its independently measured P13→L1 AR32 result is MAIN+5.897%/HOLD+2.955%, with the full protocol, drift,16matching warm/measured outputs and limits in [the inherited L1 report](../issue125-cross-layer-20261104/README.md). These values are not a composed result and are never added to other trials.

The merge preserves the shared `Metal::encode_hc_write`, private pending ownership,34triples/2doubles/14finalwrites, newly written streams passed to the following residual, whole-token rollback and synchronous export/lookahead/non-GDN fallbacks. L5 still drains and clears both per-layer HC caches and the mixer at idle. `ReadBuffers::finish` retains the lease until the already completed enclosing command, including the L1 triple. Experimental L3/L4/L5 flags remain OFF by default. No L2 source, no new artifact, no compiler/prover/formatter/check/test/GPU execution and no push has occurred in this composition.

L5's separately added true workspace OFF/ON4-token model fixture is imported, with97uses per selected row, complete independent initial84-state SHA oracle, logits/hidden/full-state continuation and retained-prefix checks. It remains source-only/uncompiled just like the composition. The L5 owner independently reviewed the original L3/L4+L5 source and found no concrete ownership/key/output/admission defect; the later L1 merge still requires its own qualification.
