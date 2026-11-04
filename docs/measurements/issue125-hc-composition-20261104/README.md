# Issue125 HC kernels + workspace composition

Status: source preparation only. No formatting command, inventory checker, build, proof, test, GPU inference or timing has run for this composed source. The individual L3/L4 and L5 CPU receipts remain independent; their frozen binaries do not qualify this merge. L1/L2 are deliberately not included.

Parents are L3/L4 fdf99c6d91f0b5d04258ac5fa0b9fb9b03555162 and L5 source3d47c4f4d2e01b23182e13c439b7acd684c47fc5, with its separate artifact receipt6feb262d. The preregistration was committed asab104157 before cherry-picking. L5 source merged asf5285b84 after resolving four files (`gpu.rs`, both proof manifests, native inventory); its artifact receipt merged as32a06530. Both optimization histories and all six proof obligations are preserved. The native inventory is a reviewed union pending regeneration/check after a CPU grant.

The semantic adaptations are limited to:

- Determine L3 eligibility before acquiring a lease. `with_up` is false only for selected F32 untraced fusion. F16 retains the fourth buffer even when the L3 flag is1; component traces retain the original fresh buffers.
- Let the selected L4 joint GEMV write the leased down buffer. Raw injection, mixed and activated gate remain separate fresh outputs. There is no324-wide output or offset subview.
- Retain the named `_workspace` lease in `ReadBuffers::finish` through completion/readback. Cache Key/State, command ownership, non-Completed invalidation, busy/dtype fallback and idle release are unchanged from L5.
- Reserve97×43520B =4221440B for the actual F32 model when L3 is active, otherwise8194560B. The workspace remains disabled by default and is accounted only when selected.

All HC shaders and the workspace state/cache policy are byte-identical to their respective parents. No arithmetic, FMA/shuffle, typed load, stream order, SSD, routing, public output lifetime or model state changes are introduced by the merge.

Physical assertions are prepared for exact combined behavior. The workspace component checks optional owner counts3/4, matching poison/acquire mode, fresh requests6(F32/L3) or7(F16/legacy), reused3, true busy/dtype fallback and all dropped-command/late-error/recovery cases. Its marker reports `fresh_requests_by_dtype_f16_f32` (expected `[7,6]` with L3 enabled); the standalone marker's scalar warm count is not reused. The model kernel matrix also requires97actual workspace acquisitions per untraced row when selected, alongside97L3/96L4 encodes. Ledger and admission use the exact selected capacity, with complete state/logit and retained-snapshot checks preserved.

The future CPU batch must format, regenerate/check the inventory, run strict lint, relevant full suites and builds against a private cleaned target. Existing six Kani properties apply only to their identical CPU helpers and stated bounds, never to Metal or this composition's resource lifetime. The changed factory and admission wiring still require physical tests after freezing fresh artifacts. A wrong `with_up=true` or an L4 fresh down destination should fail the exact allocation assertions; these mutations have not been executed on the composition.

After CPU qualification and an exclusive GPU grant, use the existing bounded process owner with the prepared workspace component, memory ledger, HC component/model oracles, then a real CLI golden/ledger/error/cancel path on the final binary. Never infer composed throughput by adding standalone gains. No push, activation, app installation, merge or release is performed here.
