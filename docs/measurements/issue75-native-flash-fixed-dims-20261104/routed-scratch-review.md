# Independent routed scratch source review

Preregistered 2026-11-04T11:55:59.177740+00:00 before source audit. Review only `/Users/justin/.codex/worktrees/issue75-native-routed-scratch/mix-stq1_0`, baseline main21cf plus current local diff. Revisit the qualified native routed projection allocation/ownership contract for the new bounded scratch reuse mechanism: shape/overflow guards, aliasing and same-command-buffer lifetimes, error paths, and the physical fixture's actual allocation counts10/6. Read the local optimization journal/history first. No compiler, proof, test or GPU workload is authorized here. Status: completed; no production defect identified, one fixture coverage gap below.


The production path is consistent on inspection:

- Only `FlashExpert::from_backing` sets `routed=true`, fixing K2/MUL1/NT1 and the record geometry. The selector additionally requires exact one-row extents for gate/up/down; shared/from-projections, other batches and malformed-setting fallbacks remain separate. `fits` uses useful extents and widened u64 arithmetic, never backing length for dispatch geometry.
- Two new private buffers (5120/10240bytes) are local to one expert encode. Gate/up/SwiGLU/down destinations remain four distinct live buffers. Each existing compute encoder ends before the next, using tracked resources in one retained command buffer. Gate/up partial reads precede their reuse; down consumes its640-half prepared prefix while producing2560 partial floats. No offsets, kernels or numerical order change.
- The outer retained command keeps bound buffers after local owners drop, including before commit. No inter-expert/wave pool is introduced. Capacity/device/scratch-self-alias failures occur before preparation; later closure errors discard the uncommitted work. Production inputs cannot alias the newly allocated scratch pair.
- The allocation fixture samples the actual `newBufferWithLength` wrapper on the current test thread, after input/record upload: old3+3+1+3=10, new2+1+1+1+1=6 requested buffers. Successful output, exact nonzero finite independent words, actual three scratch bindings, destination nonalias, poison/tail preservation, dropped owners, repeated same-CB encodes and failure/recovery gates are present. These counts are expected assertions, not measurements by this review.

**Fixture finding (reported to root and owner before freeze):** `native/src/gpu/routed_scratch_tests.rs:105` counts distinct prefixes across the entire corpus, including the required final shared expert. A corpus containing one routed expert repeated with varied inputs plus the shared expert satisfies that guard. The same-CB sequence at line165 uses indices0/1/0 without checking that0 and1 identify different routed experts. It therefore does not enforce its advertised two-routed-expert coverage. The README's proposed generator does produce experts0/1, so its particular corpus is correct. Recommended hardening: require two distinct routed identities in `cases[..len-1]` and require the first two identities to differ (path+prefix if multiple checkpoints are allowed). This is a fixture coverage gap, not an observed product failure.

Limits: source-only review; no allocation/alias behavior, independent numerical parity, GPU hazard behavior or throughput was executed or proved here. Root still owns the physical component/composed-model gates. Review snapshot SHA-256 follows; subsequent fixes require checking only the affected lines.

- `native/src/flash_expert_scratch.rs`: `dbed12179c89ceb145a9f133d4f38c76d4548e545b7f1bad7d909161c7a2f21b`
- `native/src/gpu.rs`: `355469a433cfefe6b0f4d969e730110824755f0782a2c45af851cd8c9bda1704`
- `native/src/gpu/projection.rs`: `eea1bc2c5ef58264eb598e76be7dfdac6ed52009875cd31f6c1591adcf83e39c`
- `native/src/gpu/routed_scratch_tests.rs`: `459c896fa50ad6735767bc1a425af3e88e24e8a579533dffc8fa11edbe7ee70f`
