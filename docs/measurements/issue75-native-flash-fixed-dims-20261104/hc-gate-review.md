# Independent HC→gate source review

Reviewed 2026-11-04T11:55:21.516690+00:00; source worktree `/Users/justin/.codex/worktrees/issue75-native-flash-hc-gate/mix-stq1_0`, base `21cfc9ad0b6679f9e6975bf6f02bc7a5b7b96d47` plus the local diff. Scope preregistered with root before detailed inspection: revisit the HC118 owner/lifetime/order guarantees for the new gate consumer, including callers, errors, prefetch, snapshots and fixtures. This review modifies no executable source and runs no compiler, verifier, test or GPU workload.

**No concrete defect identified in the reviewed source.**

- `LayerResidual::forward` keeps attention → HC write → HC read → gate dot/softmax/top10 → route publication → demand SSD/expert computation → MLP residual order. The kernels and their numeric operations are unchanged.
- `read_after_write_with_gate` holds both `ReadBuffers` and `GateBuffers` in the closure result through the existing `Metal::command` commit, wait and completion-status check. Only then are `RouteOutput` IDs/scores exposed. Command-buffer resource retention and the same queue preserve the producer/consumer order.
- HC instances and MoE must share the same `Metal`; gate encoding additionally compares the actual command queue. A foreign MoE can fail after out-of-place HC encoding but before commit; the uncommitted command is discarded. The owner/queue regression exercises this failure and subsequent recovery.
- Joint selection requires the new flag, HC enabled, depth zero and no prefetcher. Other modes use the existing route, polling and prediction path. The `next` MoE is unused only in the depth-zero branch, which has no prediction to schedule. FlashNext snapshot/restore and prefill rollback remain unchanged.
- The gate fixture uses a nonzero independent analytic uniform oracle for F32/F16 logits, probabilities, tie-selected IDs and scores, with strided batches and foreign-owner/queue errors. The AR fixture primes both restored QSA branches symmetrically, retains logits/hidden/84 full-state comparisons at prefixes0/2/9 and counts 48 removed **standalone gate** command buffers separately from unchanged helper commits and wait calls. Archived depth-zero numerical evidence and the predictive fallback have separate gates.

Limits: this is source inspection, not numerical qualification, a Metal proof or performance evidence. The full-state archive covers its existing stream/GDN contract; no extra QSA sparse-context claim is inferred. Subsequent edits require checking the affected source again.

Reviewed SHA-256 snapshot:

- `native/src/flash_support.rs`: `02589a78f742810d2ac07e4f110195bd6bc53f7689e890b7682eb6b3e36a36db`
- `native/src/gpu.rs`: `174da3db874f7bf1b3c74bfde8732c9df80d7eb5837f78cb89690c39095b9c47`
- `native/src/gpu/hc.rs`: `332dae50175f16ed144be090f9c4ecccd20ee6df0832a6acf79cfe7d5a89dc6a`
- `native/src/gpu/gate.rs`: `a89c331329ccb87433d30a07da4fd694b50e00b2fbbfceacaabb503decb2c5f3`
- `native/src/gpu/gdn.rs`: `0b6c9d9d22dbc153117abdb7494f8e5453130eca266ff7741a88b2e8d59ad178`
- `native/src/gpu/moe.rs`: `bc61f50ea4372f99c8613b6b772bdef1e2ccf47f3c852e8228065804ecb21ae2`
- `native/src/gpu/router.rs`: `5aec15c0225f231cb8e1bd205b979952eef0b3090497bfabd19e0e6be8a37c80`
- `native/src/gpu/flash.rs`: `656272a6a0f6c0494e798550aa076227f16486da54d1348aadef2f9fda205970`
