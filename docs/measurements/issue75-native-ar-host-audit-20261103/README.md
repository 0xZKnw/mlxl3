# Native AR host audit — preregistration

2026-11-03, source-only, before runtime inspection or prototype. The user has
prioritized Flash Next AR; DFlash/MTP work is parked. No build, proof, model run,
benchmark, activation or gain claim is authorized by this audit.

Baseline: native runtime from main599628120c95 (GPU source unchanged by the
packaging-only #110 branch140025d9). A separate audit branch reuses that existing
checkout to avoid copying a large worktree during root's GPU queue. Published
packaging/source refs remain intact. Compare relevant runtime identities with the
frozen NT2/current AR graph where needed; do not attribute a prototype's timing to
another final source.

Read first: opti.md; T01 source/host audit in
`docs/measurements/issue59-t01-20261031/audit-before-decision.md`; P09 profile
limits; POSTP09 T01B/prepared evaluation and DEC-ONE negative/withdrawn results.
The historical T01 construction microbenchmark excludes GPU evaluation and does
not establish a model gain. DEC-ONE's model-level mixed/negative outcomes stay
rejected/inconclusive; a wrapper is not a new optimization.

Hypothesis to inspect, not assume: the independent native engine may insert
synchronous command completion or CPU copies between operations whose dependency
is entirely on Metal buffers. Unlike the old MLX lazy graph, this is a different
eager runtime. Identify actual consumers before suggesting that a synchronization
or readback can disappear. CPU route/expert IO decisions and normalized greedy
FP16 ties remain required boundaries.

Plan:
1. Trace actual AR bridge generation → target forward/prefill → layer operators
   → queue submission/completion and final normalized sampling/readback.
2. Count source-level barriers/copies for concrete straight-line paths, keeping
   branch/shape conditions explicit. Static counts are not measured latency.
3. Map old validated host/runtime mechanisms to already present, inapplicable,
   missing or rejected paths. Avoid NT2/kernel changes and SSD scheduling/prefetch
   work owned by the other agents.
4. Choose at most one narrow candidate only if real production callers and buffer
   ownership permit it. Preserve operation order, casts, states, route IO,
   publication and rollback. Record uncertainty rather than inventing a benefit.
5. Any executable prototype or timing needs its own precise preregistration,
   root CPU/GPU slot, meaningful source checks and exact finite logits/states/
   continuation plus real AR bridge quality before speed.

Status: preregistered source inspection only. No executable changes or jobs.

## Source findings

The AR loop uses native `FlashNext::forward/prefill`; it does not execute the
MLX array graph. Comparing main599628 with the frozen NT2 source89fdabbf shows
only the NT2 pipeline field/initialization in gpu.rs among the audited
forward/bridge/command helpers. The observed host boundaries below remain valid
for that graph; prefix-cache work changes request ownership but not this layer
boundary.

| Path | Actual native behavior | Decision |
| --- | --- | --- |
| Prepared pipeline/MLX ABI | Metal::new owns compiled pipeline states; resident projections encode actual buffer views. | Already present/inapplicable. Do not repeat the unfavorable prepared-wrapper port. |
| Full GDN attention | Normal forward already encodes projections, recurrence, norm and output in one Metal::command; intermediate values remain device views. | Already grouped; do not propose per-op wrapping. |
| Residual attention injection→MLP HC read | gdn.rs LayerResidual::forward calls attn_hc.write (commit+wait), immediately followed by mlp_hc.read (commit+wait). Only the written MatrixView flows between them. | Narrow missing grouping candidate. |
| CPU route/SSD demand | MoE waits for routing indices before reserving/loading experts, then overlaps shared/resident work with demanded IO. | Required boundary and another agent's IO scope; leave unchanged. |
| Final greedy/sample readback | The head downloads248320 half words, normalization uploads them and reads one half normalizer, then CPU builds normalized probabilities and selects. | Real transfer work, but preserving FP16 normalized ties is mandatory; no old isolated winning readback result. Defer. |
| Embedding conversion/repetition | Host reads one BF16 row, converts through F16 and repeats four F32 streams. | Concrete small copy, no measured recoverable budget. Defer. |
| DEC-ONE/old prepared evaluation | Old component wins did not establish stable model gains; prepared AR pilot was−1.710% decode/+5.704% prefill time. | Keep negative evidence; no kernel/factory port in this candidate. |

Chosen candidate: combine only attention HC write and immediately following MLP
HC read in one command buffer, keeping existing shader encoders, order, casts,
allocations and returned buffers. GDN and QSA use LayerResidual::forward; PLE
calls its GDN body. Each completed row of the current48-layer checkpoint reaches
this pair once:48 source-level command submissions/waits could disappear.
This is not a measured time, recoverable-latency bound or speed claim.

Ownership: both HC objects are loaded from the same Metal instance in
LayerResidual::load. The written stream is also needed after MoE for residual
injection, so it must be returned and retained. Read intermediates must stay
owned through command completion, including error cleanup. Keep the wait at the
end of the combined pair; do not globally remove waits or change router/SSD
publication. Existing target/layer snapshots must still restore on error.

Audit complete, source-only. A separately preregistered experiment is required
before prototype. No code, build, proof, GPU or measurement ran for this audit.
