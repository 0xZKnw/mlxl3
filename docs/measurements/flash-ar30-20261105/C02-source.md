# C02 paired router — source prototype, unqualified

Base574e4a8c, issue135 and C02 matrix preregistration precede this source edit.
`MLXL3_NATIVE_ROUTER_PAIR` uses the existing strict experimental-OFF/exact-M5
policy. OFF creates no paired PSO. `gpu.router_pair` is the only new central
holder; its `enabled` Cell and test `uses` counter are visible to the composed
model gate. No accepted AR10 or AR20 defaults changed.

Each SIMD computes two fixed output rows, sharing one activation load per
column. Every output keeps the canonical80 FMA sequence per lane and the
16/8/4/2/1 shuffle tree. The new grid is256×batch,32threads, against the
independent existing512×batch canonical PSO. Shape stays exactly[rows,2560]
to512 outputs; there is no odd-output tail in this fixed model. Signed row/
column strides and offsets are unchanged. Original validation rejects partial
widths/overflow before encoding. Missing/unsupported paired pipelines select
the canonical path.

All production callers already converge on `FlashGate::encode`:
submit, predictor, trace, and HC/MoE chains. Softmax, TopK, score normalization,
destination ownership and producer waits are unchanged. The counter increments
only after the paired dot dispatch is encoded, once per encode, not per batch
row. Successful numerical qualification must also establish command completion.

New production CPU helper/harness:
`ar30_router_policy::paired_router_actual_geometry_preserves_every_reference_output_bound`.
Symbolic enabled/shape/capability values, no assumptions, four covers; checks
exact admission,256pairedgroups, overflow-safe final output and complete pair
indexing. It is prepared but **not run**; root must add the exact harness name
to the combined proof registry. This would prove CPU geometry only, not Metal.

Prepared physical component:
`gpu::gate::ar30_router_tests::physical_ar30_paired_router_preserves_reference_words_strides_and_routes`.
Marker `AR30_ROUTER_PAIR`:10cases,11264finite logits,11264finite probabilities,
220IDs/finite scores,10actual paired uses. Both F16/F32, rows1/3, stride2,
negative and broadcast columns, analytic uniform ties, wrong-second-weight-row
counterfactual, and invalid shape before command encoding. Canonical and
candidate selectors are genuinely OFF/ON in the same loaded GPU; the former
keeps the original dot PSO. No separate full-model test was added: root owns
one composed C01/C02/C03 trajectory/state/CLI gate.

Formatting with the explicit installed rustfmt path and `git diff --check`
passed. First bare `rustfmt` lookup failed because the tool shell omitted
`~/.cargo/bin`; no compiler or test was launched by that failure. No Cargo,
lint, Kani/CBMC, Metal compilation, component/model/CLI or speed test has run.
No tolerance, verifier domain or numerical oracle was weakened. This source
prototype is neither correctness-qualified nor a measured performance gain.
