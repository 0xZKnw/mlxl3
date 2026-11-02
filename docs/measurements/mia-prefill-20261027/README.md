# Mia prefill tile → M5/EXL3 — rejected, 27 October 2026

The requested adaptation was implemented and tested, then completely removed after a negative component screen. The retained HEADONLY improvement is unchanged. No new kernel is enabled in the CLI or installed app; no push or publication.

## Implementation and scope

Source idea: [Mia patch0014](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/blob/e1f564eb915bb5272474b6c1c527aec3b9ed3de6/patches/0014-engine-prompt-expert-tile.patch), shared decoded weights across several rows. This prototype uses our EXL3/MUL1/FP16 math and preserves the M1 FMA, split and reduction order. It does not copy CUDA cp.async, tensorcore accumulation or NVIDIA FP4 arithmetic.

Two to four rows share a decoded tile in threadgroup memory; each row retains four SIMDgroups for its original partition of K. All threads reach the barriers. At most512threads and20480bytes of threadgroup arrays. The existing route partitioner produces groups of at most4, so larger matrices comprise multiple such groups. Singleton groups use the original shader. Only an explicit test constructor selects this prototype; there was never a CLI prefill/decode connection.

Call path exercised: real checkpoint/SSD record fill → ExpertSlots → expert_slots/expert_mapped_layout → native MLX Metal compilation/dispatch. Both raw FP32 sums before the output transform and complete gate/up, SwiGLU and down contributions were checked. Independent resident Exl3Linear projections read the checkpoint separately for the complete contribution oracle.

## Correctness before speed

M5 standard,24GiB, MLX0.32.2, Flash EXL3 checkpoint65c89531. Layers0/23/47, real expert weights0..4, bank capacity53. Deterministic synthetic inputs, repeated or mixed expert IDs, F16 contiguous/F32 strided, rows1/2/3/4/5/7/8/16/32. These are routed projection occurrences, not32whole prompt tokens selecting320experts. No full target model or draft model was loaded by this component test.

-108complete cases passed, with full shape/count/finiteness/byte equality to independent resident projections. For every case, raw GU and down FP32 results also matched the original kernel:216raw cases, two comparisons in v1 and three in v2. Launch counters required the actual new shader for repeated rows2..32; rows1 remained a fallback. Invalid counts/slots/capacities were rejected.
-Normal native library v1:182passed/157ignored; CPU suite247passed/6ignored. Strict Clippy with mlx,chat/all-targets, format and release libtest builds passed. The v2 change adds only the control with forced grouping; its108physical cases passed again before timing.
-Mutating the shared shader's initial sum from0to1024 triggered native exit101 immediately after four successful M1 controls, at the shape/count/finiteness guard on the first shared M2 case. The wrapper expected the later byte-comparison error string and consequently marked the run failed. That failure is preserved in negative-plan-result.json. [Independent inspection](negative-inspection.json) establishes functional corruption detection, not a successful wrapper run. The normal shader was restored byte for byte before inference.
-Kani0.68.0 checked the actual Rust width selector for arbitraryu32 with no assumptions:106obligations/0failures,3/3covers, selected widths2..4, threads≤512 and shared arrays≤20480bytes. **CPU bounded model check only**, on the subsequently removed prototype. Actual MLX entry attempts v1/v2 hit the compiler ICE at intrinsics.rs:243 before obligations. Metal/GPU not formally proved. xcrun--findmetal failed72; the real MLX runtime Metal compiler did compile and run the shader.

Initial GPU admission was blocked by an external Swift build before spawning a child. It was allowed to finish. The subsequent anchor41013133312bytes Swapouts stayed unchanged through all four completed GPU groups: v1 library, v1 physical quality, v2 quality and v2 speed, plus the separate negative control. All children were joined. No other task was stopped or messaged. RSS fields in plan results are child maximumRSS, not a complete Metal allocation/physical-footprint measurement.

## Speed screen

V2 uses four variants:0original non-grouped,1existing grouped with its existing heuristic,2new shared tile with explicit grouping,3the exact same grouping as2 with the original register-sharing kernel. Variant3 prevents attributing removal of the old16occurrence threshold to the new shader. V1 had no speed measurements.

Three warmups,12alternating rounds per cell,32forwards per sample, every sample retained. Rows1/2/4/8/16/32 ×2patterns ×3layers ×3components =108cells,5184timed samples,165888forwards. Raw GU/down operate on already transformed inputs; contribution includes transforms/activation and route construction. Times include host dispatch, GPU work and fencing, **not exclusive GPU time**. No timed expert I/O. Whole run79.535s, zero new Swapouts, child joined.

Changes below are component throughput ratios: median of12paired ratios per cell, then descriptive median across the three layers for repeated rows. All per-layer ranges, mixed patterns and absolute milliseconds/call are in [analysis-v2.json](analysis-v2.json); all samples in [speed-v2.json](speed-v2.json).

| Rows sent to an expert | Raw GU vs original | Raw down vs original | Complete contribution vs original | Raw GU vs same grouping/original kernel |
|---|---:|---:|---:|---:|
|1|+0.10%|+2.47%|−1.65%|+0.99%|
|2|−10.02%|−7.42%|−4.17%|−3.73%|
|4|−4.39%|−4.31%|−4.57%|−8.63%|
|8|−2.71%|−3.67%|+0.08%|−18.53%|
|16|−2.68%|−0.23%|−0.10%|−23.31%|
|32|+0.89%|−0.46%|+1.24%|−32.72%|

M1 uses the same GPU source; its fluctuations are controls, not gains. The isolated+1.24% contribution at32rows is still−33.29% against the same grouping with the original kernel. Mixed patterns do not establish a useful new shared-tile advantage. The original grouped kernel already reuses decoded values in registers. Added staging/barriers plausibly account for losses, but no exclusive GPU profile establishes causality. This rejects this concrete adaptation, not every possible Metal prefill kernel.

## Decision and retained code

**Rejected before model integration.** The preregistered useful-component condition failed, so complete prefill, MTP/DFlash2 verification blocks and decode A/B were not launched. No model slowdown or speedup is inferred from the component screen. The current prototype would not help the requested path; P01B/P01L and the old grouped experiment were not re-enabled.

All new Rust changes and the shader/test module were removed. The three retained HEADONLY source hashes and frozen baseline engine are checked against the earlier final manifest. Unaffected full-model correctness/bridge checks are reused by source identity; no new GPU run was performed just to repeat unchanged calculations. Format/strictClippy/diff checks passed after restoration. No new CI check is needed for removed production code, and no remote CI covers unpublished local changes.

Provenance: source-v1.json/source-v2.json, full snapshots source-v1/ and source-v2/, prototype-v1.patch/prototype-v2.patch, mutant-shader.metal/mutant.json and logs. The frozen test binaries remain in build/mia-prefill-v1/ and build/mia-prefill-v2/, not the app. [Restoration receipt](closure.json).

Commands: cargo fmt --all; cargo clippy --features mlx,chat --all-targets -- -D warnings; cargo test --no-default-features --features chat; cargo test --release --lib --features mlx,chat --no-run --message-format=json; cargo kani --lib --no-default-features --harness exl3_prefill_tile_has_bounded_threadgroups; cargo kani --lib --tests --features mlx,chat --harness ssd_moe::mia_tile::actual_mia_shared_tile_entry. MLXL3_MLX_ROOT fixed to the pinned local MLX package. GPU commands, environment, exact test filters/timeouts/expected exits, before/after system snapshots and cleanup are in the *-plan.json/*-plan-result.json files. All trials are closed. Next requested work is Sushi research, replacing the TensorFold pool follow-up.
