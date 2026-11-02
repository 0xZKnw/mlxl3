# T05 independent source review — 2026-10-29

Review of `ISSUE59-T05-WEIGHTED-01`, before the first physical GPU run. Read only; no implementation changes, build, test, GPU execution or prover run by this reviewer. The root agent owns verification results and the optimization journal.

**Conclusion:** no outstanding concrete implementation defect found after the rank-zero weight correction. The current candidate is suitable for physical correctness qualification; this review establishes neither numeric parity nor a speed gain.

## Baseline and reviewed sources

Compared `ssd_moe.rs` with the retirement snapshot used to reconstruct baseline CLI `a60eb4ada3a948110fc057d9551032b6d03598ab404201145f5a5410f26f8e75`. The snapshot is `docs/measurements/issue58-20261029/before-runtime-pool/native/src/ssd_moe.rs`, SHA-256 `f5145c58c83da2e6e08920b83eddc069f5fd4523759162e1234d042441522672`.

Reviewed source SHA-256 values at review time:

| File | SHA-256 |
| --- | --- |
| `native/src/weighted_epilogue.rs` | `eb51065aef9f636809b3ffb63dc78bdcbca034dee5bd5f9de893ed76df5aa09c` |
| `native/shaders/flash_weighted_epilogue.metal` | `ad4e58575c56d4bcbf91821efd77b5460c4090fdbafa0cf476a528e5cdfa0f8b` |
| `native/src/weighted_epilogue/tests.rs` | `201207af415ab64e2b46b1820da276d3e2983f8035c71d636d07830d2125d1c9` |
| `native/src/weighted_epilogue/capture_tests.rs` | `ee467a98baa59e64de73b84bda22c98f4c6fb482c6d1ba986bb394998e6be84b` |
| `native/src/ssd_moe.rs` | `4aa0b893e3a6c53ee6c5348a1def4d6dce040ffb00ce9c8fdb61990d56f31e0b` |
| `native/src/ssd_moe/hybrid_tests.rs` | `d8af2b4895f1ae9ad63c81dbc9fe13ee275f7036c0b3e3fa47ac0e7bbe6630d8` |
| `native/src/linear.rs` (unchanged from retirement) | `42106754300edc10d4b73b24a63720681626805acd0d97e27737642b973be18e` |

## Concrete findings

- **Previously discovered and corrected:** a scalar weight input would be emitted by MLX as a scalar reference, incompatible with `weights[row]`. `apply` now rejects rank zero before index upload or kernel construction; the negative R=1 case and positive `[1]`/`[1,1]` cases cover the chosen contract. Production weights have rank two. No reshape was added to production.
- **Mixed path is correctly delimited:** the new helper runs only with a physical Hot buffer and a nonempty Cold list. Fully Hot and no-Hot paths retain the original cast/multiply; the new real-MoE test asserts their kernel request counter does not increase. Default flag zero constructs the stock concatenate at its original point, before the CPU order loop.
- **Physical indexing matches the baseline:** early Hot uses the full occurrence buffer, so Cold starts at `ids.len()`, not the number of logical hits. Compact Hot uses `hot.len()`. The helper reads physical `hot.shape()[0]`. It preserves each canonical route occurrence and duplicates, including when all logical hits are absent but the resident snapshot produces a masked physical Hot buffer.
- **Split windows remain local:** `forward_unique_rows` completes the captured Hot graph before possible eviction, slices its input/output by the cohort's complete route rows, and passes corresponding local IDs/weights. The original order calculation consequently indexes the sliced physical Hot buffer correctly. The existing unique-window test actually forces `6+2` and `4+4` splits, eviction of captured future Hot, unrelated resident Hot, pending slot exclusion and I/O failure recovery; it is meaningful integration coverage with the candidate flag enabled.
- **Numeric operation order is preserved in source:** routed input is explicitly F16 before expert calculation; the down path returns its activation dtype, so Hot/Cold inputs are F16. The helper casts the selected source to activation dtype, promotes it and the weight to the original multiplication dtype, and writes one product. F16/F16 produces F16; all other supported combinations produce F32. The original `reshape([rows,10,H]).sum(1,false)` and shared add remain identical. The shader branches to select a source instead of adding masked buffers, avoiding contamination from unused poisoned values and avoiding an added zero operation.
- **Bounds are checked before submission:** geometry limits both physical heights and the canonical count to 1..80, validates every source index, checks source addresses against u32 and output elements against i32. Positive equal widths and supported dtypes are checked before upload. Weight count uses checked arithmetic and rejects nonpositive dimensions. These checks support the shader's row, column and linear address arithmetic.
- **Ownership and synchronization remain the original model:** the existing pending-Hot `last` retention, fill error handling and completed-Hot barriers are unchanged. The new lazy output owns its array dependencies and the newly uploaded order array. `apply` adds no evaluation, GPU readback, global fence or persistent array cache. The existing custom-kernel API materializes non-row-contiguous inputs; normal production Hot row slices, Cold outputs and route weight cohorts are contiguous. Strided component cases explicitly exercise that materialization contract.

## Qualification still required

The numerical tests use both stock MLX operations and an independent canonical CPU oracle. Canonical expected values are built from truth rows rather than from the candidate's physical-order lookup. They assert nonempty sizes, finitude, dtype, exact bytes, unchanged routed sums and unchanged input aliases. Poisoned unused buffers, reversed/duplicate source orders, all 1024 partitions, source/weight strides, R=1..80 and H=2560 are meaningful coverage. The literal test independently fixes products and signed zeros. At review time these physical tests are prepared, not run.

The two geometry Kani harnesses target real CPU guards, including symbolic addresses and all eighty canonical indices. Their source scope does not include MLX/FFI, Metal arithmetic, lazy graph lifetime, pending bank consumers or concurrent I/O. A verifier attempt or unsupported/timeout result must remain distinct from a successful bounded check; this review makes no proof claim.

Before timing, run the planned component oracles and real MoE integration with the candidate enabled. Retain the existing physical tests `demanded_reads_start_while_gpu_resident_experts_are_pending`, `early_hot_failure_preserves_ready_keys_and_recovers`, `wide_hot_is_opt_in_and_overlaps_routes_for_three_to_eight_rows`, and `unique_windows_preserve_outputs_pending_slots_and_greedy_fallback` under the same flag. They cover pending ownership and real split mapping beyond the component shape matrix. Full model logits/states, MTP/DFlash and bridge qualification remain necessary under the preregistration.

The capture test's counter proves one graph request only. Actual compute and blit commands still need inspection in the two `.gputrace` artifacts; neither a request count nor an encoder count proves physical dispatch reduction. No speed or recoverable-millisecond estimate is inferred from this review.
