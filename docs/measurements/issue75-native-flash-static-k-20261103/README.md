# Native Flash static K/codebook qualification

The optional native specialization preserves the canonical QMV body and fixes only K, codebook and packed-word width at compilation. Five MUL1 PSOs cover K2/NT1, K3/NT1, K3/NT2, K4/NT1 and K4/NT2. `MLXL3_NATIVE_FLASH_STATIC_K=0|1` defaults to 0; NT selection is preserved. The immutable Metal-context setting avoids environment lookups on each cold expert record. Generic families, fc_hidden, K7 and other codebooks retain their original selection.

Source checkpoint: **b8f03233**, based on default-NT2 candidate 717b9231 over main17c66327. The executable production source remained unchanged while qualification fixtures were completed. Frozen release CLI SHA256 **d2ff7420acae31ca0123e5ccff9d12bec5198c47947a739c2c51e016c3b04c8e**; final direct libtest **9825ac7debdfaf22e1d6ea7d532df6451857cf9a35f751878192fba975dd7613**. Independent MLX oracle **405b025f5c6ecb7c6ca08b18a25b75ff98e7260d3272cc283887ab9b9764284d**.

CPU and bounded verification evidence is in [cpu-receipt.json](cpu-receipt.json). The archived CPU suites pass 326/341/377 tests with 8/39/257 ignored respectively. Actual unchanged Rust helper Kani checks cover arbitrary selector inputs and UTF8 inputs up to four bytes with unwind10: 290 checks without failures, 13 covers satisfied; seven standard-library obligations are unreachable. This does not prove Metal, allocation, concurrency or inference. New evidence-validation tests pass 7 tests plus20subtests; CrossHair's two helper postconditions remain **Not confirmed** after15seconds each. The failed initial numpy import and all inconclusive results remain archived.

## Physical quality, 2026-11-04

[Plan](quality-01/plan.json), [master results](quality-01/report.json), [component 0](quality-01/component-0.json), [component 1](quality-01/component-1.json), [production AR](quality-01/ar-1.json).

All six phases pass, in152.68seconds total, with every owned parent joined and process group empty. This is correctness evidence; phase durations are not benchmarks. Root granted the exclusive GPU window after builds/provers drained, AC100%; known desktop Equibop CPU activity was recorded. No concurrent inference was observed.

- Each component setting runs the existing48-case independent stock matrix in NT2=0/1/unset modes: **2,214,912 finite Half words exact per setting**. Disabled static counts are all0. Enabled actual PSO counts, in the order above, are **[6,32,28,16,32]**. Invalid shape/input and both flag rejections pass; malformed static configuration retains nonempty exact K7/generic fallback controls. fc_hidden remains excluded.
- Both static settings pass the independent archived target oracle: **84 states /28,803,072 finite F32 words and496,640 full logits per setting**, including imposed continuation and reset. Enabled actual counts **[5760,0,0,480,723]** confirm the cold K2 record path and resident K4 paths execute their specialization. These target tests use qualified lookahead1; production AR below uses lookahead0.
- The measured-memory budget/admission, alias accounting, shrink and replay fixture passes with static1.
- The frozen release CLI with static1 passes the real AR bridge: four independent golden two-token completions (`9f850af1cbba074b`), invalid-message rejection, context and delta cancellation followed by exact recovery, native ownership ledger and48×53 cache slots.706 DYLD log entries are audited without MLX/JACCL; this count is not a count of distinct images.

Raw logs, synthetic checkpoints and numerical output files remain local under `quality-01`; the adjacent ignore file prevents accidental publication of the69MiB fixture tree. The metadata reports preserve individual counts and source identities.

Performance is **not measured by this qualification**; root owns the separate alternating release screen. Composition with HC or other candidates is unqualified. The component matrix uses contiguous Half/Float views; a dedicated new strided-input matrix was not repeated because preparation is unchanged. Five PSOs are created even with the flag disabled; their cold-start cost has not been isolated. No push, merge, app replacement or release has been performed for this candidate.
