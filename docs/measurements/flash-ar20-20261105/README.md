# AR20: native Flash Next AR experiments

AR20 implements twelve additional upstream-inspired native AR mechanisms above the previous ALL12 AR10 baseline. All twelve physical component gates passed, and the selected full-model/CLI compositions preserve the independent numerical reference. **No additional AR20 speed boost is established.** The final integration keeps the accepted AR10 composition active by default and measures10.302951tok/s on MAIN32 (10.019801 forced ON); this default-policy result is distinct from the experimental screens. The controls vary too much to attribute the best isolated result to an optimization. The user-directed QUIET-01 repeat is complete and remains **non-conclusive** because its three controls range from5.313 to9.717 tok/s.

Work is tracked in [issue133](https://github.com/0xZKnw/mlxl3/issues/133), [the preregistered matrix](matrix.md) and the root [optimization ledger](../../../opti.md). No MTP/DFlash, P04, HC L3–L5 or cancelled encoder work is included. All twelve new options remain experimental and OFF by default. No installed app, merge or release is implied.

## Scope and frozen artifacts

The measured model is Qwen3.8-Flash-Next, checkpoint `65c895314393431c09050b2e04e250836b3a6eb4`, EXL3 average 2.05-bit, on Apple M5 / 24 GiB. The previous ALL12 AR10 composition was explicitly enabled as the comparison baseline in these campaigns. The user has separately requested enabling those twelve AR10 mechanisms by default in PR132; that activation is now qualified and pushed in PR132 and does not activate the twelve AR20 experiments. Fresh default/forced MAIN32 samples there are10.424960/10.406348tok/s; this is a separate default-policy result, not an AR20 speedup. Its historical 8.349829 tok/s is context, not a substitute for the fresh controls below. See [AR10](../flash-ar10-20261105/README.md).

The MAIN protocol is 87 prompt tokens, 32 generated tokens / 31 timed decode tokens, one warm generation and one measured generation per arm, context 4096, cache 3 GiB/48×53 expert slots, four SSD readers, depth 0, overlapON, prefix cacheOFF, dynamic readersOFF, temperature 0 / top-k 1 / repetition 1. Prior AR10 flags stay explicitlyON; unrelated old options stay explicitlyOFF. Every arm must return the complete golden sequence through the real bridge, with bounded timeouts and joined child/readers. One model GPU process runs at a time; compilation and proving finish before the GPU campaign.

| Freeze | Source identity | Release CLI SHA-256 |
| --- | --- | --- |
| Common, completed campaigns | `41988dca2142b2e9082430517a376e93e5abbd23` plus the exact dirty-source hashes in [freeze-common.json](freeze-common.json) | `8d74a852d2bbdbfb08a6754c8e33585afa6b6e1fb54ed461c66c5aff26634875` |
| QUIET-01, current campaign | `b849f5a5833a5c656fde52557a1d987a34c8eaa7` plus the exact549-source/dirty-state inventory in [freeze-quiet.json](freeze-quiet.json) | `af0bc861b4af4e9ccd3a340cdd867f9516c6b2c4fc49368d570b10ea66cd9318` |

The native release libtest is byte-identical across these freezes: `157a86d728031888dd575282d0c193f6be439c3c84e289f88fb5aeeb35249192`. The final source delta comprises test-only cfg/CI/inventory work, with no shader or numerical-runtime change; the CLI hash still changed, so QUIET uses its own artifact-bound quality receipts. The old common binaries and measurements are retained. The earlier B01 screen has a separate [fast freeze](fast-freeze.json); it does not validate a later binary by itself.

## Correctness and verification

[COMPONENTS-01](components-01/report.json) passed 12/12 explicitly selected physical gates. These include independent codeword/reader/H128 oracles and mutation checks, frozen stock expert/resident projections, 60 QSA cases/368,640 Half words, resident bundles, the finish/mix gate with 10,240 Float32 mix words plus 76,800 stock Half words, real FD/inode/EOF/EBADF behavior, and state rollback. Nonempty shapes, finiteness and actual selected-path counters are required. Physical gates are separate from the ordinary ignored-test suite.

[QUALITY-01](quality/report.json) uses a separate ALL12/new-options-OFF reference process and an immutable SHA-bound oracle. The candidate comparison checks all 84 initial archived states, 57,606,144 state words, 993,280 logits and 81,920 hidden bytes, including continuation/restore behavior. The reference-capture phase truthfully reports `parity:false` because it creates an oracle; only comparison phases claim parity. The eleven-option candidate excludes B01 and includes B12 for numerical/cache qualification; the ten-option AR composition B02–B11 with B12OFF then passes its own selected-model gate. The real CLI golden/no-MLX/shutdown path also passes. [QUIET quality](quiet-01/quality/report.json) binds the final freeze/CLI to the independent reference again before its speed screen.

Local checks and their limits:

- DIRECT/chat: strict Clippy and format passed; 390 tests passed, 85 ignored.
- No-direct/chat on this Mac: strict Clippy and format passed; 379 tests passed, 10 ignored. This is not a Linux cross-build.
- Seven targeted Kani harnesses passed 932 reported obligations and 26 covers. The earlier 933 count was a bookkeeping error corrected from the preserved raw logs. These are bounded CPU source properties, not proofs of Metal arithmetic, driver lifetime, OS reads/advice or concurrent execution.
- The actual SLRU queue harness timed out without a verdict: capacities 10/11, six forced valid hit promotions followed by four symbolic `(u8 slot,bool hit)` updates, optional clear, unwind 40 with unwinding assertions retained. Its domain was not reduced and it remains unverified. The global Kani inventory also remains unverified after the known compiler ICE.
- The final focused Python regression suite passed 7 tests/16 subcases, and the native-inventory suite passed six tests. The reviewed source inventory passes at 272 files. Forty-four Linux-only orchestration tests were skipped locally. Controller source checks are recorded in [quiet-controller-checks.json](quiet-controller-checks.json); they do not run a model.
- CrossHair is unavailable, so no Python source proof is claimed. Earlier stale-manifest and cleanup-regression failures remain archived before the corrected passing runs. Hosted CI must be checked for the exact published commit; no global green-CI claim is made here.

Proof receipts: [attention](attention-proofs/README.md), [kernel mask](kani-kernel-mask/README.md), [IO/B09 plus SLRU timeout](io-proofs/README.md), [isolated B11 follow-up](io-proofs/b11-isolated/README.md), and [root B01/B10 source binding](root-proofs/source-verdict.json). The root B01/B10 historical terse log did not serialize exact argv or a process-group join receipt; those missing fields are explicit rather than reconstructed.

## Completed speed observations

Rates below are decode tok/s rounded to three decimals, with the same short MAIN32 protocol. They are observed screens, not durable throughput or independently established gains. Grouped arms cannot attribute a result to a single member. No gains from different groups or campaigns are added.

| First MAIN32 arm, in execution order | Decode tok/s |
| --- | ---: |
| Frozen AR10 control start |4.847|
| Common binary, all new optionsOFF |6.537|
| B05–B08 kernels |8.150|
| B02–B04 attention/resident |6.152|
| B08+B09 finish/mix |4.785|
| B10+B11 host/IO |5.011|
| Compatible B02–B11 |8.922|
| Frozen AR10 control end |5.097|

[First matrix and exact phase commands](main32/report.json). The compatible 8.922 and kernels 8.150 observations did not survive the necessary confirmations below. The [conditions snapshot](main32/conditions.json) shows an active desktop, including Equibop/video-related work; it does not establish a particular cause for the variation.

| Confirmation | OFF start | ON A | ON B | OFF end | Control endpoint drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compatible B02–B11 |6.083|5.210|5.076|9.564|+57.2%|
| B05–B08 kernels only |6.097|5.152|5.270|8.950|+46.8%|

The [compatible confirmation](confirmation/report.json) and [kernel confirmation](kernels-confirmation/report.json) retain all failures/limits. Across the first matrix and these two confirmations, 32/32 warm/measured complete outputs match the golden and all process groups/readers are joined. Quality is qualified; speed is **non-conclusive** under the large drift. The isolated high values do not establish an 8.9 tok/s improvement, and the completed campaigns do not meet the 10 tok/s target.

B01 is excluded from the speed composition after the audited early screen: 5.274 tok/s with head gates versus 8.420 with the same early binary's new optionsOFF. Its correct arithmetic did not rescue that adaptation's short speed result; the [fast receipts](fast/report.json) and [same-binary isolation](fast/isolation/report.json) are retained. B12 is OFF for decode after the [exact reconstructed replay and source audit](slru-audit/README.md): aggregate decode misses +1.7674% MAIN/+1.4473% HOLD, despite prefill misses −1.8394%/−2.5362%. These miss ratios are not throughput measurements; SLRU remains available for unmeasured prefill potential.

Before any rejection, [runtime-cost review](runtime-cost-audit/README.md) checks concrete B09/B10/B11 costs: no B09 PSO is loadedOFF; B09 moves ready-cohort finishes to the final command and has a wide-threadgroup mix tail; B10 can spend its bounded poll budget before blocking; B11 adds per-record identity/lookup work but only configures each inode once. These are plausible tradeoffs, not measured explanations for the drifting results.

## QUIET-01 after application closure

The user authorized normally quitting the observed heavy applications and repeating the screen. Equibop and Brave are confirmed closed; the Mac is on AC at 100% charge. macOS reports no thermal/performance warning, but `powermetrics` requires administrator privileges and `sudo -n` was refused without a password prompt. Temperature, frequency and throttling are therefore **not measured**; app closure does not establish a thermal cause or thermal stability. [Preparation receipt](quiet-01/preparation.json).

The user then cancelled the planned 90-second artificial idle. The sleep tool reported 20.8389 seconds before interruption; no completed 90-second cooling interval is claimed. The preparation receipt now records 58.6228 seconds of wall-clock preparation interval from epoch timestamps, which includes bookkeeping and is not a temperature measurement. Its original negative cross-process monotonic delta is preserved only as an invalid diagnostic.

The final quality gate and selected-model comparison passed before speed. All eight MAIN32 arms and their sixteen complete warm/measured golden outputs passed; every child is joined. B01 and B12 remain excluded.

| QUIET-01 arm | Decode tok/s |
| --- | ---: |
| Same-binary OFF start | 7.023266 |
| B05–B08 kernels | 5.896139 |
| B02–B04 attention | 5.424946 |
| Same-binary OFF middle | 9.717219 |
| B08+B09 finish-mix | 9.629735 |
| B10+B11 host-IO | 9.516374 |
| B02–B11 compatible | 9.446448 |
| Same-binary OFF end | 5.312801 |

The large control variation persists after app closure, so this repeat establishes no reliable optimization gain and no thermal explanation. Ten tok/s was not reached. AR20 options remain OFF; numerical qualification does not promote them. [Performance receipt](quiet-01/performance/report.json), [summary](quiet-01/summary.json), [controller](quiet-run.py.txt), [speed driver](speed-quiet.py.txt), [selected composition](quiet-01/selected.json).

## Experimental options

All twelve options default to OFF and accept literal `0`/`1`; M5/shape/capability guards and canonical fallbacks still apply. The four kernel options affect compiled shader specialization, so model comparisons use separate processes rather than an ineffective same-binary runtime toggle.

| ID | Environment option | Mechanism / origin |
| --- | --- | --- |
| B01 | `MLXL3_NATIVE_GDN_HEAD_GATES` | Per-head whole-GDN gates; oMLX |
| B02 | `MLXL3_NATIVE_QSA_CHAINS` | Split-chain QSA CTAs/paired queries; oMLX |
| B03 | `MLXL3_NATIVE_QSA_PREFETCH` | K/V software pipeline in canonical online order; oMLX |
| B04 | `MLXL3_NATIVE_GDN_RESIDENT_BUNDLE` | Resident qkv/z co-dispatch adapted to separate EXL3 inputs/outputs; oMLX |
| B05 | `MLXL3_NATIVE_MUL1_SWAR` | Packed byte-pair codeword sums; TurboEXL3 |
| B06 | `MLXL3_NATIVE_TRELLIS_READER8` | Eight-codeword lane reader; TurboEXL3 |
| B07 | `MLXL3_NATIVE_QMV_K_PRELOAD` | Two-K-tile preload; TurboEXL3 |
| B08 | `MLXL3_NATIVE_H128_SIMD` | Register/SIMD H128 with native rounding; TurboEXL3 |
| B09 | `MLXL3_NATIVE_MOE_FINISH_MIX` | Down-finish/H128/SVH plus canonical ten-route mix; TurboEXL3 mechanism |
| B10 | `MLXL3_NATIVE_GPU_SPIN_WAIT` | Fixed 50µs/4096-poll spin before blocking; Strata |
| B11 | `MLXL3_NATIVE_EXPERT_READAHEAD_OFF` | `F_RDAHEAD=0` on independently opened expert descriptions; Strata |
| B12 | `MLXL3_NATIVE_CACHE_SLRU` | Per-layer 50/50 SLRU at unchanged 53 slots; llama.cpp |

Forced LRU/prefetch overrides SLRU. B09 suppresses the old A05 output slab only when the new path is actually selected; the other AR10 mechanisms remain distinct. The [matrix](matrix.md) states each numerical contract, fallback and new reason for revisiting a mechanism already considered in another runtime. Other models, GPU families, long-context behavior and sustained throughput are not newly qualified by these short M5 checks.

## Evidence, commands and attribution

Exact argv, flags, timeouts, source/binary hashes and process cleanup states are in the per-phase JSON receipts linked above. Controller snapshots are [profile.py.txt](profile.py.txt), [components.py.txt](components.py.txt), [quality.py.txt](quality.py.txt), [speed.py.txt](speed.py.txt), [speed-confirm.py.txt](speed-confirm.py.txt) and [speed-kernels-confirm.py.txt](speed-kernels-confirm.py.txt), alongside the two QUIET drivers. The [ALL12 profile](profile/report.json) is diagnostic only; overlapping CPU/GPU intervals must not be summed into a predicted gain.

The [archive manifest](archive/manifest.json) records raw/compressed paths, sizes and SHA-256 hashes; gzip round trips are checked byte-for-byte. The [archive guide](archive/README.md) and [closed-campaign addendum](archive/README-addendum.md) explain restoration and explicitly separate previously live scopes. Resident projection fixtures are in [components-fixtures.tar.gz](components-fixtures.tar.gz) with [the member/hash index](components-fixtures.index.json). Original local logs remain retained; readable receipts and earlier negative/non-conclusive results are not overwritten. QUIET closed later and is archived in a separate addendum; the initial manifest remains immutable.

Thanks to [davidtai/TurboEXL3](https://github.com/davidtai/turboexl3/tree/c3bb0e260091709f8ad64c84d1fd02808b12dfd1) for implemented EXL3 decoder/reader, preload, SIMD transform and finish/mix mechanisms; [jundot/oMLX](https://github.com/jundot/omlx/tree/2165e75726ba83d9651dbaced61c97728da77bba) for GDN/QSA scheduling examples; [Niko1221/Strata](https://github.com/Niko1221/Strata/tree/ffa80d73133ea5c548aefe57889a16e8085a938f) for host-wait and file-advice examples; and [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/2bc46baa9207d098d03741ac394e7faaf1a955b6) for SLRU and its capacity-dependent outcomes. [MLX-LM](https://github.com/ml-explore/mlx-lm/tree/736ab0d8c1d0edc79c3d798044fe6904e2fa23be) supplies the previously established canonical recurrence basis; [ashhart/TensorFold](https://github.com/ashhart/TensorFold/tree/0311c9280878775ea8167d2e26bb7068d7cebc6f) was examined for already-covered expert-axis ideas. These latter references are not counted as extra new gains. Pinned source/caller links and adaptation limits are in [Turbo research](turbo-kernels.md), [GDN/QSA research](gdn-attention.md) and [SSD research](ssd-schedule.md).

## Final default-policy integration

PR132 head `e555bc89` is merged into this branch. All twelve accepted AR10 mechanisms now initialize ON on the qualified M5 without optimization environment variables; all twelve AR20 experiments remain OFF. The final [five-phase integration receipt](default-integration/report.json) passes no-override model/84-state archive/full logits+hidden/continuation and counters, explicit-OFF six-PSO absence, real no-override CLI/no-MLX/shutdown, then MAIN32 default **10.302951 tok/s** versus forced **10.019801 tok/s**. All four new warm/measured outputs match. No additional experimental gain or sustained throughput is inferred.

Final [freeze](freeze-default-integration.json): source `0e59616c` with551 exact hashes; native-libtest `4577d8cd14812f57bf312c8329e27be6cfa44fa0ecb161bc6e6c69eb83e4446b`, CLI `3a8547c9061a5bb09131edc02784949bf2453d25929bfb2611a247f8b619d5e7`. Post-freeze changes are reports and one import-spacing-only Python verifier lint fix; all compiled Rust/Metal/C++/Cargo source hashes remain identical. The [controller](defaults-controller.py.txt) is byte-identical to the qualified parent controller. [CPU policy source equivalence](default-integration/source-proof-reuse.json) verifies the complete AR10 policy file and the expert flag/proof/callee bodies before reusing their bounded receipts. Final CPU suite391PASS/86ignored, fmt and strict Direct/no-Direct Clippy pass; inventory273files and eight portable manifest/cleanup tests with16subcases pass. Earlier proof timeout/ignored/Linux-only and non-conclusive measurements remain explicit above.

Logs and raw source identities are preserved in the [final integration archive](archive/root-final-integration.json), with its [restore guide](archive/README-default-integration.md). Publication is a stacked PR on PR132, with no merge, app installation or release.
