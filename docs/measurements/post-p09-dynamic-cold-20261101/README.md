# POSTP09-DYNAMIC-COLD-01 — Dynamic readers with Cold U

Update, 1 November 2026: Justin explicitly accepted the +0.700% short-MTP
prefill tradeoff and requested the default. [Default activation and its checks](../post-p09-dynamic-default-20261101/README.md)
supersede the activation decision below; all original measurements and limits
remain unchanged. The following status describes the original campaign closure.

Status: **quality validated; small decode gain observed; promising code retained opt-in for composition**. Campaign closed, all planned runs inspected. Dynamic remains off by default because short-prefill MTP results are still mixed. No installed app, commit, push or publication change.

## Scope and hypothesis

The previous Rust loader explicitly disabled Cold U when Dynamic readers were requested. The asynchronous `DiskReadBatch` also used static stripes exclusively. This change reuses the existing `DiskWorkQueue` for that batch and removes the exclusion. It preserves the number of workers (four), expert cache (3 GiB, 53 slots per layer), math, canonical expert reduction, and the two Cold waves.

Reference A is the same frozen executable with `MLXL3_SSD_DYNAMIC_READERS=0`; candidate C uses `1`. Both use `MLXL3_SSD_COLD_READY=1`. This measures their composition against current Cold U, not against an old non-Cold baseline. The earlier T18a synchronous experiment does not measure this composition. The existing local MTP prefill-head optimization is identical in both arms.

`manifest.json` contains the pre-registered bridge17 AC, short128 ACCA and conditional long512/64 CA matrix for AR, MTP depth one and DFlash2 mode one. Each process warms each prompt once and measures it once. MAIN has 87 input tokens and HOLD 82; 128 outputs correspond to 127 decode transitions. No tracing during release speed measurements. All samples are retained.

## Identity and resources

- Source HEAD: `d5a776e53425bb4d0ba61c0774c5399ae7987c06`, with existing local changes preserved. `before.json` and `before/` separate the state before this port from `candidate-source/`.
- Frozen executable: `build/post-p09-dynamic-cold-20261101/cli`, SHA-256 `689548f8e07e341804f902730ba638988c645a6bc3de3e438d4cd13d4414131a`.
- Frozen libtest: same directory, `libtest`, SHA-256 `85d7d0d603aa054a963d6272a7f7aa3a2d9679e0bcea8e1653397589f2d3e9f4`.
- `identity.json` pins 188 source/plan files and three executables, including Python. The existing supervisor checks identity before each process.
- Apple M5, MLX 0.32.2, Flash Next EXL3 2.05 bpw `65c89531`; DFlash2 draft `a22a1744`. Absolute checkpoint/oracle paths and all environment options are in each plan.
- Existing authorized `decode-profile-p09-20261101/anchor-mtp-head.json` is retained: cumulative 32 MiB swap writes / 256 MiB swap reads. No reanchor. Model admission 10.5 GB estimated free plus file-backed memory; RSS cap 10 GiB, native lifetime footprint below 12 GB. AC and quiet-process checks; one model, no compiler/prover concurrent with GPU.

## Implementation audit

The new batch owns a bounded job counter; claiming an index uses relaxed atomics only for uniqueness. Existing release/acquire publication of each completed record still protects payload visibility. The same GCD callbacks read directly into the existing Metal buffers. No new payload copy, staging allocation, MLX call from workers, or GPU fence was introduced. Worker count and file ordering metadata are unchanged.

Dynamic falls back to Static for one worker or at most one job per worker. Cold U still requires B1/B2, no prefetch pipeline, a Hot result and more than two missing experts. Wider verification, empty-cache and small-lot paths retain their existing fallback. The flag also selects the pre-existing Dynamic synchronous reader on those fallback paths; model measurements therefore measure the complete requested option, not an isolated asynchronous batch.

The first Cold wave still starts after two published records. Dynamic can change which records arrive first and the number already ready when the CPU wakes. That can change wave sizes, GPU dispatch efficiency and overlap. A better-balanced SSD queue alone does not guarantee faster token generation. The completed natural-wave diagnostic is detailed below.

Errors remain stored by canonical job index; finish joins every worker before reporting the first canonical error. Batch destruction joins readers before buffer owners are released. Existing consumers retain Hot/W1/W2 arrays through success/error handling. Flags are read once per thread; A/C comparisons use separate processes.

## Completed verification

- C++ ASan+UBSan: 83 cases pass; TSan: the same 83 cases pass. Static/Dynamic, 1–4 workers, 1/2/3/6/20/512 jobs, partial reads, EINTR, invalid metadata, truncation, timeout, held job, destruction and error recovery. See `async-asan.log`, `async-tsan.log`.
- Negative mutation: an isolated copy forced the actual queue to Static while reporting Dynamic. The held-job test failed with a readiness timeout, as required. Production code was unchanged. See `mutation-result.json` and `mutation.log`.
- Rust CPU suite: 276 passed, 9 ignored. Rust release MLX/chat suite: 311 passed, 203 ignored. Ignored physical tests were not counted as coverage from these suites. Python relevant suite: 1,311 passed. Strict Clippy, Rust/Python format, Ruff and diff whitespace checks pass.
- Physical selections: initial 10/10 pass, plus both corrected target-oracle selections (12/12 total). Storage checks cover 36 cases and 3,288 records per flag, including 18 actual Dynamic batches under flag1; invalid option leaves destination bytes unchanged.
- Cold quality in both schedules: 20 mixed and 20 all-Hot cases each, empty/small/wide fallbacks, duplicate routes with distinct activations; byte-exact component outputs and compute-before-held-tail verified. Prepared/fenced/error/drain/invalidation/recovery checks pass.
- Independent target oracle: five steps, all 48 layer outputs, selected GDN states and logits/reset checked. MTP budgets 1/2/3/17 preserve sequence, target and head state. DFlash widths 1–8, all 36 retained-prefix cases, invalid/abandoned verification and continuation preserve target logits and states byte-exactly; 64,000 capture values equal the independent fixture.

Kani was attempted on the actual Rust Cold entry with symbolic failure-path selection, but failed before obligations with compiler ICE `intrinsics.rs:243` (8.37 s, peak ~895 MB). CBMC was first absent from PATH; the discovered Kani-bundled CBMC 6.11 then failed parsing system C++ headers before obligations. These properties remain **unverified formally**; sanitizer tests and sampled physical oracles are not complete proofs. Logs and bounded process results are retained. CrossHair is unavailable. No prover was running during GPU tests.

The existing `.github/workflows/rust.yml` already executes the modified C++ tests with both sanitizers, proof syntax and relevant Python tests on PRs/main. No remote CI run exists for this unpublished local diff.

## Reproduction

Recorded invocation from the repository root, using the installed project Python:

```sh
/Users/justin/Documents/mix-stq1_0/.venv/bin/python \
  docs/measurements/decode-profile-p09-20261101/run-mtp-head.py \
  docs/measurements/post-p09-dynamic-cold-20261101/quality-01.json \
  docs/measurements/post-p09-dynamic-cold-20261101/identity.json
```

Each numbered quality plan and named bridge/short/long plan records the exact command/options, timeout and environment. The shown historical identity predates the test-only oracle correction; the current source corresponds to `identity-confirm.json`. A new reproduction requires fresh result paths and a pre-registered identity, rather than overwriting existing results or bypassing the source check. Do not reuse resource authorization after its cumulative limits are exceeded. The wrapper checks identities, quiet conditions and memory, joins children, and records incomplete/error results. For bridge qualification, the existing `validate` and external `check_golden` functions are applied independently to every warmup and measured response; numerical parity comes from the separate native tests, not token hashes.

CPU/build commands: `cargo test --locked --no-default-features --features chat -- --test-threads=1`; `cargo test --locked --release --features mlx,chat -- --test-threads=1`; `python -m pytest -q tests/test_flash_*.py tests/test_decode_capture.py tests/test_decode_profile_p09.py tests/test_decode_p09_attribution.py`; `cargo clippy --locked --features mlx,chat --all-targets -- -D warnings`; `cargo fmt --all --check`; Ruff format/check on the modified Python test. Builds use one job, the installed MLX root and Xcode Developer directory recorded in the plans.

## Final results and retention

All 28 bridge/benchmark processes completed: 100 warmup/measured responses match the external output references. MTP/DFlash acceptance/block counters are identical across arms for each prompt. Twelve physical test selections and two complete native diagnostics also passed. All 42 supervised processes were reaped.

Short results below include the entire original ACCA pilot and the single CA confirmation for MTP/DFlash. AR has two measured processes per arm; MTP/DFlash have three. Each process measures MAIN and HOLD once after warmup. Throughput is total decode transitions divided by total decode time, not an average of percentages. Prefill columns report relative time, so negative is faster. All individual samples remain in `final-summary.json` and the raw reports.

| Mode / prompt | Static + Cold U tok/s | Dynamic + Cold U tok/s | Decode change | Prefill time change | Whole generation time change |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR / MAIN | 11.4089 | 11.4667 | +0.507% | -1.212% | -0.798% |
| AR / HOLD | 11.3026 | 11.3661 | +0.562% | -0.651% | -0.595% |
| AR / ALL | 11.3555 | 11.4162 | +0.535% | -0.941% | -0.698% |
| MTP / MAIN | 11.9761 | 12.0855 | +0.913% | +0.197% | -0.434% |
| MTP / HOLD | 11.1442 | 11.2560 | +1.002% | +1.236% | -0.114% |
| MTP / ALL | 11.5452 | 11.6560 | +0.960% | +0.700% | -0.273% |
| DF / MAIN | 12.5067 | 12.6517 | +1.159% | +0.628% | -0.362% |
| DF / HOLD | 12.0423 | 12.1516 | +0.907% | -0.682% | -0.808% |
| DF / ALL | 12.2702 | 12.3966 | +1.030% | -0.008% | -0.585% |

Long input: 512 input tokens / 64 outputs, one C-A pair per mode, one warmup and one measured response per process. The older P09 long report is used only as an output oracle, with its SHA recorded in `long-golden-reference.json`; it is never a timing baseline.

| Mode | Static decode tok/s | Dynamic decode tok/s | Decode change | Prefill seconds A → C | Prefill time change |
| --- | ---: | ---: | ---: | ---: | ---: |
| AR | 10.8294 | 10.8407 | +0.105% | 42.5382 → 42.2586 | -0.657% |
| MTP | 11.7333 | 11.8509 | +1.003% | 43.1538 → 43.2166 | +0.145% |
| DF | 12.2680 | 12.3059 | +0.309% | 43.8018 → 43.5825 | -0.501% |

The initial bridge17 controls were less favorable; none were discarded or treated as equivalent speed evidence. In the short MTP pilot, HOLD prefill was +2.250% slower; the one confirmation is -0.800% on HOLD and +0.066% averaged across both prompts. Combining every MTP sample still leaves +0.700% prefill time overall (+1.236% on HOLD). DFlash short prefill is -0.008% overall after confirmation. These small samples do not establish a statistical guarantee or universally zero prefill cost.

**Decision:** retain the implementation and regression tests as a small, correct, composable optimization. Do not discard it for its gain size. Keep the existing default Static schedule; the tested combination is available locally with `MLXL3_SSD_DYNAMIC_READERS=1` and `MLXL3_SSD_COLD_READY=1`. The user requested no prefill slowdown; the residual short-MTP uncertainty prevents claiming that condition globally. AR and DFlash are favorable on the tested cases; no phase/mode-specific automatic policy was invented without measuring it. No additional repeated confirmation or new optimization is pending.

Follow-up audit found and fixed a stale cfg(test) condition that skipped the positive natural-Cold-wave assertion under Dynamic. Actual route/logit/state comparisons had still run, and the separate Cold component test required the mixed path. The corrected oracle passes A/C with 177/154 natural waves. `oracle-assertion.diff` is test-only; libtest SHA-256 is `d8fef182be1a49d55934eb6e79952a114ac7e557d9083ce847ade0baed593a24`. The original identity, new `identity-oracle.json`, and final `identity-confirm.json` preserve the unchanged release CLI and the test-only source revision. No speed change is attributed to the assertion.

Cumulative resource maxima from the same authorized anchor: **2.9375 MiB swap writes**, **10.3438 MiB swap reads**; limits were 32/256 MiB. Maximum child RSS 7532871680 B; maximum reported model lifetime footprint 7933499896 B. These are different metrics, not extra cache allocation. Closure records AC power, no remaining experiment executable, and verified source/binary hashes. All incidents remain in the per-command results; no reanchor.

An ad hoc closure snapshot initially mislabeled unrelated VM event counters as bytes. `closure.json` was corrected to retain only the actual page-based memory/swap fields; the original and diagnostic are retained in `closure-invalid-extra-units.json`. Supervisor guards and performance results were unaffected.

Verification limits: Kani/CBMC did not reach proof obligations; CrossHair unavailable; no remote CI for the local diff; no installed GUI/app update, other hardware or other checkpoint qualification. Existing local work was preserved, with the scoped final diff in `final-own-change.diff`.

## Diagnostic audit of actual scheduling

The MTP/HOLD native natural C-A profiles completed all six phases per arm. The existing `profile_flash.analyze` validated external tokens, complete internal parity, finite/full states, event nesting, job/byte accounting and all 48 layers. Stage retention is disabled after warmup. Results are in `diagnostic-analysis.json` and the two profile directories.

Across the three decode traces in each arm, routing hits/misses and read volumes are identical: 66,316 read jobs in 11,007 batches; 6,733 batches (61.2%) can benefit from Dynamic. Almost every eligible batch uses all four workers; there is no accidental serialization to one reader.

| Metric, eligible decode read batches | Static | Dynamic |
| --- | ---: | ---: |
| Mean requests concurrently in flight | 2.932 | 3.190 |
| Mean first-start to last-finish span | 1,085.04 µs | 1,045.42 µs |
| Mean spread of worker finish times | 559.10 µs | 362.67 µs |
| Mean total gap between jobs on the same workers | 3.020 µs | 3.497 µs |

Across 381 decode transitions, Cold wait wall time decreases by 0.208 ms/output and final finish/join wall time by 0.599 ms/output. Construction/submission/other batch work offsets about 0.123 ms, leaving 0.685 ms less Cold batch wall time per output. Mean first-wave rows remain nearly unchanged (2.418 → 2.415). These are nested native host wall intervals, **not exclusive GPU time and not an independently attributable CLI gain**. The mechanism works, but balancing the tail cannot remove all SSD service or MLX submission work.

Only 31.9% of prefill batches in this diagnostic are Dynamic-eligible. The diagnostic repeats HOLD while the CLI alternates MAIN/HOLD, and model reset intentionally retains the expert cache. Therefore this trace is evidence about actual scheduling and overhead, not an exact reproduction of the short CLI prefill regression. The one planned confirmation uses the original CLI prompt history. No engine adaptation defect was found beyond the corrected test-coverage gap; the small-gain retention decision and remaining short-MTP prefill uncertainty are recorded above.
