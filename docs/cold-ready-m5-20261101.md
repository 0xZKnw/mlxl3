# Flash-Next Cold expert overlap on M5

The native engine now starts the first ready Cold expert wave while the remaining SSD reads finish, and prepares the second wave lazily before joining those reads. The final merge restores canonical Top10 occurrence order, including duplicates, and retains the existing numerical operations. The cache stays at 3 GiB. Hot/W1 consumers, file descriptors and bank owners remain retained until safe completion; error paths drain and invalidate pending reservations.

`MLXL3_SSD_COLD_READY` now defaults to `1`; `0` selects the stock path. The optimization is limited to qualified B1/B2 paths with Hot available, static readers, depth zero, no prefetcher and no grouped/weighted epilogue. Other paths keep their existing fallback. No new Metal kernel or second expert bank is introduced. Installed Desktop applications have not been replaced.

## Measured signals

M5 standard, 24 GB; MLX 0.32.2; Qwen3.8 Flash-Next EXL3 2.05 bpw `65c89531`; 3 GiB cache, four readers. MAIN/HOLD use 87/82 prompt tokens and 128 generated tokens. Each process warms both prompts, then measures each once. Rates aggregate token counts divided by elapsed time, not averages of percentages.

| Mode | Stock decode | Candidate decode | Decode change | Prefill rate change | Full request time change |
|---|---:|---:|---:|---:|---:|
| AR, final cleaned binary | 11.7196 tok/s | 11.8839 tok/s | +1.40% | +1.74% | −1.52% |
| MTP, qualified U prototype | 11.5024 tok/s | 11.6639 tok/s | +1.40% | +0.23% | −0.90% |
| DFlash2, qualified U prototype | 12.1618 tok/s | 12.4881 tok/s | +2.68% | +3.13% | −2.79% |

These are modest measured signals, not guaranteed speedups. Final AR used S/U/U/S with the same binary. Its stock controls drifted −3.59%/−1.93% in decode. MTP/DFlash used S/P/U/U/P/S (P is the original first-wave prototype). DFlash stock controls drifted +2.06%/+1.52%. MTP HOLD prefill was 1.37% slower on average while its total request duration was essentially unchanged (+0.004%). Both AR prompts and both DFlash prompts improved in aggregate decode, prefill and total duration. No percentages from separate campaigns are added.

MTP/DFlash throughput was measured before removing the unused ablation branch and changing the default. The executed U algorithm is retained; final numerical suites were rerun on the cleaned binary. Those earlier timings are not relabelled as measurements of the final binary.

Final AR binary SHA-256: `09db0413ee1de40d738c891b4f30fd94e948406433e64b8186fc735c8652d19f`.

## Audit and discarded variant

Deferring W1 submission when SSD reads had already completed was not automatically beneficial: early GPU execution can still overlap later CPU graph preparation. Separate U/D ablations were implemented and qualified before selection. U prepares W2 early; D defers W1. Considering decode and prefill together, U had the slightly lower total request duration. D was removed from the final runtime. The initial combined P2 result and both ablations remain recorded.

The first final AR campaign was contaminated by detached Git maintenance triggered by our fetch while preparing publication. The maintenance and campaign were stopped, all partial results preserved, and only AR was repeated after confirming no Git/model process remained. MTP/DFlash had finished before that fetch. The contaminated 6.21 tok/s stock sample is not used to claim a speedup.

## Verification

- Final `cargo test --locked --features chat`: 269 passed, 10 ignored across the nonempty test targets. Two zero-test targets are not GPU validation.
- Final MLX release libtest: 205 passed, 191 ignored. Seven ignored physical GPU selections were then explicitly run alone and passed: component rows/duplicates/fallbacks, completion/fenced/error paths, failure recovery, independent full-model oracle, paired verification, MTP and DFlash states.
- DFlash prefix validation covers 36 prefixes, widths 1–8 and 64,000 captured values, with byte-exact logits/states/captures on checked paths. Finite nonempty shapes are checked. Full benchmark bridge outputs and acceptance counters match independent goldens; benchmark text equality alone is not numerical parity.
- The real component loading path without COLD_READY fails on the prior default-OFF binary and passes on the final default-ON binary. The earlier mutation suppressing W2 preparation is detected.
- Publication Python suite: `python -m pytest -q tests/test_flash_*.py tests/test_decode_capture.py`: 1,266 passed.
- `cargo fmt --check`, strict MLX/chat all-target Clippy, release MLX/chat library/CLI builds, targeted Ruff/py_compile, YAML and diff checks pass. Controller self-tests exercise valid schedules, wrong flags/hashes/order, bounded swap and durable failure reporting.
- Kani was attempted on the actual Rust test entry with symbolic path selection, no MLX stubs. The final attempt stopped at the preregistered 512 MiB process-group RSS cap before a verdict (3.57 seconds). No source proof is claimed. Earlier C++ CBMC attempts fail while parsing libc++; CrossHair is unavailable. CPU checks do not prove MLX, Metal, the allocator or concurrency.

The authorized fixed anchor includes the earlier swap incident. The final cumulative ceilings were 64 MiB new writes and 1 GiB reads; no reanchoring was performed. Full conditions and failures remain in `opti.md`.

## Evidence and publication scope

- [Final AR results](measurements/post-p08-cold-ready-20261101/p2/integration/r2/results.json)
- [MTP/DFlash results](measurements/post-p08-cold-ready-20261101/p2/finalist/results.json)
- [Ablation results](measurements/post-p08-cold-ready-20261101/p2/ablation/AR-result.json)
- [Adaptation audit](measurements/post-p08-cold-ready-20261101/p2/ablation/adaptation-audit.md)
- [Final Kani receipt](measurements/post-p08-cold-ready-20261101/p2/integration/kani-result.json)
- [Publication CPU tests](measurements/post-p08-cold-ready-20261101/publication/cpu-tests.log)
- [Publication Python tests](measurements/post-p08-cold-ready-20261101/publication/python-tests.log)

The commit also preserves this worktree's previously implemented profiling and independently gated experiments, with their historical decisions in the journal. Their presence does not mean they are enabled or deliver additional gains. Raw multi-gigabyte Xcode traces and large local telemetry are not published in Git; they remain on disk, listed in the [local-only inventory](measurements/post-p08-cold-ready-20261101/publication/local-only-artifacts.json). Absolute source/binary identities describe the measurement workspace; they are not portable replay paths. Remote CI and later PR integration are separate from the local results above.
