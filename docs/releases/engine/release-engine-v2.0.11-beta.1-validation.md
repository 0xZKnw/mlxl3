# MLXL3 Engine 2.0.11 beta 1 — validation

Engine source `a7241d8ba758da79ba2e2aff72a9f6993584b22d`. It is `origin/main` `c967e007`, plus the head layout commit `bdf4cc82` and the version commit. The archive was built with `scripts/build-engine.sh` from a clean checkout, and `runtime-info` reports no local modification. Release documentation and measurement records are separate commits.

## Archive and compatibility

**MLXL3-Engine-v2.0.11-beta.1-arm64.tar.gz** · 68,097,098 bytes · SHA-256 `dfd6e4470d3e4a581c2f7b2206f235a0004911f1367a0b93f698bdabcf81106e`.

The signed arm64 binary reports version 2.0.11, MLX, chat enabled, profile release and source revision `a7241d8ba758`. The manifest records release tag `engine-v2.0.11-beta.1`, minimum Desktop 1.4.0 and minimum macOS 26.2.0.

| Component | Bytes | SHA-256 |
|---|---:|---|
| libjaccl.dylib | 1,546,672 | `b6bdf1d1d472298217a0fb27893e15c35e78d1436f2bd023dc711ef84640b2b3` |
| libmlx.dylib | 21,815,104 | `40cb0b60beb0a1dde0e13647a644ece3625d6dd1a048ac846005db2c67c9fff2` |
| mlx.metallib | 182,351,120 | `dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed` |
| mlxl3 | 12,708,704 | `5b1051fcba1d90f8c28c5846fb181f4ebd11aa557723c470096fbfcfcf96be11` |

The three libraries are byte-identical to engine 2.0.10.

The archive was extracted into a new directory. The extracted files are exactly the manifest set plus `engine.json`, with matching sizes and hashes. All three Mach-O files passed `codesign --verify --strict` and `lipo -verify_arch arm64`.

The real Desktop `EngineRuntimeStore` installed the archive into a disposable store, resolved it, executed it and rejected it back to the bundled fallback. That check used a temporary copy of `tests/updater-check.swift` that reads the beta release tag from the environment. The first attempt failed while compiling Desktop with the default SDK 27.2 (missing `SwiftUIMacros` plugin with Command Line Tools). It passed with SDK 26.5, the same SDK the engine build script selects.

## Decode throughput of the packaged engines

**Setup:**

- **Machine:** Apple M5, 24 GiB, macOS 27.2, MLX 0.32.2.
- **Baseline:** the published 2.0.10 engine, archive `a8e4a9d5…`. It came from the Desktop updater store, was checked against its manifest and was run from a working copy.
- **Model:** Qwen3.6-35B-A3B EXL3 2.49 bpw with the same affine 4-bit MTP head.
- **Generation:** greedy, context 4096, prompt cache off.
- **Prompts:** the French prompt is 27 tokens and the code prompt 43 tokens.
- **Order:** each engine ran twice per prompt in BAAB order. Each pass had one 128-token warmup and two measured 128-token replies.
- **Conditions:** AC power, battery charged, no thermal warning, swap unchanged at 1.9 GB, no concurrent build or GPU job, on 2026-11-15 between 16:32 and 16:37.

| Mode | French, 2.0.10 → 2.0.11b1 tok/s | Change | Code, 2.0.10 → 2.0.11b1 tok/s | Change |
|---|---:|---:|---:|---:|
| AR | 51.26 → 53.91 | +5.18% | 50.16 → 53.91 | +7.46% |
| MTP1 | 72.10 → 72.56 | +0.64% | 70.77 → 71.25 | +0.67% |
| MTP2 | 64.95 → 69.32 | +6.73% | 64.20 → 69.43 | +8.14% |
| MTP3 | 65.96 → 71.10 | +7.79% | 63.62 → 68.63 | +7.88% |

| Mode, French / code | Prefill change | TTFT change | End-to-end latency change |
|---|---:|---:|---:|
| AR | +5.11% / −0.15% | −4.83% / +0.01% | −4.91% / −6.40% |
| MTP1 | −1.97% / −0.11% | +1.88% / +0.15% | −0.43% / −0.58% |
| MTP2 | −2.74% / −3.33% | +2.70% / +3.30% | −5.68% / −6.52% |
| MTP3 | +1.96% / +1.77% | −1.87% / −1.67% | −6.86% / −6.77% |

Per-pass decode medians, in run order:

| Mode, prompt | Pass medians (tok/s) |
|---|---|
| AR, French | B 54.43, A 51.25, A 51.27, B 53.40 |
| AR, code | B 54.25, A 50.05, A 50.28, B 53.56 |
| MTP1, French | B 72.34, A 71.17, A 73.03, B 72.77 |
| MTP1, code | B 70.60, A 69.67, A 71.87, B 71.89 |
| MTP2, French | B 68.37, A 64.58, A 65.32, B 70.27 |
| MTP2, code | B 68.85, A 64.23, A 64.17, B 70.00 |
| MTP3, French | B 70.79, A 65.84, A 66.09, B 71.40 |
| MTP3, code | B 68.04, A 63.02, A 64.22, B 69.22 |

The MTP1 passes overlap, so no MTP1 change is established.

Prefill and TTFT on these short prompts have no consistent sign. Each campaign validated token IDs, text and history between the two engines. Acceptance was identical in both:

| Depth | French accepted / proposed | Code accepted / proposed |
|---|---:|---:|
| MTP1 | 58 / 69 | 57 / 70 |
| MTP2 | 72 / 108 | 73 / 107 |
| MTP3 | 81 / 138 | 80 / 139 |

## Isolated measurements

The following results have their own baselines. Do not add them to the archive comparison or to each other.

- **Head layout, same binary:** `MLXL3_Q36_HEAD_LANE=0` against `1`, same workload as above, BAAB then ABBA, one setting change and nothing else.

  | Depth | French BAAB / ABBA | Code BAAB / ABBA |
  |---|---:|---:|
  | MTP2 | +2.95% / +6.09% | +8.06% / +4.04% |
  | MTP3 | +2.93% / +4.82% | +3.55% / +4.50% |

  In every cell, each pass with the new layout was faster than each pass without it. The MTP1 control (BAAB) moved −0.58% (French) and +0.46% (code); MTP1 never uses the path. Prefill and TTFT moved by up to 5.4% in either direction without a consistent sign, as in the MTP1 control. The prefill applies the head to one row, which cannot select the new layout.
- **Earlier commits, measured on their own:**
  - The default layer pipeline was measured on battery against the previous default: AR +7.7% to +8.4% in both orders, DFlash2 mode 3 +2.5% to +6.4%, MTP2 neutral.
  - The parallel route plan gained +3.65% and +4.02% prefill throughput in the two orders on a 1,097-token prompt.
  - The smaller final prefill layer gained +1.07% and +1.11% on the same prompt.

## Numerical and functional verification

On the extracted archive:

- The CLI `forward --tokens 1,53,106,159` matched four saved 248,320-logit F16 vectors bit for bit. All values are finite.
- `scripts/check-mtp-bridge.py` reproduced the AR/MTP1 reference token hashes for budgets 1, 2, 3, 17, 64 and 128 (128 → `04d09d6803d2329b`).
- `scripts/check-mtp-depths.py --tune` reported depth parity, prefix parity, cancellation recovery and a verified tuner. Its 35 completions (request, token hash, count) and the tuner hashes matched the release build of `bdf4cc82` with the head layout on.
- `scripts/check_dflash2.py --skip-tune` reported complete with parity, and produced 33 completions identical to that build.
- `scripts/smoke-release.py` ran two real text turns through the Desktop NDJSON protocol. The second turn recalled the supplied code.

On the head layout source, before packaging:

- **Components:** 18/18 exact on the real `lm_head`: four seeds at widths 1–6, new layout against the stock and serial paths. Only widths 3 and 4 activated the new layout. A permuted gather index was detected at both widths.
- **Source mutation:** removing the input permutation was detected, and the restored source rebuilt to an identical test binary.
- **Production verification:** `verify_mtp` matched the serial decode loop in 24/24 cases (logits, MTP hidden state and 80 cache/state arrays), on three prompts of 27, 43 and 1,097 tokens, at widths 1–4 with both settings.
- **Snapshots and pipeline:** the checkpoint snapshot test, the layer pipeline test and the qualified verification test passed.
- **Release binary of that source:** CLI logits were bit-identical with the setting unset, `1` and `0`. The values `2`, empty, `true` and ` 1` were refused at load. Bridge hashes matched, and MTP depths and DFlash2 were identical with the setting on and off.

## Local checks and verification bounds

- **Format and lint:** Cargo fmt and strict Clippy passed for `mlx,chat` all targets, `chat`, `direct-metal,chat` and `hybrid-dispatch`. The first Clippy attempt without MLX failed on an unused function and was fixed before these results.
- **Test suites:**
  - The MLX/chat library suite passed 467 tests with 250 ignored. The physical GPU tests above were selected explicitly and run alone.
  - The CPU suites passed 548 tests (`chat`) and 550 (`hybrid-dispatch`), each with 10 ignored. In both, the library binary accounts for 439 passed and 10 ignored. The totals also count the other test binaries and the single-test child processes started by the suites.
  - The MTP tune binary tests passed 3 with 1 ignored.
  - The MLX site inventory matched 322 files.
  - Python checks passed: the Kani manifest (5), native coverage (6) and ruff.
  - The Linux-only Kani CI tests (53) were skipped on macOS and are not counted as run.
- **Kani 0.68 (CBMC):**
  - `contracts::q36_head_lane_setting_is_strict`: 0 of 229 checks failed, 6/6 covers reached, unwinding assertions passed. Seven unreachable checks are inside the standard library.
  - `contracts::q36_head_lane_layout_admission_and_grid`: 0 of 134 checks failed, 9/9 covers reached.

  This is bounded model checking of the CPU admission contract. It does not cover Metal, MLX, the allocator, concurrency or model quality.
- **Test source:** the tests and proofs above ran on the head layout source before the version commit, which changes only the version string. The extracted archive passed the real-path checks listed earlier.
- **Not verified:** hosted CI status is recorded separately after publication. Apple M1–M4, macOS 26.2, other models for the M5-only paths, long-prompt prefill of the packaged archive and quality beyond token parity were not measured in this campaign. The archive comparison used one order on AC power.

## Reproducibility

The archive comparison ran `benchmarks/compare_native.py MODEL --baseline 2.0.10/mlxl3 --candidate 2.0.11b1/mlxl3 --mtp-head HEAD --prompt-file code-prompt.txt --tokens 128 --warmup-tokens 128 --repeats 2 --settle-seconds 2 --order BAAB --mtp-depth 0|1|2|3`.

The head layout comparison used the same command with one binary and `--baseline-env MLXL3_Q36_HEAD_LANE=0 --candidate-env MLXL3_Q36_HEAD_LANE=1`. It ran BAAB and ABBA for depths 2 and 3, and BAAB for depth 1.

Scripts, per-pass results and logs are in [docs/measurements/rel-engine-2.0.11-beta1-20261115](../../measurements/rel-engine-2.0.11-beta1-20261115/) and [docs/measurements/q36-head-lane-p24-20261115](../../measurements/q36-head-lane-p24-20261115/). The raw CLI logit dumps (about 6 MB each) are kept locally and summarized by their comparison logs.
