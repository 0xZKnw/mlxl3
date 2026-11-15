# MLXL3 Engine 2.0.11 beta 2 — validation

Engine source `3fe7b08972f59c51969962aaa536b48ea6401e15`. It is beta 1 `c42e4c95` plus four commits:

- `cbf0bebc`: post-norm MTP wiring;
- `24d6ee42`: a `compare_native` option for the candidate depth;
- `f568dd08`: confidence-based verify truncation (EVICT), with post-norm and EVICT enabled by default;
- `3fe7b089`: the reviewed MLX source inventory update.

The archive was built with `scripts/build-engine.sh` from a clean detached checkout of that commit, and `runtime-info` reports no local modification. The Cargo version stays 2.0.11; the release tag `engine-v2.0.11-beta.2` sets the release identity. Release documentation and measurement records are a separate commit.

## Archive and compatibility

**MLXL3-Engine-v2.0.11-beta.2-arm64.tar.gz** · 68,100,591 bytes · SHA-256 `143a5bfc3eb042132b7a77c550f8a0ec7ab7a82c19be6d99ca5df7f7b580a553`.

The signed arm64 binary reports version 2.0.11, MLX, chat enabled, profile release and source revision `3fe7b08972f5`. The manifest records release tag `engine-v2.0.11-beta.2`, minimum Desktop 1.4.0 and minimum macOS 26.2.0.

| Component | Bytes | SHA-256 |
|---|---:|---|
| libjaccl.dylib | 1,546,672 | `b6bdf1d1d472298217a0fb27893e15c35e78d1436f2bd023dc711ef84640b2b3` |
| libmlx.dylib | 21,815,104 | `40cb0b60beb0a1dde0e13647a644ece3625d6dd1a048ac846005db2c67c9fff2` |
| mlx.metallib | 182,351,120 | `dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed` |
| mlxl3 | 12,727,680 | `c68e04883bba184874b98d4eb4eb6fc4e19c5a7cea85c7ff715c52bfaca3b94d` |

The three libraries are byte-identical to beta 1 and 2.0.10.

The archive was extracted into a new directory. The extracted files are exactly the manifest set plus `engine.json`, with matching sizes and hashes. All three Mach-O files passed `codesign --verify --strict` and `lipo -verify_arch arm64`.

The real Desktop `EngineRuntimeStore` installed the archive into a disposable store, resolved it, executed it and rejected it back to the bundled fallback. As for beta 1, the check used a temporary copy of `tests/updater-check.swift` that reads the beta release tag from the environment, compiled with SDK 26.5.

## Decode throughput of the packaged engines

**Setup:**

- **Machine:** Apple M5, 24 GiB, macOS 27.2, MLX 0.32.2.
- **Baseline:** the published beta 1 engine, extracted from its archive (`mlxl3` `5b1051fc…`).
- **Model:** Qwen3.6-35B-A3B EXL3 2.49 bpw with the same affine 4-bit MTP head.
- **Generation:** greedy, context 4096, prompt cache off. No MLXL3 tuning variable was set, so each engine ran its own defaults.
- **Prompts:** the French prompt is 27 tokens and the code prompt 43 tokens.
- **Order:** each engine ran twice per prompt in BAAB order. Each pass had one 128-token warmup and two measured 128-token replies.
- **Conditions:** AC power, battery charged, no thermal warning, swap about 3.6 GB before and after, on 2026-11-15 from 22:09 to 22:14.
  - The campaign started after the machine had been idle for 85 s.
  - A contention sampler saw a Swift compiler from another session for about 21 s during the MTP1 campaign (22:11:14–22:11:35). Other campaigns had no recorded contention.

| Mode | French, beta 1 → beta 2 tok/s | Change | Code, beta 1 → beta 2 tok/s | Change |
|---|---:|---:|---:|---:|
| AR | 58.85 → 58.94 | +0.15% | 58.95 → 58.97 | +0.05% |
| MTP1 | 79.59 → 81.67 | +2.62% | 76.72 → 78.19 | +1.91% |
| MTP2 | 73.96 → 76.11 | +2.91% | 73.41 → 79.31 | +8.04% |
| MTP3 | 75.90 → 89.63 | +18.08% | 72.99 → 81.59 | +11.78% |

| Mode, French / code | Prefill change | TTFT change | End-to-end latency change |
|---|---:|---:|---:|
| AR | −0.11% / +0.43% | +0.10% / −0.44% | −0.13% / −0.07% |
| MTP1 | +1.49% / +1.74% | −1.47% / −1.73% | −2.48% / −1.87% |
| MTP2 | −3.14% / −0.01% | +3.78% / +0.02% | −0.26% / −6.69% |
| MTP3 | −1.61% / +0.18% | +1.73% / −0.17% | −14.09% / −9.50% |

Per-pass decode medians, in run order:

| Mode, prompt | Pass medians (tok/s) |
|---|---|
| AR, French | B 59.30, A 59.00, A 58.70, B 58.58 |
| AR, code | B 58.59, A 58.82, A 59.07, B 59.36 |
| MTP1, French | B 81.35, A 78.81, A 80.37, B 82.00 |
| MTP1, code | B 77.54, A 75.95, A 77.50, B 78.84 |
| MTP2, French | B 84.23, A 73.89, A 74.03, B 68.00 |
| MTP2, code | B 79.65, A 73.49, A 73.33, B 78.97 |
| MTP3, French | B 88.63, A 75.91, A 75.89, B 90.63 |
| MTP3, code | B 80.98, A 72.87, A 73.11, B 82.20 |

- **MTP3:** in both prompts, each beta 2 pass was faster than each beta 1 pass.
- **MTP2, code:** each beta 2 pass was faster than each beta 1 pass.
- **MTP2, French:** the second beta 2 pass (68.00) was slower than both beta 1 passes, while the first was the fastest pass (84.23). The median gain is therefore not established from this campaign alone.
- **MTP1 and AR:** the passes overlap, so no change is established.
- **Prefill and TTFT:** on these short prompts, they have no consistent sign.

Each campaign validated token IDs, text and history between the two engines. Acceptance, as accepted / proposed / verify cycles:

| Depth | French, beta 1 → beta 2 | Code, beta 1 → beta 2 |
|---|---|---|
| MTP1 | 58 / 69 / 69 → 59 / 67 / 67 | 57 / 70 / 70 → 58 / 69 / 69 |
| MTP2 | 72 / 108 / 54 → 73 / 84 / 54 | 73 / 107 / 54 → 72 / 88 / 55 |
| MTP3 | 81 / 138 / 46 → 82 / 94 / 45 | 80 / 139 / 47 → 78 / 100 / 49 |

At MTP2 and MTP3, beta 2 verifies 18–32% fewer draft tokens per reply at about the same number of cycles. This reduction comes from EVICT. At MTP1 EVICT never changes the width; the small MTP1 differences come from the post-norm draft.

**Discarded first campaign.** An earlier run of the same script (22:02–22:07) overlapped a Swift build from another session for its whole duration. It is kept in `speed-attempt1-contended/` and is not used as the result. For reference, it measured AR −0.07% / −0.70%, MTP1 +1.79% / +0.57%, MTP2 +12.56% / +7.59% and MTP3 +15.57% / +9.35% (French / code), with identical tokens.

## Isolated trials

These trials have their own baselines. Do not add them to the archive comparison or to each other. Their records are in `opti.md` (entries Q36-MTP-DIAG-T0, Q36-MTP-POSTNORM-T1B, Q36-MTP-EVICT-T3A, Q36-MTP-EVICT-T3B and Q36-MTP-DEFAULT-T3C).

- **T1b, post-norm wiring:** measured with the setting on and off in one binary. It raised accepted tokens per cycle at MTP2 and MTP3 and requalified the compact draft vocabulary at those depths.
- **T3a, EVICT calibration:** fitted the survival table and the draft and verify costs from recorded post-norm cycles, and passed its offline replay criterion. It is a simulation, not a speed result.
- **T3b, EVICT:** measured on the post-norm build against full-width verification. Decode moved +1.1% to +9.1% in 8 of 8 cells, but the pre-registered criterion was missed, so the trial is recorded as not conclusive. The option was kept by decision and is now the default. A deliberate policy mutation that ignored confidence was detected by the policy comparator.
- **T2, depth choice:** interrupted at the user's request after 11 of 18 campaigns. It is not conclusive, and nothing from it is shipped.

## Numerical and functional verification

On the extracted archive:

- The CLI `forward --tokens 1,53,106,159` matched four saved 248,320-logit F16 vectors bit for bit. All values are finite.
- `scripts/check-mtp-bridge.py` reproduced the AR/MTP1 reference token hashes for budgets 1, 2, 3, 17, 64 and 128.
- `scripts/check-mtp-depths.py --tune` reported depth parity, prefix parity, cancellation recovery and a verified tuner. Its 35 completions (request, token hash, count) and the tuner hashes matched the beta 1 release build.
  - Tune selected MTP3 (AR 48.20, MTP1 69.14, MTP2 73.13, MTP3 79.08 tok/s). This qualification was not isolated from other processes, so these rates are indicative only.
  - The default Tune key ends with `lookup=0:mtppostnorm-v1=1:mtpevict-v1=1:q36headlane-v1=1:lanelayout-v1=1:smallm-v2=1`.
- `scripts/check_dflash2.py --skip-tune` reported complete with parity, and produced 33 completions identical to the beta 1 release build.
- `scripts/smoke-release.py` ran two real text turns through the Desktop NDJSON protocol. The second turn recalled the supplied code.

On the release source, with a physical test binary built from `3fe7b089` (`5b3b47d3…`):

- **Defaults:** with both variables unset, the MTP cycle diagnostic over 4 prompts × depths 1–3 (12 cases) matched the earlier explicit `MLXL3_MTP_POST_NORM=1 MLXL3_MTP_EVICT=1` run exactly: tokens, cycles, accept histograms, draft levels and all 1,366 EVICT records.
  - The first comparison reported a trace difference in all 12 cases because the comparator also compared the wall-clock `uptime_ns` field of trace marks. The comparator was corrected to ignore only that field (a missing field still fails) and replayed on the same diagnostic file; it passed. Both logs are kept.
- **Policy oracle:** an independent float64 recomputation of the pre-registered policy from the host top-1 log-probabilities agreed with every recorded width. The device top-1 bits equalled the host values, and MTP1 cycles were unchanged.
- **Opt-outs:** with both variables set to `0`, the diagnostic matched the previous pre-norm, full-width reference exactly. With only `MLXL3_MTP_EVICT=0`, it matched the post-norm trial exactly.
- **State tests:** the recursive session, lookup and cache tests passed with the new defaults and kept exact states.
- **Rejection:** real model loads rejected `MLXL3_MTP_EVICT=2`, `MLXL3_MTP_EVICT=true` and `MLXL3_MTP_POST_NORM=2`.

## Local checks and verification bounds

- **Source:** the suites below ran on `f568dd08` in a worktree that also held other sessions' uncommitted Swift and Markdown edits. No Rust, C++ or Metal file differed from the commit. The check script reported `worktree changed during checks` for that reason and is recorded as such. The next commit, `3fe7b089`, changes only `docs/native-coverage/mlx-sites.json`.
- **Format and lint:** Cargo fmt and strict Clippy passed for `mlx,chat` all targets, no default features, `chat`, `hybrid-dispatch` and `direct-metal,chat`.
- **Test suites:**
  - MLX/chat library: 490 passed, 251 ignored (474 in the library binary, the rest in single-test child processes). The binary tests passed 51 with 9 ignored.
  - CPU, no default features: 448 passed, 9 ignored.
  - CPU, `chat`: 554 passed, 10 ignored. CPU, `hybrid-dispatch`: 556 passed, 10 ignored.
  - The MLX site inventory was regenerated, reviewed and committed as `3fe7b089`; it then matched 322 files, and its tests passed 6/6.
  - Python checks passed: `compare_native` (71), the Kani manifest (5) and ruff.
  - The Linux-only Kani CI tests (53) were skipped on macOS and are not counted as run.
- **Kani 0.68 (CBMC 6.11), `cargo kani --lib --no-default-features --harness …`:**

  | Harness | Checks failed | Covers reached | Unreachable checks |
  |---|---:|---:|---:|
  | `mtp_post_norm_setting_is_strict` | 0 of 229 | 6/6 | 7, in the standard library |
  | `mtp_evict_setting_is_strict` | 0 of 229 | 5/5 | 7, in the standard library |
  | `evict_band_is_total_and_monotone` | 0 of 134 | 4/4 | 1, in `core` |
  | `evict_width_is_bounded_and_rejects_invalid_input` | 0 of 216 | 5/5 | 1, in `core` |

  The parser harnesses cover symbolic strings of up to four bytes. The band harness covers every `f32` bit pattern. The width harness covers every band combination for widths up to 4. This is bounded model checking of CPU contracts. It does not cover Metal, MLX, the allocator, concurrency or model quality.
- **Not verified:** hosted CI status is recorded separately after publication. Apple M1–M4, macOS 26.2, MTP models other than the 35B, long-prompt decode and quality beyond token parity were not measured in this campaign. The archive comparison used one order on AC power, and the EVICT calibration applies to this model and device.

## Reproducibility

The archive comparison ran `benchmarks/compare_native.py MODEL --baseline beta1/mlxl3 --candidate beta2/mlxl3 --mtp-head HEAD --prompt-file code-prompt.txt --tokens 128 --warmup-tokens 128 --repeats 2 --settle-seconds 2 --order BAAB --mtp-depth 0|1|2|3`, through `rel-speed.sh` started by `speed-guard.sh`.

Scripts, per-pass results and logs are in [docs/measurements/rel-engine-2.0.11-beta2-20261115](../../measurements/rel-engine-2.0.11-beta2-20261115/) and [docs/measurements/q36-mtp-default-t3c-20261115](../../measurements/q36-mtp-default-t3c-20261115/), with the earlier trials in the neighbouring `q36-mtp-*-20261115` directories. Files larger than 1 MB (diagnostic JSON reports, the CLI logit log) are kept locally and summarized by their comparison logs.
