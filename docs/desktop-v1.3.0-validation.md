# MTP1/2/3 and Tune MTP · 1.3.0 validation

Implementation and checks on 2026-10-05, Apple M5/24 GB, MLX 0.32.2, Swift 6/SDK 26.5, Rust and Kani 0.68.0. Baseline source `d0160f43a3c0379ab3c2bfa0d289b62b4ab885f8`. Checkpoints: local Qwen3.6-35B-A3B EXL3 2.49 bpw and MLX affine4/group64 MTP head. No weight download or change. App and engine published from clean source `3733e9df7cbc0298e81e778865e3abd8ac1ebd4b`; delivery evidence follows below.

## Implementation

[MTPLX](https://github.com/youssofal/MTPLX/tree/9882703f3105363ddc37eca9f97aa09a1d387112) provides the primary architectural reference: Qwen head recursion feeds back the residual before output norm, proposals form a lazy device chain, the target verifies the block, and tuning warms candidates before timing. MLXL3 retains its own EXL3/GDN/Metal execution and normalized greedy selector. Apache-2.0 attribution is recorded in `THIRD_PARTY_NOTICES.md`.

Depths are 1..3, bounded by remaining context/output minus the final target token. Verification commits exactly anchor plus accepted prefix; every delivered draft or correction is the target's choice. Discard approximate draft K/V, restore the first exact pair, append only accepted real target pairs with single-row arithmetic and evaluate final K/V once. The initially attempted batched repair changed FP16 K/V rounding; it was rejected, retained in the journal, and replaced by this exact version.

The resident tuner uses no tools or chat history. Warmup32 for each of four modes, then two prompts ×96 tokens per mode, forward and reverse candidate order, fresh prefill for each sample. Decode excludes load, warmup and prefill; first token excluded from token denominator. Candidate token IDs must exactly match baseline; samples below32tokens, zero acceptance or invalid rates are ineligible. Integer milli-tok/s ranking prefers shallower exact ties and baseline unless gain exceeds3%. A result is indicative for these two short prompts, not a universal fastest depth.

Selection persists by canonical model/head path, file size/modification fingerprint, hardware, context, runtime version/commit/profile and MLX version. Manual depth override also persists. Cancellation, malformed/obsolete results, changed configuration or engine exit preserve the previous selection; benchmark output is never appended to chat. Older engine capability defaults to depth1, with tuning disabled and an upgrade explanation.

## Verification executed

- `cargo fmt --all -- --check`, `cargo clippy --locked --features mlx,chat --all-targets -- -D warnings`, `cargo test --locked --features mlx,chat`, release build: pass. MLX root `.venv/lib/python3.12/site-packages/mlx`, deployment target26.2. Ordinary Rust suites36+7+14 passed; physical GPU tests remain explicit `--ignored` runs. [Full suite](measurements/mtp-04/rust-full.log), [Clippy](measurements/mtp-04/clippy-final.log).
- `cargo kani --lib --no-default-features --output-format terse`:29 successful harnesses,0 failures. New bounded model checks cover all usize depths/context/output and u32 proposal/target IDs at depths≤3, retained target prefix and budget bounds; all Option<u64> candidate scores and overflow-safe/tie-aware winner; parity/acceptance filter at fixed finite64tok/s. Loops unwind6, candidate array4 and draft array3. Covers reach full acceptance, rejection, partial acceptance, zero-width, baseline/depth3 winners and collapsed acceptance. Full floating-point score calculation, MLX/Metal/FFI/concurrency and Swift are **not formally proved**. Unsupported foreign/caller-location constructs reported by Kani are not reachable from these harnesses. [All results](measurements/mtp-04/kani-all.log). A copied-source zero-acceptance mutant fails the regression; production code was never mutated. [Negative check](measurements/mtp-04/mutant-rejected.log).
- `cargo test --release --features mlx,chat --lib mtp_prefill_verification_and_rollback_match_target -- --ignored --nocapture`: target logits/residual and80 recurrent/KV states bit-exact against serial decoding for prefix lengths1/23/24/129/256, verification widths2..4 and **every retained prefix**; malformed IDs, empty/overlong blocks and full context rejected. [Result](measurements/mtp-04/gpu-rollback.log).
- `cargo test --release --features mlx,chat --lib mtp_recursive_sessions_match_greedy_and_exact_caches -- --ignored --nocapture`:24 blocks across D1/D2/D3, target tokens/states/residual and head K/V bit-exact versus independently stepped target and serial cache repair. Zero output budget rejected without consuming pending tokens. [Final result](measurements/mtp-04/gpu-sessions-final.log).
- `.venv/bin/python scripts/check-mtp-reference.py TARGET HEAD FIXTURE --recursive`, then `MLXL3_MTP_REFERENCE=FIXTURE cargo test --release --features mlx,chat --test runtime_gpu native_mtp_head_matches_independent_mlx_reference -- --ignored --exact --nocapture`: independent official MLX-LM DecoderLayer, three recursive pre-norm feedback steps, all raw output bits match, max absolute error0. [Python](measurements/mtp-04/reference-python.log), [Rust](measurements/mtp-04/reference-rust.log).
- `python3 scripts/check-mtp-depths.py ENGINE TARGET HEAD --tune`: real production JSON-lines bridge, budgets1/2/3/4/17/64 at all depths, target token hashes/text/history/count parity; invalid depths, cancellation at prefill/delta and recovery, cache reuse across D2→D3 and conversation isolation, cancelled tuner and recovery, full tuner and winner generation. [Events](measurements/mtp-04/bridge.jsonl), stderr empty.
- Swift6 warnings-as-errors compile of all production sources and `tests/studio-hardening-check.swift`, actual subprocess fixture: D2 auto selection reaches next generation, restart/manualD3/baseline persistence, cancelled/ejected tune, malformed/error/collapsed acceptance, engine/head fingerprint invalidation and old engine. Prior context/download/library/MCP/lifecycle checks also pass. [Result](measurements/mtp-04/swift-final2-test.log).
- Actual `MTPTuning.swift` in `tests/mtp-tuning-check.swift`:64 rate grids in forward/reverse order, ties/noise boundary, bad hashes/counts/rates, zero acceptance, duplicate/missing rows, safe float-to-integer boundary and JSON roundtrip. [Result](measurements/mtp-04/tuning-test.log). Test harness compilation mistakes were corrected before passing; logs/journal retain the failures. Swift checks are finite tests, not deductive proof.

## Initial MTP-04 pilot

| Mode | Decode tok/s | Accepted / proposed | Eligible |
|---|---:|---:|---|
| Baseline | 51.037 | — | yes |
| MTP1 | 63.443 | 89 / 99 | yes |
| MTP2 | 63.522 | 117 / 142 | yes, selected |
| MTP3 | 60.042 | 129 / 176 | yes |

Both prompts yielded identical token hashes across all modes. [Raw result](measurements/mtp-04/tuning-result.json). Battery100% discharging, no thermal/performance warning in `pmset`, swap3060.69 MiB used, other apps open, temperature not measured. No concurrent build/prover during this tuning. MTP2 exceeds MTP1 by only0.124%; this **does not establish a reproducible advantage** of MTP2 over MTP1. The subsequent MTP-05 trials below are separately registered; this pilot remains historical evidence.

## MTP-05 kernel changes and final comparison

MTPLX's affine/BF16 verify kernels do not directly match our EXL3/FP16 target. The retained adaptations are opaque device routes (public external gathers still validate IDs), weight sharing for two/four-row EXL3 batches, a 24-accumulator three-row tile only for large projections, and the hierarchical SIMD top-8 structure from `qwen_row_owned_router.py`. Scores retain ascending FP16 addition/division and the legacy monotone-key/index tie order. Other router geometries retain the original kernel. Pairing three rows with a padded fourth was **rejected**: MTP2 regressed4.7–5.9% in adjacent comparisons. A NaN-sign mismatch in the SIMD prototype was caught and corrected before activation; failed logs are retained.

Four sequential production-tuner runs, reference/candidate/candidate/reference, two96-token prompts and32 warmup tokens per mode, no simultaneous build/prover:

| Mode | Reference median tok/s | Candidate median tok/s | Observed change |
|---|---:|---:|---:|
| Ordinary | 49.156 | 50.434 | +2.60% |
| MTP1 | 61.917 | 65.603 | +5.95% |
| MTP2 | 62.623 | 63.715 | +1.74% |
| MTP3 | 58.734 | 62.605 | +6.59% |

Every token hash and acceptance count agrees:89/99,117/142,129/176 for depths1/2/3. MTP1 wins both candidate trials. MTP2's small increase is not established beyond machine variation, and greater depth is not always faster. Ordinary decode also changes with the new router, so normalization against it mixes optimization and machine drift; it is diagnostic, not causal isolation. Battery-powered M5, other apps open, no `pmset` thermal/performance warning; temperatures not measured. Exact conditions accompany each run. [Raw runs](measurements/mtp-05/final-matrix.log), [summary/limits](measurements/mtp-05/summary.json), full negative/positive trial history in `opti.md`.

Final checks for this source:

- `cargo fmt --all -- --check`, `cargo clippy --locked --all-targets --features mlx,chat -- -D warnings`, `cargo test --locked --features mlx,chat`:37 library,7 CLI,14 contract tests passed; physical tests explicitly ignored in the ordinary suite. [Results](measurements/mtp-05/rust-final.log).
- GPU `routed_gather_matches_checked_and_rejects_wrong_experts`: device/public gather equality, mismatched expert count/dtype, out-of-range public IDs and invalid routing rejected. [Result](measurements/mtp-05/gather-test-fixed.log).
- `MLXL3_DISABLE_TENSOR_OPS=1 cargo test --release --features mlx,chat --test runtime_gpu portable_exl3_prefill_matches_independent_single_rows -- --ignored`: exact K1..8, small rows/odd tails/23–25 boundaries and batches through513; fallback simulation on M5, not actual M1–M4 hardware. [Result](measurements/mtp-05/b-portable-fixed.log).
- Final activated kernels:24 recursive session blocks and all80 states/KV/residual/tokens exact; all logits and every retained prefix at verify widths2..4/prefix lengths1/23/24/129/256 exact. [Sessions](measurements/mtp-05/c-sessions.log), [rollback](measurements/mtp-05/c-rollback.log). Independent MLX-LM recursive head reference also passes with zero mismatches/max_abs0. [Reference](measurements/mtp-05/c-reference-final.log).
- SIMD router:96 fixtures covering ties, signed zero, signed NaN/infinity, denormals and varied FP16 values, normalized/un-normalized scores, production dispatch, malformed type/top-k and five-row fallback. All indices and scores agree bit-for-bit with the original kernel. Microtimings include host and synchronous evaluation, not pure kernel latency. [Result](measurements/mtp-05/c-dispatch-final.log).
- Final production bridge: all four modes/budgets1/2/3/4/17/64, invalid depths, cancellation/recovery, prefix reuse across depths and conversation isolation pass. Tuning itself was exercised separately by the final matrix and MTP-04 cancellation/winner tests. [Events](measurements/mtp-05/bridge-final.jsonl).
- `.venv/bin/python -m pytest -q`:26 pass, local Ling checkpoint test skipped. Swift6/warnings-as-errors hardening final3 passed and the production MTP view was rendered through AppKit with a disposable fake-engine fixture. Preview rates are explicitly fixture values. [Hardening](measurements/mtp-04/swift-final3-test.log), [preview](measurements/mtp-05/tuning-preview.png).
- A copied production QMV layout with truncated group count fails the regression; production source remained intact. [Negative test](measurements/mtp-05/mutant-rejected.log). The new Kani harness checks the actual production grid layout for all i32 row counts/grouping/large-output flags, every valid symbolic row, at most one padded row and reachability of M3/M4 last rows. Its85 obligations and3 covers pass. This is a bounded source-level check of Rust arithmetic, **not a proof of Metal/MLX/FFI or all GPU inputs**. Full final Kani results and delivery evidence follow below.

Full final `cargo kani --lib --no-default-features --output-format terse`: **30/30 harnesses verified,0 failures**. [Complete results](measurements/mtp-05/kani-final.log). Known unsupported foreign/caller-location constructs remain outside the reachable pure-Rust proof paths. Unreachable checks/covers are retained in the log; no unwinding or safety checks were disabled. This does not prove the whole engine or Desktop.

## Delivery

Pushed directly to main without a PR. Both release tags resolve to `3733e9df7cbc0298e81e778865e3abd8ac1ebd4b`. Packages were built before this documentation-only delivery record, from a clean tracked checkout. [Remote refs](measurements/mtp-05/release/refs.txt).

- [Rust workflow](https://github.com/0xZKnw/mlxl3/actions/runs/37304447742): all jobs succeeded on the release source. Linux/macOS formatting, strict Clippy, native tests/build, bridge benchmark validation and Kani completed; **30 successfully verified harnesses,0 failures**. [Status](measurements/mtp-05/release/ci-rust.json), [full log](measurements/mtp-05/release/ci-rust.log).
- [Desktop/protocol workflow](https://github.com/0xZKnw/mlxl3/actions/runs/37304447782): complete success on the same source. Python18pass/4optional skips in CI, full Desktop hardening, updater and MTP64-grid checks passed. Hardware/model checks were executed locally as described above; CI does not supply those checkpoints. [Status](measurements/mtp-05/release/ci-desktop.json), [full log](measurements/mtp-05/release/ci-desktop.log).
- `scripts/build-macos-dmg.sh`: Desktop1.3.0/build22, embedded engine1.3.0/protocol1/release/MLX0.32.2; clean build-info source matches the release commit. Existing SwiftMath/CLT-path warnings did not fail the build. [Build log](measurements/mtp-05/release/dmg-build.log).
- `build/mtp-v1.3.0/release-check --dmg "$PWD/dist/MLXL3-Desktop-v1.3.0-b22-Apple-Silicon.dmg" 3733e9df7cbc0298e81e778865e3abd8ac1ebd4b`: the real production updater validates digest, mounted DMG, deep/strict signature, arm64, version/build/minimum OS and runtime manifest hashes. Runtime-info/list execute without loading a model. [Proof](measurements/mtp-05/release/dmg-proof.json).
- `build/mtp-v1.3.0/updater-check "$PWD/build/mtp-v1.3.0/archive-fixtures" "$PWD/dist/MLXL3-Engine-v1.3.0-arm64.tar.gz"`: real archive signed installation, relocation, execution, fallback and hostile archive rejection passed in disposable directories. The actual-archive fixture uses engine1.3.0 with app/bundled version1.2.0 to check older-client compatibility. [Result](measurements/mtp-05/release/engine-install.log). This does not validate activation of an already-loaded model without restarting.

[Desktop v1.3.0](https://github.com/0xZKnw/mlxl3/releases/tag/v1.3.0) was published at11:52:39UTC and selected as latest. [Engine v1.3.0](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.3.0) was published at11:52:34UTC as the independent engine channel, not latest. Neither is a draft/prerelease. Draft lookup by tag returned404 before publication; the authenticated release inventory provided the asset metadata before publishing. Both uploads completed before publication and matched the local sizes/SHA-256 values:

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `MLXL3-Desktop-v1.3.0-b22-Apple-Silicon.dmg` | 73252487 | `bf5ec38cdeb1d022a79ccd70d029e2f54b6837bc74699c43ca6a6857dc850d72` |
| `MLXL3-Engine-v1.3.0-arm64.tar.gz` | 66851015 | `ab2cbf3f756468e1606ed29b9e900c5ee7ee8aa7ac1843c6664b3f82a2709530` |

`gh release download` retrieved both published assets again and both files matched their local/GitHub sizes and SHA-256 values. [Publication metadata](measurements/mtp-05/release/publication.json), [download proof](measurements/mtp-05/release/download-proof.json). `build/mtp-v1.3.0/release-check --channels build/mtp-v1.3.0/published-releases.json` applied the production updater selection to the live GitHub release list and selected Desktop1.3.0/build22 and engine1.3.0. [Channel result](measurements/mtp-05/release/channels-proof.txt).

Delivery is published, not installed on the user's Mac. Signatures are ad hoc; the packages are not notarized. Hot activation after engine download remains deferred. Actual M1–M4 hardware, all GPU/FFI/Swift states, whole-engine correctness and universal MTP2/MTP3 performance are unverified; the bounded CPU proofs and finite tests above do not establish them. No optimization trial or test inference process started by this work remains running.
