# v1.2.0 verification record

## Environment and evidence

Apple M5, 24 GB unified memory, macOS27.2; Swift6.4 with SDK26.5; Rust1.98.1; MLX0.32.2; Kani0.68.0 / CBMC6.11.0. Sector power100%, other desktop applications open; thermal conditions uncontrolled. Repository start `d54540a317b5`, prior local changes preserved in `measurements/desktop-v1.2.0/opening-tracked.patch`. Results are local until publication is recorded in `opti.md`.

## Commands and inspected results

- `cargo fmt --check`; `cargo clippy --locked --features mlx,chat --all-targets -- -D warnings`; `cargo test --locked --features mlx,chat`. MLX root `.venv/lib/python3.12/site-packages/mlx`, deployment target26.2. Logs `clippy-final.log`, `rust-full-final.log`.
- CPU builds/checks with `--no-default-features --features chat`; logs `clippy-cpu-final.log`, `rust-cpu-final.log`. The first CPU Clippy rejected an unused GDN contract in test configuration; a real boundary regression test now exercises it.
- `cargo kani --lib --no-default-features --output-format terse`: **25 successful harnesses**, 0 failures. `kani-final.log` contains all assertions/covers. A separate copied-source mutant that ignores hardware capability fails the dispatch assertion; `kani-dispatch-mutant.log`. Production source was not mutated.
- `MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk MLXL3_TEST_ENGINE_ARCHIVE="$PWD/dist/MLXL3-Engine-v1.2.0-arm64.tar.gz" PYTHONPATH=src scripts/check-e2e.sh`: complete Python/Desktop suite, `e2e-final.log`. Archives, independent channels, URL/digest/version rejection, failure to copy the app, output bounds, atomic activation, corruption/disabled-engine fallback, real signed engine installation and execution from a temporary directory. Lifecycle, MCP, Unicode streaming, renderer, clipboard action, history and subprocess cancellation also pass. Native accessibility hierarchy was unavailable in the earlier copy harness; action and clipboard are tested, no AX click claim.
- Python packaging regression: deterministic archive bytes, complete member set and SHA-256, missing/empty/linked/oversized files, malformed versions, concurrent source changes preserving prior archive. Ruff on new Python files; Swift format/type checking; plist lint and zsh syntax.
- `scripts/build-macos-dmg.sh`: self-contained arm64 app/runtime/Metal assets, signatures verified; engine manifest hashes still match after app signing. `dmg-build.log`. Final mounted artifact checks recorded alongside release hashes.

## Numerical and physical GPU checks

- `MLXL3_DISABLE_TENSOR_OPS=1 cargo test --locked --features mlx,chat --test runtime_gpu portable_exl3_prefill_matches_independent_single_rows -- --ignored --exact`: EXL3 simple K1–8 and grouped K1–6/8, rows1/2/23/24/25/46/47/256/513, bit-exact compared with independent single-row execution. `portable-gpu.log`. This simulates the portable M1–M4 dispatch on M5.
- `scripts/check-mtp-reference.py TARGET HEAD build/verification/mtp-reference.json`, then `MLXL3_MTP_REFERENCE=... cargo test --features mlx,chat --test runtime_gpu native_mtp_head_matches_independent_mlx_reference -- --ignored --exact --nocapture`: official MLX-LM Qwen full-attention/MoE reference with absolute norm gains, affine4/group64 and independent embedding-row reads. 47 positions, chunks1/2/3/17/24: **bit-exact normalized hidden outputs**, maximum absolute error0. `mtp-reference-{python,rust}.log`.
- `cargo test --features mlx,chat --lib qwen35::tests::mtp_prefill_verification_and_rollback_match_target -- --ignored --exact --nocapture`: prompt lengths1/23/24/129/256, short-prefill execution matches ordinary decoding, exact logits and all recurrent/KV states after commit widths1/2. Invalid IDs, empty/overlong verification and context exhaustion discard partial state. `mtp-target-state-final.log`.
- Production JSON-lines bridge: cold MTP/ordinary IDs, text, histories and output budgets1/2/3/17/64; context and first-delta cancellation/recovery; penalty1.1 ordinary fallback. Prefix tests with402/408-token prompts, 256-token reuse and different-conversation misses. First prefix run exposed a misspelled negative-test field (`dflash` instead of `dflash2`), corrected before final rerun. `mtp-bridge-first.*`, `mtp-prefix-bridge.*`, `mtp-final.*`. Initial timings with compiler overlap are diagnostic, not claimed gains.

## Formal verification domain and remaining gaps

Kani checks actual CPU implementations: checked tensor ranges/grid arithmetic, codec permutations, GDN tape bounds, reusable-prefix eligibility, stop/tool markers, speculative acceptance and token/context budgets. MTP-specific harness uses symbolic u32 proposal/target IDs and usize budgets, depth1, unwind4, covers acceptance/rejection/zero and one-token budgets. Other harness bounds are in their source; covers were inspected. Unsupported Rust constructs reported by Kani are not reachable in these harnesses. Tests outside these domains are not proofs.

CrossHair0.0.101 `check scripts/package_engine.py --per_condition_timeout=20 --report_all`: no counterexample but both filename contracts **not confirmed**, no deductive proof. CBMC on `mlxl3_array_affine4` is blocked by modern Apple libc++ parsing; log `cbmc-ffi.log`. FFI/MLX/Metal/SwiftUI/network/concurrency are not formally proved. No verifier stubs were used to claim a GPU proof.

No physical M1–M4 validation, no guarantee for every Qwen-family checkpoint, no non-greedy speculative MTP, no universal performance or zero-bug guarantee. The dependency `block0.1.6` emits a Rust future-compatibility warning; vendored SwiftMath emits CoreText deprecation warnings. Hosted CI does not run physical GPU checks.

## Logo

Generated with the built-in image generator, then copied into `apps/MLXL3Studio/Resources/AppIcon.png` (1024 square). Design: large readable M3 monogram, ivory/platinum bevel, full dark graphite canvas, subtle material texture, no nested tile or inset icon. The same full-canvas source supplies every ICNS size; sidebar/menu bar use an M3 text mark. Final source is in the repository, not only the generator cache.

Reference architecture: [MTPLX qwen3_5_mtp_patch.py](https://github.com/youssofal/MTPLX/blob/9882703f3105363ddc37eca9f97aa09a1d387112/mtplx/qwen3_5_mtp_patch.py), Apache-2.0; [pinned MTP weights](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-MTP-4bit/tree/0295b81421bf4d0fccca9a7c0fcfb1418dda3516).

## Final bridge run

`mtp-final.jsonl` / empty `mtp-final.stderr`: all tests passed, including the corrected simultaneous-MTP/DFlash rejection. Final64-token ABBA observations on the402-token prompt: ordinary48.85/49.32tok/s; MTP57.48/56.64tok/s. Median ratio about+16.2% on this one prompt. Other desktop applications remained active; no thermal control and one ABBA group, so this is a measured local observation, not a universal model gain. The prefix test reuses256tokens with exact IDs; its warm/cold timings include cache state and are not added to decode gains.
