# PR 72 cloud integration — 2026-11-03

Source refs: upstream main `b9d5cbdce50aa98adfee560e06da75614e730bd3`, original PR head `bacee9d99f6fc1251b45a42172d117e3e4a13c54`, common ancestor `8c08c4c45c1f7795f476ca514e7c7d3b8cebce7b`. The initial cloud checkout was clean. No Mac, installed app, user configuration or model was accessed.

## Resolution

The merge retains the three Desktop feature commits and current main behavior. `native/` (128 tracked files) and `.github/workflows/rust.yml` match main byte for byte. Main already provides snapshot compaction and its F16 codec; the original RAM commit's BF16 exponent codec is a distinct candidate, preserved in commit history and historical reports, deferred for separate numeric, loading-peak, memory and latency qualification against the current engine. The old 806–867 MB results do not validate this integration.

Desktop conflicts combine the preferences and notifications with Agent registry preparation, Server/Lab, DFlash controls, bounded generation cancellation, process retirement and current streaming/preview rendering. SyntaxHighlighter keeps its module visibility because the current native code editor calls it. The main load guards remain intact. Completion notifications use a non-nil active request and are suppressed after either manual or watchdog cancellation. Stopping services also cancels pending model-load tasks so a queued task cannot restart the engine after shutdown.

## Validation

Linux x86_64, Python 3.12.14. Commands use an isolated `/tmp/pr72-venv`; no product dependencies changed.

- `python -m pytest -q tests/test_copy_test_runtime.py`: **20 passed**.
- Ruff check and format for the runtime-copy helper/tests: passed; Python compileall for helper/tests/engine fixture passed.
- `zsh -n scripts/check-desktop.sh scripts/build-desktop-test.sh`: passed.
- Every tracked native file and the Rust workflow compared with main: identical. No Rust/C++ numeric implementation changed, no CPU/GPU benchmark run, no memory/loading-speed result claimed.
- Swift Tree-sitter 0.7.4 / Python bindings 0.25.2: 18 changed Swift files parsed against main. No new diagnostics; existing parser limitations in MLXL3Bridge (1) and MarkdownResponseView (13) remain. This is a syntax aid, **not Swift type checking or an AppKit execution result**. Bindings 0.26.0 crashed initially; that attempt is not counted as a pass.
- `crosshair check scripts/copy-test-runtime.py --analysis_kind=asserts --per_condition_timeout=10`: no checkable functions, **no property verified**. The required `formal-proof-skill` was unavailable after checking repository/local skills and executor/cloud catalogs; it was not silently replaced or claimed as loaded. No Swift/Metal/concurrency proof claimed.

A second CrossHair PEP316 check calls the real `version` function from `version_contract.py` with symbolic strings of length ≤32, accepting ValueError or a three-element nonnegative tuple. The 15-second condition budget returned **Not confirmed**; no property proved and no counterexample established. The harness and diagnostic are preserved beside the local test logs.

The Desktop test script includes the existing preferences/retry tests and new deterministic service-shutdown, watchdog/completion-race and context-full notification regressions. The new processes run under the existing 90-second process-group supervisor. Linux has no Swift/AppKit toolchain, so these tests require the macOS PR CI. Real Apple-GPU model startup and numeric/performance validation were not run.

## Publication and CI

Code commit `1775c093de17bec727f08d606789bf6a54467f47` was pushed to `HENK0O/mlxl3:codex/desktop-theme-startup`. The remote ref was read back exactly; GitHub reports PR72 open, unmerged and mergeable=true. Main was rechecked unchanged at the baseline SHA. The initial exact-code CI snapshot is preserved in `docs/measurements/pr72-cloud-20261103/initial-ci-snapshot.json`: profile compatibility and EXL3 conversion/packing proof passed, while Desktop/Native/Apple guard proofs were still in progress.

The following documentation-only commit closes the local verification record and preserves the additional bounded-verifier attempt. It does not change the product source tested in `1775c093`. Its remote ref and CI must also be checked; the final handoff reports their observed status without presenting a pending job as successful. No local experiment or benchmark remains active. The PR remains open and unmerged.
