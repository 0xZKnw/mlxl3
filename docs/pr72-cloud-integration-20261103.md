# PR 72 cloud integration — 2026-11-03

Source refs: upstream main `b9d5cbdce50aa98adfee560e06da75614e730bd3`, original PR head `bacee9d99f6fc1251b45a42172d117e3e4a13c54`, common ancestor `8c08c4c45c1f7795f476ca514e7c7d3b8cebce7b`. The initial cloud checkout was clean. No personal Mac, installed app, user configuration or model was accessed.

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


## CI counterexample and logo correction

The exact-head Desktop run [41600416935](https://github.com/0xZKnw/mlxl3/actions/runs/41600416935) failed at the historical logo visibility assertion on `492ad7c7` (exit 133). The full Python suite passed 2,362 tests with 137 skips and the app compiled before the failure. The preferences and notification executables had not been reached, so they were not counted as passed. The failure excerpt is preserved in `desktop-492ad7c7-failure.log`.

The previous pixel oracle assumed a dark background and required a channel to exceed the background by 0.3; that is impossible on the new light canvas. The light artwork also needs a darker tint for the left half of the mark. MonogramMark now uses the existing masked artwork as a template tinted with the adaptive ink in light appearance; dark and Midnight Blue preserve the original artwork. No asset or mask is changed. The historical oracle is explicitly dark; a new production-view bitmap test checks light/dark/blue, both halves of the mark, transparent corners, missing-mark negatives and the original untinted light rendering as a negative control.

`check-desktop.sh --preferences-only` executes the same preference/notification fixtures plus the logo bitmap check under a dedicated macOS CI job. The complete suite still runs them too. This gives an independently visible result even if a later existing Desktop test fails. Zsh argument validation runs before compilation; invalid options exit 2. Swift formal verification remains unavailable; no physical model or benchmark is run. The correction requires new exact-head CI validation and is not validated by the earlier failed run.


The focused [Desktop preferences job](https://github.com/0xZKnw/mlxl3/actions/runs/41602738499/job/125805101079) **passed on `43308c84d7e3889fa5b0147bdfad87a9ba8e23f8`**: 62 preference assertions, 17 notification/retry/shutdown assertions, 25 logo bitmap/negative-control assertions and 240 context-memory oracle cases. Actual Swift 6 compilation and the production subprocess boundary are exercised in the hosted macOS environment. Light/dark/blue left/right logo counts were 226/308, 248/317 and 248/309 respectively. The failure on `492ad7c7` remains a failure; it is not relabeled as successful. The raw success excerpt is preserved in `desktop-43308c84-focused.log`. Full-suite and native-proof statuses are reported separately at handoff.


The Apple `guard-proofs` job [125805101681](https://github.com/0xZKnw/mlxl3/actions/runs/41602738669/job/125805101681) was cancelled at its configured 15-minute job limit. The optional-Metal-API harness passed its 2 obligations and 4 covers, then `qwen_gated_delta_widths_match_wide_arithmetic_or_reject` did not finish before cancellation. This is **unverified**, not a counterexample or a success; the attention-width harness was not subsequently qualified. Native sources and the Apple workflow are unchanged from main. No bound was reduced and no blind rerun was requested to obtain green. The Metal/loader job in the same workflow passed. The diagnostic excerpt is preserved in `apple-guard-timeout.log`.


The complete Desktop job on `43308c84` reached its 25-minute job timeout while compiling the rendering fixture (`swift-frontend`), after successful 1,179 Studio-mode checks, 114 idle-unload checks, the 62/17/25 new assertions, 965 speculative-sampling checks, updater, HTML, file-import, workspace and bridge checks. No functional assertion failed in those completed stages; rendering and later checks were not completed. The timeout excerpt is retained in `desktop-43308c84-timeout.log`.

The full Desktop job budget is therefore increased from 25 to 35 minutes to include the additional real-source fixture compilations. All test cases, assertions and individual 90-second process limits are unchanged. The focused job remains at 15 minutes. This is a bounded CI-time allowance, not a benchmark or a relaxation of the correctness oracle. The next run must be inspected separately; the timeout does not count as a full-suite pass. The source of the product remains identical to the qualified `43308c84` implementation.


## Final cloud handoff

On `8b03ca93`, the focused Desktop job (62/17/25/240), native macOS/Linux jobs, Metal/loader, profile and conversion jobs passed. The complete Desktop job progressed past the old timeout and passed 141 raw-thinking cases, 1,194 code fragments and 4,845 Markdown fragments, then failed compiling the inherited ResponseUI fixture with Swift 6's `sending snapshot risks causing data races` diagnostic. It did not complete the bitmap/OCR and later Studio checks.

The final test-only correction explicitly isolates `checkFrame`, `capture` and `saveBitmap` on MainActor; the mutable AppKit snapshot stays on that actor while detached CPU OCR receives only the existing immutable CGImage. No unchecked Sendable conformance, test assertion or domain reduction is added. The exact optimized ResponseUI compilation is also included in the focused job to expose this diagnostic earlier. That final change requires remote Swift verification; no pass is claimed before the next job completes. Product and engine source remain unchanged from `43308c84`.

The native `routes` Kani shard on `8b03ca93` failed because two grouped-route harnesses each reached their 600-second bound. These properties remain unverified; no counterexample was reported. Apple guard verification also reached its 15-minute job limit. Native source and the Rust/Apple workflows are identical to main. The diagnostics are retained in `routes-kani-timeouts.log` and `desktop-8b03ca93-compile-failure.log`. The global CI is **not green**. No physical Apple-GPU/model or performance validation was run. Public PR title and description remain unchanged per the user's explicit choice.
