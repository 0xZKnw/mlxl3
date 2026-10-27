# Updater progress, HTML previews and Agent recovery

Base: `7678456da8ff163f0609ea4e709a6cf5d5a97895`. Worktree: `codex/updater-html-agent-guards`. The original personal checkout, installed app and published beta 3 assets were preserved. No weights, numerical inference implementation or Metal shader changed.

## Behavior

The app and engine updater display received bytes, percentage and transfer rate. A session-level download delegate reports real byte counts, throttled to five updates per second. The existing monotonic rate tracker is reused. A stopped stream displays zero rate after two seconds; SHA-256 verification has its own label and installation still requires exact size and digest verification. Cancellation invalidates obsolete callbacks and removes staging files.

Completed HTML blocks and standalone documents expose **Run HTML** in both Chat and Agent. The entire message must be finished and error-free, including when an older answer part or Markdown chunk is already frozen. Agent also lists actual HTML files in its private workspace after a turn. The preview uses an ephemeral WKWebView, a restrictive response-header CSP and independent WebKit resource rules. There is no native script-message bridge. New windows, host navigation, file pickers and media capture are refused.

Inline JavaScript, ES modules and import maps work. Remote dependencies are restricted to HTTPS jsDelivr, unpkg, esm.sh, cdnjs, Skypack and threejs.org. Local Agent modules, CSS and static resources are read only inside the task folder, at most 2 MiB per resource. Canonical validation is followed by descriptor-relative `openat` traversal with `O_NOFOLLOW`; the opened descriptor must be a regular file with one hard link. Reads are bounded and nonblocking opens reject FIFO/device substitutions. Chat has no task-folder access.

Agent command preparation is now visible while the model generates hidden tool arguments. Agent requests have a 4,096-token budget shared across tool rounds, a ten-minute overall deadline and a two-minute silence deadline. Stop first requests cooperative cancellation; after three seconds it unloads an unresponsive engine, with the existing two-second SIGKILL fallback. A new model or Server waits for retiring processes to exit. Partial output and the reason for stopping remain visible. Deleting the final Agent task now reloads the actual Chat registry.

The command sandbox remains native Seatbelt, without Docker. Its existing five-second command and 32 KiB output limits remain in force. It has no hard aggregate RAM/disk quota.

## Verification

Evidence: [raw records](measurements/desktop-updater-html-agent/).

- Final Python suite, with `MLXL3_SKIP_METAL_CHECK=1 PYTHONPATH="$PWD/src"` and the shared Python 3.12 venv: **1,833 passed, 115 skipped**, ten warnings, 143.28 seconds. The package and four surviving modules resolve inside this worktree; no foreign module is recorded in `python-provenance.json`. The original shared-venv attempt imported the personal checkout and is not used as source qualification. A provenance-driver launch then caused two collection errors because its initial `sys.path` excluded the repository root; that diagnostic is retained and the driver was corrected to match `python -m pytest`. `check-e2e.sh` now pins and asserts the current source root. Skips include unavailable Torch/pony conversion dependencies, explicit hardware cases and absent fixtures. Python production code was unchanged.
- Final `MLXL3_SKIP_METAL_CHECK=1 MLXL3_MACOS_SDK=... MLXL3_TEST_PYTHON=... scripts/check-desktop.sh`: **passed in 467.28 seconds**, within its 1,100-second bound. SDK 26.5, Swift 6 strict app-source harnesses. Relevant observations include **1,174 modes/lifecycle assertions**, **114 idle-unload**, **965 speculative settings**, **25 real sandbox**, **17 MCP protocol**, **22 real updater transfers**, **15 signed updater preflight** and **229 workspace** checks. The unrelated physical Metal footprint check is explicitly skipped. The first full attempt caught an introduced logical `isRunning` regression; a separate retiring-process list corrected it before this final pass.
- Updater HTTP fixtures exercise partial byte counts and positive rates with known and missing Content-Length, cache verification, truncation, oversized responses, HTTP 503, wrong hashes, cancellation and staging cleanup. The same **22 checks pass under AddressSanitizer**. The first async task-delegate implementation failed to expose intermediate callbacks; the retained session delegate passes the actual HTTP oracle.
- **34 default WebKit checks**, or **36 with the optional real CDN and generated-file journeys**, check inline scripts, interactivity, import maps, transitive local modules, CSS, dependency imports, denied host files/LAN/navigation/popups/storage/cookies, symlinks/hardlinks, size boundaries and non-regular files. A separate loopback server verifies that forbidden requests never arrive. A real **Three.js 0.179.1** import renders a green cube into a 32×32 WebGL canvas; its central pixel and GL error state are checked. The actual Qwen-generated HTML also executes in the production preview and its button changes to `42`.
- Native UI captures verify Chat and Agent Run-button visibility during streaming, subsequent tool processing, completion and interruption. The failed long-transcript oracle auto-scrolled outside its capture region; the corrected finite pixel fixture uses a short multipart message plus explicit global streaming-gate checks on a large prefix. Internal accessibility queries returned no usable tree and are not reported as a UI success.
- Removing the real wait-before-reload call in a disposable source copy causes the lifecycle test to fail because the prior engine PID is still alive. Production sources were not mutated. The original final-Agent deletion bug was also reproduced before correction.
- `cargo test --locked --lib --no-default-features`: **155 passed, 5 ignored**. `cargo clippy --locked --release --features mlx,chat --all-targets -- -D warnings` and the actual `mlx,chat` release build pass. Rustfmt, strict formatting of new Swift/harness files, Ruff checks/formatting and shell syntax are checked separately.

## Source-level verification and limits

Kani 0.68.0 / CBMC 6.11.0 verifies the actual preparation flag over all four `inside`/`passthrough` boolean states and `remaining_tokens` over symbolic `i64`/64-bit `usize` inputs. Both functional assertions are reached; **2/2 and 3/3 cover properties are satisfied**. Two unreachable standard-library layout/format checks belong to constructing the empty buffer, not the functional assertion. See `kani-targeted.log` and its bounded supervisor result.

The attempted full String-filter transition harness did not finish and was stopped after approximately 2h47. Its source and log are retained; it establishes no proof. It is excluded from CI. Finite production tests cover all 45 opening-marker split positions across the three supported formats and ordinary passthrough text. These are tests, not a proof for arbitrary streams.

No practical Swift source verifier is installed. CrossHair is absent for the Python verification drivers. Swift compilation, ASan, pixels, temporal fixtures and sampled browser/model journeys do not prove all concurrency, operating-system, MLX, Metal or JavaScript behavior.

## Physical Qwen3.6 journey

Apple M5, 24 GiB; OS, power state, model metadata/config hash, binary hashes and source-file hashes are recorded in `qwen-journey.json`. Qwen3.6-35B-A3B EXL3 **2.49 bpw**, context 4096, greedy/top-k 1/repetition 1, MTP and DFlash off. One model at a time and no compilers/provers during the run. This is functional qualification, not a throughput comparison.

The real native engine and real Swift sandbox helper created `index.html`, read back `MLXL3-QWEN-OK`, emitted visible command-preparation status and replied `Done.`. **209 total generated tokens, one actual tool call**. SIGUSR1 cancellation was acknowledged, a fresh turn replied `Hello.`, and the engine was stopped and joined. The generated file is retained and separately replayed through the HTML preview. The original exact user prompt and speculative settings were unavailable; this does not establish a universal cause for every reported hot/stalled generation.

## CI and publication

PR/push-main workflows are present. The independent Kani job includes both affected properties. Baseline commit `7678456d` has successful Desktop, conversion, Linux native, macOS native and independent Flash Kani jobs. Its global Kani job failed with a runner-shutdown signal/exit 143, without an established counterexample; the diagnostic is retained. Baseline CI is not CI for the modified sources.

This initial implementation qualification did not create a new release or replace an installed app or published beta 3 asset. Subsequent beta 4 packaging is recorded separately in `docs/release-v2.0.0-beta.4.md`. Both app and engine source changes are needed for the complete Agent preparation/budget behavior; an older independent engine lacks those additions. Packaging a subsequent release must give its updated engine an appropriate new version.

The final source files are identified in `final-source-identity.json`; commands and outcomes are retained beside it. The Qwen helper was built before the later HTML reader hardening: its Agent sandbox/engine source was unchanged, and the affected HTML reader was qualified separately on the final source. This is component qualification, not a claim that a new signed release package has been tested. Follow-up branch CI must be inspected at its exact head; these local results do not establish a GitHub CI success.
