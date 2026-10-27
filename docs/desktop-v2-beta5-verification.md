# Desktop beta 5 verification

Base: main `8d7f0bcc98cc419ae9572d5cf0d94f313a71ac45`. Branch: `codex/desktop-v2-beta5`. Native engine 2.0.1 is unchanged in source. Final source/package identifiers are added after the frozen build.

## Executed source checks

- `MLXL3_SKIP_METAL_CHECK=1 PYTHONPATH=src:scripts:benchmarks:apps/MLXL3Studio/Vendor/PonyExl3 .../.venv/bin/python -m pytest -q`: **1,896 passed, 93 skipped**, 143.20 seconds. Skipped cases are explicitly physical/optional cases, not a proof of their behavior. [Raw report](measurements/desktop-v2-beta5/python-full-final.log).
- Physical Mac, no concurrent compiler/prover/inference: `pytest -q tests/test_lfm25_moe_conversion.py -k test_cli_converts_complete_checkpoint_and_resumes_without_changing_weights`: **2 passed** (all-expert and routed calibration; complete checkpoint, exact coverage and resume with unchanged weights). [Raw report](measurements/desktop-v2-beta5/moe-physical-regression.log).
- `MLXL3_TEST_PYTHON=... MLXL3_MACOS_SDK=... zsh scripts/check-studios.sh`: **38 Python, 41 native project/artifact/controller, 25 sandbox, 29 protocol checks passed**. Actual Git, descriptor-relative writes, source conflicts, all-change preflight/restoration, historical sibling assets, ZIP, permission changes, malformed requests, silent/oversized processes and parent death are exercised. Controller cancellation uses a real child ignoring SIGTERM, waits for forced exit and verifies no child survives. [Raw report](measurements/desktop-v2-beta5/studios-focused-final.log).
- Swift 6 warnings-as-errors optimized response harness: parser/chunk parity (63 cases), clipboard actions, stale/cancelled renderer requests, bridge/history, HTML controls in Chat/Agent and LaTeX drawing pass. **81 native viewport frames** retain nonempty visible text across long prose, code and completion; initial/final anchor, interpolation and user follow-state transitions pass. Baseline also passes: the intermittent report was not reproduced. [Baseline](measurements/desktop-v2-beta5/response-baseline.log), [candidate](measurements/desktop-v2-beta5/response-fixed.log).
- Actual native views rendered and visually inspected: [Agent](measurements/desktop-v2-beta5/studios-ui-final/agent.png), [Artifacts](measurements/desktop-v2-beta5/studios-ui-final/artifacts.png), [Simple quantization](measurements/desktop-v2-beta5/studios-ui-final/quantization.png). Expert fields are compiled and wired to controller/backend options; not every gesture was automated.
- Ruff checks passed on changed Python worker/adapters/options/tests; `zsh -n` passed for build/check scripts. ShellCheck was attempted using Bash mode, which cannot parse the existing zsh array filtering; this is not a successful zsh source proof. SwiftMath/SDK linker warnings are inherited.

## Source verification

CrossHair was attempted on the production Python worker and is unavailable (`No module named crosshair`, retained diagnostic). No compatible Swift source verifier is installed; SwiftUI/AppKit/Seatbelt/Git/process behavior is tested, not proved. There is no new Rust implementation requiring a new Kani harness. Existing numerical adapter/packing checks remain useful finite tests, not a proof of all GPU operations.

## Retained failures and corrections

All trials are retained under [measurements](measurements/desktop-v2-beta5/). Initial targeted fixture output nested inside source was correctly rejected; the fixture was separated. Git initialization incorrectly combined bare init and work-tree flags; initialization now has its own arguments. An artifact identity assertion confused directory URL hints; canonical directory identity retains both versions. MCP list_files omitted optional arguments, now handled as an empty dictionary.

The first tiny source was not a valid hybrid LFM2 layout; inspection now rejects that profile. A valid conv/attention fixture converted 11/11 projections but its old optional quality loader lacked LFM2 mapping; quality uses the actual native engine after freeing source weights. A separate native validation scored 119 nonempty finite prediction rows. These synthetic metrics are not a model-quality benchmark.

The first full Python run exposed **32 failures, 1,927 passes, 30 skips**: a git-apply invocation beneath the parent repository silently skipped vendor paths. Raw long output was not redirected and the tool truncated it; this evidence limitation is recorded. The supplier was reconstructed from clean pinned upstream `8e7fa6b1556f59fc669e25087903b279b9b0346f`, with the repository's consolidated patch actually checked/applied in a disposable repository. Personal source modifications were never copied. Incremental cache patches are already superseded by the consolidated implementation and are not stacked. Thirty failures then passed; the remaining two child-process fixtures forced a nonexistent references path, corrected to the shipped vendor. The two physical journeys subsequently passed. Full CPU suite then passed.

UI fixture errors (nonexistent artifact.file; later hostile symlink fixture), a concurrent fixture rewrite during compilation, and an over-tall permission row are retained with explicit failed runs. The final stable source passes; the permission row now uses intrinsic height. No failed run was relabeled successful.

## Remaining limits

Large public-model conversion quality/capacity was not newly tested. Ling full-teacher quality is intentionally unavailable. Git-backed Agent/artifact functionality requires Apple's Git. The display mitigation is covered by deterministic visibility checks but the exact user failure is not reproduced. The existing CDN allowlist and Seatbelt policy remain the boundaries. No new throughput, peak-RAM, power or thermal claim is made.

## Frozen package qualification

Production source: `f200014a4e68b698c0a1a1b840e3411f3f329c2c`. The final clean build (135.75 seconds) produces Desktop 2.0.0/build36 and engine2.0.1 with this source revision. The read-only DMG passes production updater preflight **build35 → build36**, deep signature, arm64, OS compatibility and runtime-file hashes. [Preflight](measurements/desktop-v2-beta5/updater-final-validation.json). No installed app was replaced.

The actual frozen worker and native engine executed from the mounted read-only image. A synthetic conv/attention LFM2, H128/FF128/V128,247040parameters, single Safetensors file without index,4bpw,32calibration rows/32tokens and explicit q_proj exclusion completed **11 EXL3 projections**, with **54 tensors / 90379 stored values** nonempty and finite. Source files and excluded FP16 projection are byte-exact. Held-out validation scored **119 predictions** with finite native/source perplexity and KL. These random-weight metrics are functional checks, not public-model quality claims. The archive includes the disposable input/output fixture. [Qualification](measurements/desktop-v2-beta5/packaged-qualification.json).

The frozen Hub probe resolves LiquidAI/LFM2.5-2.6B to `b212e75975bbda9676a94d3cb7b2ca25b65fed48`, inspecting config/metadata without weight downloads. Actual worker termination is reaped. Killing its parent while keeping the stdin writer open still exits the worker within the bounded check. The worker manifest's complete source-file hashes match the frozen source.

The initial frozen single-file journey failed before GPU work because Pony's SafetensorIndex required a shard index. A minimal read-only header fallback fixes that production boundary; multiple unindexed shards are rejected. The retained regression now has **39 passing focused Python cases** on final source. The full CPU/Swift and physical indexed-MoE runs above predate this isolated fallback; unchanged paths were not rerun locally. GitHub runs the relevant full suite on the final PR commit.

Build/signing warnings remain inherited; the app is ad-hoc signed and not notarized. CI status is inspected separately and never inferred from these local results.
