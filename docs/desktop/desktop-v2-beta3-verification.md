# Desktop v2.0.0-beta.3 verification

Executable source: `45697feb06cac08983683c9ad5c1b8afdb8e9d32` on `codex/desktop-v2-beta3`, [PR #55](https://github.com/0xZKnw/mlxl3/pull/55). Base: `7978fceac30d991a8c06d72114fc932b2443f555`. The original checkout and its existing modifications were preserved. No native inference implementation, shader or model weight changed in this work.

## Delivered behavior

Chat / Agent / Server / Lab navigation sits above history. Agent tasks retain their own folders and mode, use a separate MCP registry, and reuse the resident model between tasks. The only exposed Agent tool runs commands through native Seatbelt. Server starts the existing loopback API with a fresh API key and owns the model until stopped. Lab records valid completed-generation observations with their model, settings and hardware/runtime identity. Server requests are explicitly excluded from Lab history.

The final welcome screen uses the original MLXL3 artwork, rendered once with transparency. The first text monogram and the second rectangular-background integration were rejected and retained as intermediate evidence. Six background samples must match the canvas, and more than 100 sampled pixels must show the actual symbol. The packaged application was also launched in isolated `--ui-preview` mode: Agent was selected through the real button, its clean logo was visually inspected, and Lab's hardware/software panel was opened. That owned preview process was stopped; no installed application was replaced.

## Commands and inspected results

- `MLXL3_MACOS_SDK=... MLXL3_TEST_PYTHON=... scripts/check-desktop.sh`: two complete local passes, including lifecycle, downloads, speculative settings, file import, persistence, updater, streaming, Markdown/LaTeX pixels and CLI transport. The later pass used `MLXL3_SKIP_METAL_CHECK=1`. Logs: `desktop-check-3.log`, `desktop-final.log`.
- `swiftc -swift-version 6 -warnings-as-errors ... tests/studio-modes-check.swift`: final executable-source compilation and **1158 checks with the screenshot**, including **1125 finite combinations** of metrics/count boundaries, separate Agent registry, actual launch arguments, task/model reuse, branches/relaunch, model identification, ephemeral-key authentication and real HTTP fixture lifecycle. This is finite boundary testing, not a proof over all floating-point inputs.
- Strict Swift sandbox harness: **24 passed**, exercising real file creation/readback, denied absolute and symlink access, inherited child restrictions, denied loopback network, sanitized environment, timeout, bounded captured output, child cleanup, rejected group escape and invalid commands/timeouts. Command byte bound 16384; raw captured output bound 32768; command deadline 0 < t <= 5 seconds. Selected rejection cases include empty/NUL/oversized commands and zero/negative/nonfinite/over-limit durations.
- `python tests/check-agent-protocol.py <helper>`: **17 passed**, with a real MCP initialization/list/call exchange, nonempty file oracle, invalid/oversized/truncated input, error responses and cleanup after killing the MCP parent.
- `swiftc -sanitize=address ... AgentSandbox.swift tests/agent-sandbox-check.swift`: **24 passed on the retained implementation**, no sanitizer error reported. This is a dynamic memory-safety check.
- A temporary `deny default -> allow default` mutant was rejected by the disposable secret-read test; production source stayed unchanged. The logo pixel test also failed on the bad background before the retained correction.
- `python -m pytest -q tests/test_native_json_process.py`: **27 passed**, covering the reused bounded process transport, invalid/partial responses and failures.
- `xcrun swift-format lint --strict` on the new Swift implementation/harness files, Ruff on the new Python fixtures, `zsh -n scripts/check-desktop.sh`, Python byte compilation and `git diff --check`: passed after formatting/import fixes. Existing changed Swift sources follow repository style and were compiled with warnings as errors in the harnesses.
- Official `scripts/build-macos-dmg.sh` with MLX 0.32.2, SDK 26.5 and the existing Python environment: passed from a clean, frozen source commit. Native release build used locked dependencies and `mlx,chat`.

## Physical packaged journeys

Apple M5, macOS-27.2-arm64-arm-64bit; AC power, battery 100%. One Qwen3.8 27B EXL3 2.00 bpw model at a time, context 2048, temperature 0, top-k 1, repetition penalty 1, MTP/DFlash off. One functional generation, maximum 512 tokens and 180-second deadline; no throughput comparison or benchmark warmup.

The actual packaged bridge and actual packaged Swift MCP helper completed a tool call, created `smoke.txt`, read back `MLXL3-BETA3-OK`, and replied `Done`. **327 generated tokens, 1 real tool call**, finite nonempty metrics and the independent on-disk file oracle passed. After that engine was stopped and joined, the packaged Server returned 401 without its key, listed exactly the selected model with the key, and produced a nonempty authenticated 8-token completion. Both native processes were stopped and joined. Raw observations are in `package-journeys.json` and `packaged-agent-events.jsonl`; they are functional evidence, not a performance qualification.

One earlier complete Desktop pass also checked its existing physical 64 MiB Metal-buffer footprint primitive. Its observed +97402880 bytes is retained as a raw unit-test result, not a new memory saving or benchmark.

## Packaging

**MLXL3-Desktop-v2.0.0-beta.3-b34-Apple-Silicon.dmg**: 74120917 bytes, SHA-256 `5a8110b9fb02f358938e588d68ea0cd0badabc372939f4daf151414b937bdb30`.

Eleven packaging commands passed: deep strict signature validation, arm64 architecture checks, DMG verification, read-only mount, mounted signature validation and relocated runtime metadata. Bundle identity is clean `45697feb06cac08983683c9ad5c1b8afdb8e9d32` and engine identity `45697feb06ca`/release; four runtime-file hashes match between the application and mounted DMG. The read-only image was detached. The actual DMG passed the production updater's preflight in a disposable installation, along with **15 signed compatibility/integrity cases**. No personal model, history, installation or credential was replaced.

## Proof limits and retained failures

Swift has no usable source-level verifier installed here. CrossHair is unavailable for the Python verification drivers. Compilation, sampled execution, finite boundary exploration, pixel checks and AddressSanitizer are **not formal proof**. No Rust code changed, so no new Kani property is claimed. The unchanged base's native/Flash Kani checks and global Kani runner failure are distinguished in the archived CI evidence.

The first global warnings-as-errors SwiftPM build failed on two pre-existing deprecated SwiftMath APIs; the normal dependency build and strict app-source harness compilation passed. Two intermediate check attempts failed on test visibility/setup or source modification during compilation; subsequent frozen runs passed. Initial Seatbelt profiles failed dyld/path handling; group escape and interactive MCP buffering exposed real failures that were corrected. Intermediate logo backgrounds failed visual/pixel checks. These failures were retained, not overwritten as successful runs. One shell cleanup command was automatically rejected before execution; updater qualification then used a disposable Python TemporaryDirectory successfully.

The sandbox is a same-kernel process policy, without a hard aggregate RAM/disk quota. It provides system tools and a private task folder, with network and external MCP denied; it does not provide arbitrary host projects, downloaded dependencies, general Homebrew/user environments, or a VM. Model tool-calling behavior remains model dependent. This single physical journey does not qualify every model or generated project. Old Desktop versions may discard the optional Agent/Lab metadata on save; Agent workspace files are retained when deleting a task.

GitHub CI for the exact executable source is recorded separately in `ci-source.json`. Pending, cancelled, failed and successful jobs must not be conflated; no all-green claim is inferred from workflow definitions or previous source commits.

Publication was explicitly requested before CI completion. At the saved source snapshot, conversion contracts passed and Desktop/Native workflows were still running; see the exact per-job statuses in the archive.

## Publication and user updater check

The prerelease is public at https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.0-beta.3. All four GitHub asset sizes and storage SHA-256 digests match local files; the three smaller assets were downloaded without authentication and verified. The complete remote DMG transfer exceeded the 120-second qualification limit, so it is not reported as a completed public download. The local signed/mounted DMG and GitHub storage digest were verified. A REST lookup by the unpublished draft tag returned 404; the published tag resolves exactly to the recorded executable source.

When the user reported a stalled update, their real URLSession file already contained 37172299 of 74120917 bytes: the transfer was slow and its UI had no percentage. The already-verified local DMG was staged atomically into the updater cache; the slow task was cancelled via channel selection, Beta was restored, and Search used the verified cache. The real user app then displayed “v2.0.0-beta.3 is ready”, verified SHA-256, and “Restart and install”; Engine was up to date. Installation was left to the user. No new inference/benchmark was run for this delivery workaround.
