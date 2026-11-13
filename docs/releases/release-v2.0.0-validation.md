# MLXL3 v2.0.0 release validation

Compiled source revision edee73ce2e4f56ed90fe4e2c973526d9cdfd518f. Desktop 2.0.0 build 53 and engine 2.0.10. Later release commits update documentation and the reviewed count of .take calls for the bounded private-key reader.

## Local results

- Rust formatting and strict Clippy for all MLX/chat targets passed.
- CPU suite passed 527 top-level tests, with 10 ignored. MLX library suite passed 461 tests, with 241 ignored.
- Two selected physical tests passed, one each for LFM3 vision primitives and Q4/F16/BF16 invalid-input recovery. The initial Q4 command selected zero tests because its module was incorrect. The corrected exact command ran one test.
- Kani passed private_server_key_rejects_empty_and_control_bytes on the real validator. Symbolic byte strings have length 0–3, with unwind 5. All three cover properties were satisfied, with zero failed checks and one unreachable obligation. Concrete tests cover empty, control, newline and 512/513-byte boundaries.
- The Desktop suite completed in stages, including 2,324 mode assertions, 384 preferences, installer/updater integrity, imports, streaming, Agent workspaces and cancellation. Intermediate snapshot failures and source-change compilation interruptions were preserved. The final server-pipe change passed all 2,324 mode assertions.
- Desktop experience passed 55 assertions and 14 background checks. Eighty-five frames captured first and subsequent sends. Real StudioModel.send calls produced four persisted messages and recalled a supplied code.
- Closing 24 native windows left zero retained hosting views in the sampled lifecycle check.
- Website format/lint/type/shell checks and 69 tests passed, with zero skipped. A Retina render test rejects UV deformation and verifies fixed-cell glyph transformation. Chromium and WebKit exercised the real renderer.
- The DMG passed image verification, read-only mounting, strict deep signature checks, build/version checks and executable byte comparison. Bundled engine files matched their manifest. Timeline/MCP checks and a disposable 16,384-parameter converter checkpoint probe passed from the mounted app.
- The separately extracted engine archive matched its manifest and reported MLX, chat enabled, arm64 and version 2.0.10. HTTP returned 200 for an authenticated request, 401 for anonymous access, 400 for a zero token budget and 200 for recovery.
- Process inspection confirmed that the final server's active key was absent from command arguments.
- LFM3-VL returned Red and Blue for two distinct solid images. The first 100-token attempt ended in reasoning. The 300-token follow-up asserted the actual answer section.
- Qwen3.6-35B-A3B EXL3 2.49 bpw generated exactly 1/2/8 tokens, rejected a zero budget and exited cleanly.
- The source-inventory check initially rejected the new bounded .take call. The reviewed count changed from 9 to 10, and all six inventory tests passed.

## Artifacts

- MLXL3-Desktop-v2.0.0-b53-Apple-Silicon.dmg, 250,940,191 bytes. SHA-256 83b88363d90e2717b96cab1ebe508b863adb40e51aab09193975e717834f79fa.
- MLXL3-Engine-v2.0.10-arm64.tar.gz, 68,092,850 bytes. SHA-256 a8e4a9d59c4574a05fc98c08627bf0241a558748370a08e0fd5bbb60aa3df9be.
- Engine executable SHA-256 5347b806e916b8ae7c779b1d8600a5a29039ed7db7cd9a55c44920a641179fcc.

Local raw logs remain in build/ascii/release-audit/ and build/ascii/refine/ in the release worktree. These ignored logs are separate from the published source and this summary.

## CI and scope

The initial hosted source-inventory failure was a bookkeeping mismatch caused by the bounded read. The corrective commit updates that count. Hosted results are inspected separately for the exact pushed SHA. Local success does not establish a green hosted run.

This campaign used Apple M5, 24 GiB and macOS 27.2. Physical M1–M4, macOS 26.2, Gemma/Ling generation, external audio devices and every model configuration remain untested. The signatures are ad-hoc. Final-rendering power and FPS were not benchmarked. Kani covers the specified CPU domain. GPU and application behavior have sampled tests.
