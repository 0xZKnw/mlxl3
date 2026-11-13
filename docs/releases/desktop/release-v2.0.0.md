# MLXL3 Desktop 2.0.0

The stable v2 release, internal version **2.0.0 / build 53**, bundling **MLXL3 engine 2.0.10**.

## Changes

- Chat, Agent, Server and Lab workspaces with local EXL3 inference, streaming responses, reasoning, code, math, conversation branches and file attachments.
- Reviewed Agent project changes, isolated workspaces, checkpoints, restore and export.
- Local model management, Hugging Face downloads and a bundled conversion environment.
- A fine ASCII aurora in the main window and menu bar widget. Characters transform in fixed cells as the light passes. Generation changes the animation smoothly, and Reduce Motion uses a still background.
- Translucent native controls, the actual MLXL3 logo in the menu bar, and a widget focused on the model, app memory and generation metrics.
- Appearance choices are System, Light and Dark. Iris is the default accent, adapted to each appearance. Existing theme preferences migrate automatically.
- A gentle transition moves the composer to the bottom after the first message. Later sends have a small insertion animation.
- A fullscreen website with the same visual direction, a fluid cursor interaction, GitHub/Discord links and the stable install command.

## Fixes

- Removed the stationary halo trace from the animated background.
- Preserved the final CLI event when stdout ends without a newline and bounded stderr-tail reads.
- Passed the Server bearer key through a private pipe. The active key is absent from process arguments.
- Build 53 enables upgrades from v2 beta 21, which used build 52.
- Retained the beta 21 fixes for unloaded-model navigation, popover sizing and disabled controls.

## Performance

The engine retains the previously released inference paths and adds the private server-key input. This release campaign measures correctness and delivery. Throughput, energy use and the final animated background's GPU cost were not benchmarked.

## Installation and compatibility

Requires Apple Silicon and macOS 26.2 or later. The bundled runtime uses MLX 0.32.2. The artifacts have ad-hoc signatures. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

Validated locally on Apple M5 with 24 GiB unified memory and macOS 27.2. Rust formatting and strict Clippy for the shipped MLX/chat features passed. The CPU suite passed 527 top-level tests, with 10 ignored. The MLX library suite passed 461 tests, with 241 physical or fixture-dependent tests ignored. Two selected physical GPU tests ran separately and passed, covering LFM3 vision primitives and Q4/F16/BF16 invalid-input recovery.

The extracted engine passed real LFM3 text turns, an authenticated HTTP request, rejection of a zero token budget and recovery, LFM3-VL red/blue image discrimination, and Qwen3.6-35B-A3B 2.49 bpw requests with 1/2/8-token budgets. Two production StudioModel.send calls produced four persisted messages and recalled the supplied code. The server key was absent from process arguments.

Kani verified the private-key validator over symbolic byte strings of length 0–3. All 3 cover properties were satisfied. This bounded result covers the validator. Whole-program concurrency, GPU behavior and model quality have sampled test coverage.

The Desktop suite was completed in stages, preserving intermediate failures and rerunning affected checks. It includes 2,324 mode assertions, 384 preference assertions, installer/updater integrity checks, imports, streaming Markdown/math/code, conversation persistence, Agent workspaces and cancellation. Final fixed server transport passed the 2,324 mode assertions. Closing 24 native surfaces left zero retained view hosts in the sampled lifecycle check.

The website passed 69 tests with zero skipped, format/lint/type checks, and Chromium/WebKit rendering checks. Physical M1–M4, macOS 26.2, Gemma/Ling generation, external audio devices and every possible model configuration remain untested in this release campaign. Hosted CI status is recorded separately from these local results.

Compiled application and engine source revision `edee73ce2e4f56ed90fe4e2c973526d9cdfd518f`. Subsequent release documentation and source-inventory bookkeeping preserve the executable source. See [validation details](../release-v2.0.0-validation.md).

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the design direction and testing on Apple Silicon. Thanks to [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for EXL3, [PonyExl3](https://github.com/beamivalice/PonyExl3) for conversion, [MTPLX](https://github.com/youssofal/mtplx) and [TensorFold](https://github.com/ashhart/TensorFold) for the existing speculative decoding work, and [SwiftMath](https://github.com/mgriebling/SwiftMath) for math rendering.

## Assets

- `MLXL3-Desktop-v2.0.0-b53-Apple-Silicon.dmg`, 250,940,191 bytes. SHA-256 `83b88363d90e2717b96cab1ebe508b863adb40e51aab09193975e717834f79fa`.
- Engine [2.0.10](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.10) is bundled.
