# MLXL3 Engine 2.0.10

Stable Rust/MLX engine, published independently from Desktop. Desktop 2.0.0 build 53 bundles these engine bytes.

## Changes and fixes

- Added `serve --api-key-stdin` for delivery of the bearer key through a private pipe. The key is validated before starting the server, with a 512-byte limit and rejection of control characters, empty and oversized input.
- Preserved the existing `--api-key` interface for command-line callers.
- Retained the EXL3 inference, MTP/DFlash2, model management and OpenAI-compatible API paths from engine 2.0.9.
- The Desktop launcher now uses the private input. API requests still require the generated bearer key.

## Performance

No inference algorithm or kernel was changed for the private-input fix. New inference throughput or memory improvements were not measured.

## Installation and compatibility

Requires Apple Silicon and macOS 26.2 or later. The bundled runtime uses MLX 0.32.2. The artifacts have ad-hoc signatures. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

Validated locally on Apple M5 with 24 GiB unified memory and macOS 27.2. Rust formatting and strict Clippy for the shipped MLX/chat features passed. The CPU suite passed 527 top-level tests, with 10 ignored. The MLX library suite passed 461 tests, with 241 physical or fixture-dependent tests ignored. Two selected physical GPU tests ran separately and passed, covering LFM3 vision primitives and Q4/F16/BF16 invalid-input recovery.

The extracted engine passed real LFM3 text turns, an authenticated HTTP request, rejection of a zero token budget and recovery, LFM3-VL red/blue image discrimination, and Qwen3.6-35B-A3B 2.49 bpw requests with 1/2/8-token budgets. Two production StudioModel.send calls produced four persisted messages and recalled the supplied code. The server key was absent from process arguments.

Kani verified the private-key validator over symbolic byte strings of length 0–3. All 3 cover properties were satisfied. This bounded result covers the validator. Whole-program concurrency, GPU behavior and model quality have sampled test coverage.

The Desktop suite was completed in stages, preserving intermediate failures and rerunning affected checks. It includes 2,324 mode assertions, 384 preference assertions, installer/updater integrity checks, imports, streaming Markdown/math/code, conversation persistence, Agent workspaces and cancellation. Final fixed server transport passed the 2,324 mode assertions. Closing 24 native surfaces left zero retained view hosts in the sampled lifecycle check.

The website passed 69 tests with zero skipped, format/lint/type checks, and Chromium/WebKit rendering checks. Physical M1–M4, macOS 26.2, Gemma/Ling generation, external audio devices and every possible model configuration remain untested in this release campaign. Hosted CI status is recorded separately from these local results.

Compiled application and engine source revision `edee73ce2e4f56ed90fe4e2c973526d9cdfd518f`. Subsequent release documentation and source-inventory bookkeeping preserve the executable source. See [validation details](https://github.com/0xZKnw/mlxl3/blob/main/docs/releases/release-v2.0.0-validation.md).

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the design direction and testing on Apple Silicon. Thanks to [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for EXL3, [PonyExl3](https://github.com/beamivalice/PonyExl3) for conversion, [MTPLX](https://github.com/youssofal/mtplx) and [TensorFold](https://github.com/ashhart/TensorFold) for the existing speculative decoding work, and [SwiftMath](https://github.com/mgriebling/SwiftMath) for math rendering.

## Assets

- `MLXL3-Engine-v2.0.10-arm64.tar.gz`, 68,092,850 bytes. SHA-256 `a8e4a9d59c4574a05fc98c08627bf0241a558748370a08e0fd5bbb60aa3df9be`.
- `MLXL3-Engine-v2.0.10-arm64.tar.gz.sha256`, 100 bytes. SHA-256 `9d875ad77c8ace013507ca5a0a33a40b167db596500529b25742fec2d88b4edb`.
- The archive manifest records sizes and SHA-256 hashes for the executable, MLX libraries and Metal library.
- Published with `latest=false` to keep the Desktop DMG as the default GitHub download.
