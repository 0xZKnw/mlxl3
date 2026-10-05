# MLXL3 engine 1.3.1

- Fix large model loading on Darwin: bound individual reads to 64 MiB while preserving partial-read, interruption and offset checks.
- Release unused MLX cache buffers after converting a large embedding during bridge model loading. Live model arrays remain intact.
- Use the existing BM64 dense prefill tile on eligible M5 shapes, with BM32 fallback for other hardware and dimensions.
- Harden native QMM and inference comparison tools: validate output shapes and finite FP16 values, preserve failed campaign reports, bound subprocess I/O and clean up children on cancellation or errors.

Includes [PR #23](https://github.com/0xZKnw/mlxl3/pull/23) and its separately tested fixes. Native MTP depths 1/2/3 and bridge protocol 1 remain available.

Independent engine update for Apple Silicon/macOS 26.2+, compatible with MLXL3 Desktop 1.2.0+. No model weights included. Restart MLXL3 Desktop after installing the engine update to load the new runtime.

The BM64 checks validate finite, bit-identical primitive outputs. They do not establish a universal performance gain or independently repeat the full Qwen checkpoint validation. [Full PR review and verification scope](https://github.com/0xZKnw/mlxl3/blob/f43118519d6af6cb15496e23520d4f61bd47e913/docs/review-pr23-2026-10-05.md).
