# MLXL3 Desktop 1.4.0 · build 23

- Automatic MTP for Qwen3.8-27B and Qwen3.6-35B-A3B: download and validate the
  matching pinned head when MTP is ON. Switching models unloads the previous
  head and prepares the new one before generation.
- Keep the global MTP ON/OFF preference across model switches, including models
  with old Tune MTP results. Saved per-model depths and measurements remain
  available. OFF, failed preparation and cancelled downloads release resources.
- Engine 1.4.0 includes target-aware head inspection/download, explicit bridge
  load/unload and accurate MLX allocation reporting after synchronization.
- MTPLX-derived add/RMSNorm kernels are numerically checked against MLX and
  attributed in the app/CLI and engine archive. They remain experimental and
  OFF by default: physical-M5 ABBA measurements did not establish a speedup.

Includes engine 1.4.0. The independent engine channel requires Desktop 1.4.0.
Apple Silicon, macOS 26.2 or later. No model weights are bundled.
[Validation, measurements and limits](mtp-dense-1.4.0-validation.md).
