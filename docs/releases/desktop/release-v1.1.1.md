# MLXL3 Desktop v1.1.1 — one-click DFlash2 setup

For the Qwen3.6-35B-A3B EXL3 target, **Generation → DFlash2** now installs
Inco AI's separate draft and applies the required greedy settings with one
switch. The app downloads only seven pinned `draft/` files (~457 MiB), checks
their SHA-256 hashes and package structure, and resumes a partial download on
retry. An existing local draft can still be selected manually. The switch can
also cancel a download in progress.

DFlash2 remains experimental, opt-in and limited to this Qwen target. The DMG
contains the SwiftUI app, Rust/Metal engine and MLX runtime, **not** model or
draft weights. Speedups depend on the prompt and hardware; the first request
may compile Metal kernels. See the [README](https://github.com/0xZKnw/mlxl3#readme)
for setup, measurements and limitations.

**Signing limitation:** this build is ad-hoc signed, not Apple Developer ID
signed or notarized. macOS may require explicit approval in Privacy & Security.
Do not disable Gatekeeper globally; keep the previous app for rollback.
