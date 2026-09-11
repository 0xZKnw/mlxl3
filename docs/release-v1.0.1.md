# MLXL3 Desktop v1.0.1

- Updated the engine bundled with Desktop to the current published MLXL3 engine.
- Fixed the macOS 27 TensorOps compilation error that prevented Qwen from loading
  (`get_destination_cooperative_tensor`).
- Faster exact EXL3 MoE prefill on M5: larger row tiles and reusable weight-address
  calculations, with the existing path retained for smaller workloads.
- Includes the engine improvements published since v1.0.0 and a release smoke
  test that now keeps the real model warmup enabled.

On the tested M5, paired 4k-token runs measured +12.4% prefill for Qwen3.6 35B A3B
and +34.9% for LFM2.5 8B A1B. A separate non-repetitive document measured +9.4%
and +25.6%, respectively. Text/cache parity checks passed. These are workload-
specific measurements, not universal speedups or a comparison against llama.cpp.
No decode or RAM improvement is claimed. See the
[full performance report](https://github.com/0xZKnw/mlxl3/blob/v1.0.1/docs/prefill-investigation-2026-09-11.md).

The DMG contains the native app, engine and dependencies, **not model weights**.
Requires **Apple Silicon and macOS 26.2+**. Physical validation is on M5;
other Mac generations and every supported architecture have not been retested.

**Signing limitation:** ad-hoc signed, not Apple Developer ID signed or notarized,
as in previous releases. macOS may require explicit approval under Privacy &
Security. Do not disable Gatekeeper globally. Keep the prior app for rollback.
