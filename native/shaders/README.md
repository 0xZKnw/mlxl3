# Metal kernels

Metal sources included or compiled by the engine: EXL3 and matrix operations (`exl3`, `_qmv*`, `_qmm*`), Flash (`flash_*`), Qwen, LFM3, DFlash, quantization, and shared primitives. The `.h` files share definitions between kernels.

Callers and path selection are in [native/src/](../src/README.md), particularly [gpu/](../src/gpu/README.md) and `linear.rs`. Compilation alone does not validate GPU numerical results; read the [instructions](../../AGENTS.md) and [journal](../../opti.md) before optimization.
