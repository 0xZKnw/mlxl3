# MLXL3 engine 1.4.0

- Add the pinned 4-bit MTP head for Qwen3.8-27B. `mtp-head --target MODEL`
  selects the dense or Qwen3.6-35B-A3B MoE head from the target configuration.
- Share head loading between bridge configuration, generation and MTP tuning.
  Reuse an identical head; unload head and caches on OFF, replacement or failure.
- Synchronize pending MLX operations at load/unload boundaries before clearing
  free buffers and reporting allocations. Generation retains its existing
  synchronization behavior.
- Add MTPLX-derived exact add/RMSNorm kernels with stock fallbacks and license
  attribution. They remain experimental and OFF by default: the measured
  workloads do not establish a speedup.

Apple Silicon/macOS 26.2+, Desktop **1.4.0+**, bridge protocol 1, MLX0.32.2.
The archive contains no weights. Restart Desktop after an engine update.
[Validation, measurements and verification limits](mtp-dense-1.4.0-validation.md).
