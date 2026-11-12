# MLXL3 Engine 2.0.7-beta.1

## Fixes

Qwen MoE models now limit unused MLX allocator buffers to **512 MiB** by
default. Long generations previously retained about 5 GB of reusable buffers
until the end-of-request cleanup, increasing memory pressure during generation.
The limit is selected when loading the model and also applies while using MTP
or DFlash2. The existing `MLXL3_ALLOCATOR_CACHE_MIB` override takes precedence.

## Memory and performance

On Apple M5 / 24 GiB, Qwen3.6-35B-A3B EXL3 2.49 bpw with MTP1, a synthetic
approximately 9,000-token prompt and 1,032 generated tokens used **18.51 GB →
14.02 GB** of process memory at completion with the installed 2.0.6 engine's
existing 512 MiB override. Both runs produced identical token and text hashes.
These measurements qualify the allocator setting; the release changes its
default. Process memory is distinct from the active MLX peak, which stayed
approximately 13.33 GB.

An incomplete timing comparison measured medians of 43.56 and 45.09 tokens/s
(three measured generations per setting). The reverse controls were not run:
**no confirmed speed improvement or universal no-regression claim is made**.
DFlash2 performance was not measured. Further benchmarks were stopped at the
owner's request. Weights, quantization, context, sampling and speculative
decoding algorithms are unchanged.

## Installation and compatibility

Independent engine prerelease for Apple Silicon, macOS **26.2+**, MLX backend.
Install through the Desktop engine updater or extract the complete engine
archive. The Desktop DMG is unchanged. This release uses `latest=false` to
preserve the Desktop update channel. Signatures are ad hoc, not notarized.
Dense Qwen and other model families retain their existing allocator defaults.

## Verification and limits

The existing engine's default/512 MiB paths produced identical output hashes
for the 1,032-token bridge case. A separate three-token CLI check compared
248,320 finite F16 logits and 80 nonempty state arrays per snapshot, with
bit-identical results across all three snapshots.

Release-specific build, loader regression and artifact results are recorded in
the attached validation report. The full test suite and additional performance
campaigns are intentionally omitted for this release at the owner's request.
Kani 0.68 was attempted on the real allocator helper but stopped with an
internal compiler error in its aarch64 intrinsic handling; it established no
proof of the helper, MLX, Metal or the complete application.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for reporting the memory usage and
requesting the default change, [Apple MLX](https://github.com/ml-explore/mlx)
for the allocator controls, [Qwen](https://huggingface.co/Qwen) for the model,
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) and
[PonyExl3](https://github.com/beamivalice/PonyExl3) for EXL3 foundations, and
[MTPLX](https://github.com/youssofal/mtplx) for the retained MTP implementation.
