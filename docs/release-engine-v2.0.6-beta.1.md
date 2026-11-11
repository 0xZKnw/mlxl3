# MLXL3 Engine 2.0.6-beta.1

Independent engine prerelease **engine-v2.0.6-beta.1**, runtime version **2.0.6**.
This is an engine update; it does not replace the Desktop application.

## Changes

LFM3 text EXL3 8 bpw on Apple M5 now selects two qualified prefill paths by
default: a grouped TensorOps QMM for gate/up and QKV projections, and temporary
rotated-basis F16 weight reconstruction followed by GEMM for selected larger
projection batches. Model weights and quantization remain unchanged. Expanded
weights are temporary and are not retained as a dense model cache.

The existing EXL3 projection bundle is reused for the FFN pair, with separate
projections retained for unsupported formats, external biases or logical padding.
Grouped kernel factories are keyed by rows rounded to 32, so nearby prompt
lengths can reuse the same factory.

## Defaults and compatibility

The new paths require the standalone `lfm3` architecture, an 8 bpw checkpoint,
MCG8 projections, Apple M5, batch one and F16 activations. Grouped QMM covers
24–256 rows. Reconstruction/GEMM covers 128–256 rows for qualified projections
with 2048 outputs, and 256 rows for the other qualified shapes. Unsupported
cases retain their previous paths. LFM3-VL, other GPUs and other quantizations
are not automatically opted into this profile.

Set `MLXL3_LFM3_PREFILL_GEMM=0` or `MLXL3_LFM3_PREFILL_GROUPED=0` before loading
the model to disable either path. Invalid override values fail closed. No draft
model, speculative sampler, asynchronous decode pipeline or experimental KV
slab is enabled by this release.

Apple Silicon, macOS **26.2+**, MLX backend. Install with the Desktop engine
updater or extract the complete archive. The package includes MLX libraries,
Metal shaders, component hashes and upstream notices. Signatures are ad hoc,
not notarized. The engine release uses `latest=false` to preserve the Desktop
update channel.

## Performance and numerical behavior

Earlier M5 component and application experiments showed a repeated prefill
latency signal of about 8–11% after warmup. Those measurements used a separate
prototype source tree and are **not a guaranteed gain for this release**.
Decode throughput and total generation time did not show a robust general
improvement. Complete French responses, including reasoning, were included in
the historical comparisons.

Grouped QMM components are checked bit for bit. Reconstruction/GEMM changes
reduction details: validation requires model KL divergence at most 0.001 and
relative state L2 error at most 0.01. Temporary reconstruction also increases
scratch memory; the historical long-context comparison observed approximately
74 MiB more peak MLX allocation. This is not a claim about peak process RAM.

The attached validation report records the exact source, final archive identity,
checks actually run, ignored tests, physical numerical controls and remaining
limits. Kani verifies the CPU admission/default-selection guards; it does not
prove MLX, Metal, I/O or the complete generation pipeline. Hosted CI status is
reported separately from local checks.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for requesting and evaluating the
LFM3 performance work, [LiquidAI](https://huggingface.co/LiquidAI) for LFM3,
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the reconstruction
strategy, and [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime
and TensorOps foundations. Thanks also to
[PonyExl3](https://github.com/beamivalice/PonyExl3) and
[MTPLX](https://github.com/youssofal/mtplx) for the retained engine foundations.
Existing license notices remain included.
