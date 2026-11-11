# MLXL3 Engine 2.0.5-beta.1

Independent engine prerelease **engine-v2.0.5-beta.1**, runtime version
**2.0.5**. Exact source and archive identity are recorded in the attached
validation report. Desktop beta 11 bundles the same runtime.

## Fixes

The Hugging Face Hub preview now reports LFM3 text (`lfm3`) and vision
(`lfm3_vl`) as supported. Previously, an outdated catalogue allowlist marked
these models unsupported even though the inference engine loaded them.
Unknown architectures remain unsupported; unknown cache estimates remain
unknown instead of inventing a memory estimate.

The existing LFM3-VL image protocol and LFM3 allocator defaults are retained.
Desktop beta 11 now exposes image attachments through that protocol. No
kernel, model-weight, quantization or numerical-operation change is included
in this fix, and no new throughput improvement is claimed.

The source includes changes already merged since engine 2.0.4, including
[PR #163](https://github.com/0xZKnw/mlxl3/pull/163)'s checkpoint integrity,
transactional recovery, bounded SSD cancellation, parser and runtime safeguards.
The native/MTP work in [PR #161](https://github.com/0xZKnw/mlxl3/pull/161)
remains in the source tree, but its standalone native backend is not shipped in
this MLX-only release. Historical performance measurements remain historical.

## Installation and compatibility

Apple Silicon, macOS **26.2+**. Use the Desktop engine updater or extract the
complete archive. All supported models use the MLX backend. Signatures are
ad hoc, not notarized. This prerelease uses
`latest=false` to preserve the Desktop release channel.

## Validation and limits

The attached report records the inspected Rust suites, Kani bounds, real Hub
responses, physical LFM3-VL image journey and package checks. The focused
support-classifier proof does not prove JSON parsing, I/O, MLX, Metal or the
whole application. Broad image understanding and performance are not measured
by the color controls. Historical ignored and platform-specific checks remain
identified in the report.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for reporting the catalogue issue,
and [LiquidAI](https://huggingface.co/LiquidAI),
[ExLlamaV3](https://github.com/turboderp-org/exllamav3),
[Apple MLX](https://github.com/ml-explore/mlx),
[PonyExl3](https://github.com/beamivalice/PonyExl3) and
[MTPLX](https://github.com/youssofal/mtplx) for the retained engine foundations.
Existing upstream notices remain included.
