# MLXL3 Engine 2.0.3-beta.1

Prerelease `engine-v2.0.3-beta.1`, internal engine version **2.0.3**, published with `latest=false`. Build source: **b8472fc9073c3ff1ec1e314de327167de1d20e6a**. Desktop **2.0.0-beta.7 / build 38** remains a separate release with its bundled engine 2.0.2.

## New features

- Automatically run **Qwen3.8 Flash Next** (`qwen4_exp`) through the native Metal backend, in a process that does not load MLX. Other model architectures continue through MLX.
- Add LFM3-1.7B text and LFM3-VL-2B image inference. CLI `--image`, JSONL bridge images and OpenAI-compatible HTTP image requests accept embedded PNG/JPEG/WebP images; remote image URLs are rejected.
- Add two Metal fusions shared by both LFM3 backbones: the seven-tap single-token B×X/convolution/state/C-gate operation, and attention sigmoid/layout fusion. Other convolution shapes retain the existing MLX path.

## Reliability and compatibility

The existing CLI and registry aliases are preserved. Separately compiled helpers are embedded, identified by SHA-256 and verified byte-for-byte in a private user cache before execution. Corrupted or linked helpers are rejected. Bundled MLX libraries and the Metal library resolve correctly after relocating the engine.

The archive retains the four-file schema-1 runtime accepted by Desktop beta 7. Apple Silicon and **macOS 26.2 or later** are required; MLX **0.32.2** is included for models using MLX. Model weights are not included. Native Flash keeps its adaptive cache default; the explicit 3 GiB setting was used for its fixed-slot qualification. This release does not enable every experimental MTP/DFlash or Flash option.

## Validation and limits

Physical checks ran on an **Apple M5 / 24 GiB**. Five selected LFM3 GPU tests passed, including scalar/MLX primitive oracles, four independent checkpoint references, real dispatch for text and vision, and exact reset/replay. Published 4.49-bpw text and VL models passed CLI logit/state comparison with the recorded MLX baseline, image bridge and HTTP JSON/SSE rejection/recovery journeys.

Native Flash passed four independent two-token golden completions, malformed-request rejection, prefill/decode cancellation and recovery; loader evidence confirms Metal loaded and MLX absent. The actual Desktop beta-7 updater passed signed installation, relocation, execution and fallback using disposable storage.

CPU suites: 418 hybrid tests (10 ignored), 429 direct tests (108 ignored); the release MLX library passed 364 tests (319 ignored) before the final path-only fix, which has two fresh subprocess regressions and the relocated real-model journeys. Targeted Kani checks cover CPU shapes, overflow, patch/window bounds and exact architecture admission. These checks do not prove Metal, MLX, filesystem races or full inference. C++ source verification was blocked by unsupported C++20/MLX headers; CrossHair was unavailable locally.

**No new speed or memory gain is claimed.** Detailed performance campaigns were skipped at the owner's request. The published mixed VL checkpoint retains its historical independent feature discrepancy (relative L2 0.0165145, above the 0.01 qualification threshold); these kernels do not resolve that quantization/runtime gap. Broad VQA/OCR/video and an installed-app GUI image journey were not qualified. CI is running on [PR #151](https://github.com/0xZKnw/mlxl3/pull/151); an isolated packaging-fixture omission was reproduced and corrected in a test-only followup. The full remote matrix is not yet green. Signatures are ad hoc; the engine is not notarized.

## Installation

Select the beta update channel in Desktop beta 7 and apply the engine update, or unpack the archive and use `mlxl3` beside its three runtime files. Keep those files together. Engine installation does not replace the Desktop DMG or bundle. The first use extracts the selected helper into the private MLXL3 user cache.

## Artifacts

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| MLXL3-Engine-v2.0.3-beta.1-arm64.tar.gz | 73029179 | `75c59a4fae616102395004804c7d5286c6d95facc7bfb128dc7db0674ae38784` |
| MLXL3-Engine-v2.0.3-beta.1.json | 17741 | `e68f323832e669600de7bbfa074d2878b43c7bea0afd2b6b47b3d4d89da39b62` |
| MLXL3-Engine-v2.0.3-beta.1-arm64.tar.gz.sha256 | 106 | `0430e3392b8fb2e665ecd610dd32e2ae21cebaca85f8ace870533a247bd771a5` |
| MLXL3-Engine-v2.0.3-beta.1-validation.json | 22327 | `bd43dd42b474bf33ff2a9ba50b940ae7a62c5cda4f84d3ed9d83ceec0d6ee35f` |
| MLXL3-Engine-v2.0.3-beta.1-verification.tar.gz | 3233011 | `8469e4dd1441f5acfb4e0269619b5ed6091c04d3fa74c88057b87272155cd431` |

The validation JSON and verification archive preserve commands, identities, proof bounds, negative controls, earlier failures and the exact inspected results.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3 and its native Flash integration; [Liquid AI](https://huggingface.co/LiquidAI) for the LFM3 architectures; [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for EXL3 and ShortConv/attention reference semantics; [Apple MLX](https://github.com/ml-explore/mlx) for tensor execution; [Prince Canuma / MLX-VLM](https://github.com/Blaizzy/mlx-vlm) for vision preprocessing references; [MTPLX](https://github.com/youssofal/MTPLX) and [PonyExl3](https://github.com/beamivalice/PonyExl3) for the retained quantization/runtime foundations. Upstream notices, including MLX-VLM's MIT notice, are included in the manifest.
