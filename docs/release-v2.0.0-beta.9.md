# MLXL3 Desktop 2.0.0-beta.9

Desktop prerelease **v2.0.0-beta.9**, internal version **2.0.0 / build 40**. Desktop source: **32d00bfc56c5e061b162a0593e6c8d573a88cf6b**. Bundled engine **2.0.4**, source **ead60d0323e99bc023c82485dddc89d9903e4ead**.

## Table fixes

- Horizontal borders now span the complete row at the tallest cell's bottom. Vertical borders span the table height, keeping short labels and long details in the same aligned row.
- Tables expand to fill the available conversation width, including wide mode. Columns still have a readable minimum width scaled with the text-size setting, and scroll horizontally when space is limited.
- The beta 8 fix for `<br>` / `<br/>` / `<br />` remains included. Wrapped content, Markdown column alignment, inline code, formulas and copied/stored Markdown keep their behavior.

## Engine and memory

This app bundles the exact independently published [engine 2.0.4-beta.1](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.4-beta.1). LFM3 text/VL now use a default 512 MiB allocator cache instead of retaining unused decode buffers throughout a response. Explicit allocator overrides remain supported. Native Flash Next and other model backends are retained.

On M5 / 24 GiB, LFM3-1.7B EXL3 8bpw completed a synthetic 6,835-token prompt plus 1,024 output tokens at **3.38 GB peak process footprint**. The old default crossed a 10 GiB safety guard and was cancelled. The fixed output matches the 512 MiB control exactly. These results do not establish a universal RAM bound or a speed gain; see the engine release for protocol and limits.

## Installation and compatibility

Apple Silicon, macOS **26.2+**. Replace the application from the DMG, or update through the beta channel. The bundled 2.0.4 runtime takes precedence over an older external 2.0.3 engine; a newer compatible external engine still takes precedence. All 3,050 converter entries match beta 8. This task did not replace the installed app or change the user's engine pointer.

## Validation and limits

32 native bitmap pairs pass across four widths (360–2,000 points), four text scales and streaming/final states, plus 15 parser boundaries. The Retina-aware border guard detects the old staggered rows and passes on the corrected captures; CPU OCR confirms the first/last cell line and following row remain visible. Wide captures were visually inspected.

Swift release compilation and Swift 6 warnings-as-errors/SIL checks pass. The mounted DMG passes runtime hashes, arm64/signature checks and two app self-checks. Fifteen updater cases pass, including the actual build39→40 image preflight. Earlier harness invocation/scaling failures are retained in the evidence archive. The full unrelated suites were omitted at the user's request; no PR-triggered remote run was found for this exact source. SIL is not a behavioral SwiftUI proof, and the engine Kani attempt failed with a compiler internal error. Signatures are ad hoc; the app is not notarized.

## Artifacts

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| MLXL3-Desktop-v2.0.0-beta.9-b40-Apple-Silicon.dmg | 251135186 | `d06b29f85627737b34df6792606b1bd2e313fec5bea3f657776e23568d3cddd4` |
| MLXL3-Desktop-v2.0.0-beta.9-b40-Apple-Silicon.dmg.sha256 | 116 | `60254252f4247e46466f0650de189c781d5b488349a45e06172441ec02841163` |
| MLXL3-v2.0.0-beta.9-validation.json | 2215 | `1cbdc91c2f7c3a0cf8e46752f4a04945bf0160631ce6240c55e887197dd812a5` |
| MLXL3-v2.0.0-beta.9-verification.tar.gz | 212502 | `eb5f66ed9d2de0479a8e60a0fb4f9a74bd2d3f0d3a2ee9cd848c75e4d19ac5a7` |

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the alignment, wide-layout and RAM reports and fixes; [HENK0O](https://github.com/HENK0O) for retained Desktop appearance and reading controls; [Apple MLX](https://github.com/ml-explore/mlx), [LiquidAI](https://huggingface.co/LiquidAI) and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the retained engine foundations; [PonyExl3](https://github.com/beamivalice/PonyExl3) / [MTPLX](https://github.com/youssofal/mtplx) for retained conversion/MTP work; and [SwiftMath](https://github.com/mgriebling/SwiftMath) for equations. Existing notices remain included.
