# MLXL3 Desktop 2.0.0-beta.8

Desktop prerelease **v2.0.0-beta.8**, internal version **2.0.0 / build 39**. Source: **6c2cb16e4c358d21a297d23c77e37269dcc4e86f**. This advances Desktop beta 7 / build 38.

## Fixes

- Markdown table cells now render `<br>`, `<br/>`, `<br />` and case/spacing variants as line breaks, including column headers.
- Wrapped and multiline cells use their full measured height; long details no longer overlap or hide the next row. Columns use the existing 260-point maximum as a definite width, scaled with the text-size setting; horizontal scrolling remains available.
- Inline code, escaped tags and formulas retain literal markup. Stored/copied Markdown and prose outside tables remain unchanged.

## Compatibility and installation

Apple Silicon, macOS **26.2 or later**. Download the DMG and replace the Desktop application, or choose the beta channel in the updater. The signed read-only image passed the actual build38→39 updater preflight.

The bundled engine **2.0.2** and all converter payloads are byte-identical to beta 7. An independently installed engine **2.0.3** remains selected; this Desktop update does not overwrite or downgrade it. LFM3 memory use is being investigated separately: **this release contains no engine memory fix**.

## Validation and limits

24 nonempty native table bitmap comparisons pass across three widths, four font scales and streaming/final states. CPU-only OCR confirms the first and last cell lines and the following row are visible. Fifteen parser boundaries preserve code/math/Unicode and reject malformed or incomplete break tags. The complete affected Response UI suite passes, including 786 fade cases, 63 drawing comparisons, 81 viewport frames, 63 parser/worker parity cases, formulas, previews, clipboard/history and bridge/cancellation journeys.

The Swift release build, Swift 6/SIL checks, 94 packaging helper tests, 15 updater cases, arm64/deep-strict signing, both mounted self-checks and runtime hashes pass. Engine/converter parity covers five runtime entries and 3,050 converter entries. Earlier failing rendering prototypes and the read-only mount-conflict retry are preserved in the evidence archive.

There is no behavioral SwiftUI formal proof; SIL verifies compiler invariants. Remote CI for this exact Desktop source is not complete. Signatures are ad hoc, and the app is not notarized. No new speed or memory gain is claimed. The installed app was not replaced by this task.

## Artifacts

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| MLXL3-Desktop-v2.0.0-beta.8-b39-Apple-Silicon.dmg | 244495828 | `d8fbab60e9d39ca608d153a77bb0768f908e2d9ea832baa2e4169339c9763b70` |
| MLXL3-Desktop-v2.0.0-beta.8-b39-Apple-Silicon.dmg.sha256 | 116 | `2b9e8300c225e959a2747c2c65f07ad6f113fe8b93fde52e553e9238ec689052` |
| MLXL3-v2.0.0-beta.8-validation.json | 2507 | `c4adc7552f9bfcf7861786513cded40bf545a8a44095d0e207c91c0e68bfa299` |
| MLXL3-v2.0.0-beta.8-verification.tar.gz | 114073 | `d9fe02667831899b9b7b9e40984a907fcc39fcb4cb49995ee85790bfa82fad05` |

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3 and this table-rendering correction; [Lucas Bauduin / HENK0O](https://github.com/HENK0O) for the retained appearance, startup, pinned-chat and reading controls; [Apple MLX](https://github.com/ml-explore/mlx) for the retained engine runtime; [PonyExl3](https://github.com/beamivalice/PonyExl3) and [MTPLX](https://github.com/youssofal/mtplx) for retained conversion/MTP foundations; and [SwiftMath](https://github.com/mgriebling/SwiftMath) for equation rendering. Existing notices remain included.
