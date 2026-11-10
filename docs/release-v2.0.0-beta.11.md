# MLXL3 Desktop 2.0.0-beta.11

Desktop prerelease **v2.0.0-beta.11**, internal version **2.0.0 / build 42**.
Bundled engine **2.0.5**, also published independently as
**engine-v2.0.5-beta.1**. Exact source and artifact identities are recorded in
the attached validation report.

## Changes

- Attach PNG, JPEG and WebP images to a loaded LFM3-VL model through the file
  picker or drag and drop. The attachment preview displays the image; local
  history keeps its original bytes, including after the source file is removed.
- Documents and images can be sent together. Images use the engine's dedicated
  image input rather than being inserted as base64 text in the prompt.
- LFM3-1.7B Heretic NX and the original LFM3-1.7B are recommended in Discover.
  Both open immutable repository revisions and explicitly select the validated
  **8bpw** variant. Existing recommendations remain available.
- Includes the already merged [PR #163](https://github.com/0xZKnw/mlxl3/pull/163)
  safeguards for checkpoint identity/recovery, download phases, retained chat
  drafts, runtime contracts, structured responses and bounded background work.
  This release is built from current main plus the LFM3 changes, rather than
  rebundling the previous beta's source.

## Fixes

- The Hub correctly recognizes `lfm3` and `lfm3_vl` as supported architectures.
- Image conversations cannot silently be sent to a text-only model or an old
  engine without image support. Pending attachments survive model switching.
- Corrupt, unsupported and oversized images are rejected before sending.
  Limits: eight images / 8 MiB per conversation, PNG/JPEG/WebP, 64 million
  source pixels per image. The engine also enforces its resized-pixel budget.

## Installation and compatibility

Apple Silicon, macOS **26.2+**. Use the Desktop beta update channel or the DMG.
The installed application is not replaced by publishing this release.
Signatures are ad hoc; the build is not notarized.

## Validation and limits

Exact test, source-verification, physical-model and packaging results are
recorded in the attached report. No new inference optimization or speed gain
is claimed. Color controls establish the image input path, not general VQA,
OCR or video quality. The Heretic NX 4-bpw candidate was not published because
it failed its accepted KL ceiling.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the requested LFM3 workflow
and model exports; [LiquidAI](https://huggingface.co/LiquidAI),
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) and
[Apple MLX](https://github.com/ml-explore/mlx) for the existing inference
foundations; [SwiftMath](https://github.com/mgriebling/SwiftMath),
[PonyExl3](https://github.com/beamivalice/PonyExl3) and
[MTPLX](https://github.com/youssofal/mtplx) for retained components. Existing
upstream notices remain included.
