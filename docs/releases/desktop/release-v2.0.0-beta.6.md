# MLXL3 Desktop 2.0.0-beta.6 — build 37

This prerelease replaces Artifacts Studio with code and HTML previews integrated into Chat and Agent, rebuilds Agent Studio around a full-window workspace, and bundles native engine **2.0.2**. Desktop's internal version remains **2.0.0**, now **build 37**. Release source: **9df28b6d2a10e0d646f8a36532ae69c88c9ec3e8**

## Features

- **Code in the conversation:** completed code blocks use a native, syntax-colored viewer with line numbers, readable spacing and a Copy action. Streaming responses retain their incremental rendering. There is no separate Artifacts Studio to navigate through.
- **HTML in Chat and Agent:** use the block's Code/Preview control to run completed HTML inside the app. Execution starts on request. Returning to Code or covering the conversation with Agent Studio releases the preview. Supported HTTPS dependency CDNs and the existing isolated HTML runtime remain available.
- **Agent Studio:** browse the isolated project in a full-window workspace with compact file navigation and consistent Code, Diff and History pages. Code has syntax colors and line numbers; Diff distinguishes additions and removals; History opens the selected project version. HTML file shortcuts open the relevant project preview.
- **Project controls:** save edits, review changes, restore checkpoints and export a selected version as ZIP. Opening another project creates a new Agent task, preserving the previous one. Tool permissions remain available from the Sandbox control.

## Fixes

- Fix long-response rendering across thinking, code and normal text: bound raw thinking chunks, keep stable text on native drawing and preserve the code scroller’s initial/following anchor. The former code-tail failure is covered by 30 native bitmap/OCR checks.
- Remove the oversized sheet margins and keep the file title separate from the editor's line-number gutter.
- Open text files as code instead of showing an empty HTML preview. Handle binary files and files absent from a historical version explicitly.
- Protect unsaved drafts during file/version navigation and closing. Refuse a save if another writer changed the file; changing permissions does not discard the draft or create an unrelated checkpoint. Ignore stale file-load results.
- Apply the exact reviewed project revision. Unreviewed workspace changes and conflicting edits to the original project are refused before application.
- Preserve snapshot and ZIP bytes despite project Git attributes, including CRLF text. Treat filenames literally in scoped diffs, including names containing Git pathspec characters.
- Include main's bounded Desktop bridge framing and its permission-envelope regression checks.

## Performance

The bundled engine includes the already qualified Cold expert overlap, dynamic SSD readers and MTP prefill work from main [`4c1d5ecffbd0a04e14385db5c286b947a634f73a`](https://github.com/0xZKnw/mlxl3/commit/4c1d5ecffbd0a04e14385db5c286b947a634f73a). See the separate [engine notes](https://github.com/0xZKnw/mlxl3/blob/main/docs/releases/engine/release-engine-v2.0.2-beta.1.md) for the measured configurations, prefill tradeoffs and disabled experiments.

No fresh GPU benchmark or general speed/RAM claim accompanies this DMG. Historical gains depend on the model and profile; several flags used by those benchmarks remain off by default. The updater and quantization workflow receive no new behavior in this release.

## Installation and compatibility

Apple Silicon and macOS **26.2 or later**. Select **Beta** in Settings → Updates and search, or download the DMG and drag MLXL3 Desktop into Applications. Build **36 → 37** advances the Desktop updater while the internal app version stays 2.0.0. The standalone engine prerelease is `engine-v2.0.2-beta.1`, with internal runtime version **2.0.2**

Agent Studio requires working `/usr/bin/git` from Apple Command Line Tools or Xcode. Agent commands continue to use macOS Seatbelt without Docker, with per-tool permissions, offline execution and the selected command timeout. Project import remains a bounded isolated copy; it excludes hidden credentials, links, dependencies and build folders.

The existing bundled converter and its compatibility limits are unchanged. Safetensors format alone does not make an architecture convertible to EXL3.

## Verification and limitations

Focused CPU/native checks exercise real project imports, save conflicts, dirty navigation, scoped/all-file diffs, historical files, reviewed application, byte-exact snapshots/ZIP and full-window integration. Native captures cover Code, Diff, History and the integrated response-code presentation. Final counts, commands and retained failures: **[CPU/native verification report](https://github.com/0xZKnw/mlxl3/blob/main/docs/desktop/desktop-beta6-studio-verification.md)**

Package signature/provenance, read-only DMG and production updater checks: **Passed on the actual signed read-only DMG: source/build/runtime hashes, arm64/signatures and production updater build36→37; standalone engine installation/activation2.0.1→2.0.2 and bad-digest rejection also passed in disposable storage**. Exact-source GitHub CI status: **At the publication snapshot, Flash profile compatibility and EXL3 conversion workflows passed; Regression/Desktop and Native workflows were still running, and Apple compatibility was cancelled on source `9df28b6d2a10e0d646f8a36532ae69c88c9ec3e8`. Publication and merge before CI completion were explicitly authorized.** Publication before CI completion does not imply a green CI.

No model inference, conversion, Metal/WebGL execution or fresh physical engine benchmark was run during this release preparation, as requested. Interactive HTML/CDN execution and the rebuilt engine's physical numerical behavior remain unverified here. Earlier GPU qualifications retain their original source identities. Swift source proof is unavailable; historical engine Kani/CBMC limitations remain documented. Tests and bounded checks are not a complete proof, and the earlier intermittent transcript-disappearance report is not claimed universally resolved.

## Artifacts

Source: **9df28b6d2a10e0d646f8a36532ae69c88c9ec3e8**. Desktop and bundled engine provenance: **[validation receipt](https://github.com/0xZKnw/mlxl3/releases/download/v2.0.0-beta.6/MLXL3-v2.0.0-beta.6-validation.json)**

| File | Bytes | SHA-256 |
|---|---:|---|
| MLXL3-Desktop-v2.0.0-beta.6-b37-Apple-Silicon.dmg | 238239552 | `234ba954e167b7cce6e0eed7464c2f798cb960f94aaaf10c0893d03945fc7070` |
| MLXL3-Desktop-v2.0.0-beta.6-b37-Apple-Silicon.dmg.sha256 | 116 | `b4e17c6d85bac739fa2ada3c16336b52a0f6865df41e6422cf39ae18419d0782` |
| MLXL3-v2.0.0-beta.6-validation.json | 2284 | `fa79fb5a87eb75214dbf4e94e7f5ee18cdfa346821ba7fae92977b69e130f93d` |
| MLXL3-v2.0.0-beta.6-verification.tar.gz | 16627814 | `2fafeba574558c679f6de8f6ffafb2d2bf4d6a57c7c1d8af5d40696f1ea2b71b` |

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the feature reports and engine integration in [PR64](https://github.com/0xZKnw/mlxl3/pull/64), [PR68](https://github.com/0xZKnw/mlxl3/pull/68) and [PR69](https://github.com/0xZKnw/mlxl3/pull/69); [Apple MLX](https://github.com/ml-explore/mlx) for the native tensor runtime; [beamivalice / PonyExl3](https://github.com/beamivalice/PonyExl3) for the already bundled EXL3 converter; [Hugging Face](https://huggingface.co/) for Hub and Safetensors tooling; and [SwiftMath](https://github.com/mgriebling/SwiftMath) for equation rendering. Existing upstream licenses and notices remain in the bundle.
