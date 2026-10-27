# MLXL3 Desktop 2.0.0-beta.5 — build 36

This prerelease adds Agent Studio, versioned HTML artifacts and an integrated Safetensors-to-EXL3 converter. It keeps native engine **2.0.1** and the existing Chat / Agent / Server / Lab workspaces. Stable remains unchanged.

## New features

- **Agent Studio:** import a project into an empty task. The Agent works on an isolated copy, with file browsing, editable code and diffs, private Git checkpoints, version restoration, action history and ZIP export. Review and explicitly apply changes to the originals; conflicting external edits are refused. Hidden credentials, links, dependencies and build folders are excluded from import. Project snapshots are bounded to 4,000 files / 64 MiB, with 2 MiB per file.
- **Real tool permissions:** separately enable file reading, writing and commands. Disabling writes also makes command execution read-only at the sandbox level. Commands default to 5 seconds, adjustable from 1 to 30 seconds. Network and other Mac files remain inaccessible to Agent commands.
- **Artifacts Studio in Chat and Agent:** open finished HTML, inspect or edit code, compare Git versions, restore a checkpoint, request an AI revision and export ZIP. Historical previews use CSS/JS from the same selected version. HTML retains the existing isolated WebKit runtime and supported HTTPS module CDNs, including Three.js dependencies.
- **Quantization Studio:** open Models → Quantization Studio. Choose a local model folder/Safetensors file or a Hugging Face model URL. Inspect compatibility before conversion; inference support and EXL3 conversion support appear separately. Remote sources are pinned to an immutable Hub revision and do not execute downloaded Python code.
- **Simple / Expert:** Simple offers 3/4/6 bpw presets. Expert exposes target bpw, calibration rows/context/text, excluded module prefixes, head precision and optional held-out KL/perplexity. Question-mark help explains the fields. Outputs must use a new directory, are checked before completion, and enter the model library only on request.
- **Conversion lifecycle:** unload the interactive model before starting the converter. Stop waits for the worker to exit; interrupted/failed files remain outside the library. The converter is bundled and runs only when inspecting or converting a model.

## Fixes

- Keep the latest transcript row and Markdown chunks measured as long answers grow, reducing a source of unstable lazy-layout heights in Chat and Agent. An 81-frame native viewport regression covers growing prose, code and completion. The reported intermittent disappearance was not reproduced, so this mitigation does not establish that every occurrence is fixed.
- Handle MCP tools without optional arguments and reject malformed permission state. Preserve artifact identity across AI revisions and directory-URL normalization.
- Bundle the existing consolidated PonyExl3 conversion fixes, including fixed-bit planning, routed MoE calibration, measurement diagnostics and bounded scratch-cache handling.

## Performance

No new inference-speed, RAM-saving or quality gain is claimed. Conversion uses the existing documented LFM2, LFM2-MoE and Ling adapters and Metal search. The standalone conversion runtime increases download/disk size and is inactive between jobs.

## Installation and compatibility

Apple Silicon and macOS 26.2 or later. Enable **Beta** in Settings → Updates and search, or open the DMG and drag the app into Applications. Desktop build 35 upgrades to build 36 even though the internal version remains 2.0.0. The engine remains independently versioned at 2.0.1. No installed app is replaced by building this release.

Agent mode and Git-backed artifact features require working `/usr/bin/git` (Apple Command Line Tools or Xcode). The sandbox uses macOS Seatbelt without Docker. Project import is a bounded copy, not a complete clone of dependencies or secrets; install/build commands are still constrained by tool permissions, network denial and the selected timeout.

EXL3 conversion is limited to the supplied **LFM2, LFM2-MoE and native Ling/Bailing-hybrid** adapters. Safetensors format alone does not imply compatibility; Qwen/Gemma and other inference architectures are not automatically convertible. Already quantized, incomplete or unsafe sources are refused. Large conversions can require substantial memory/disk space. Optional full-source KL/perplexity requires a supported LFM profile, a separate held-out UTF-8 text and enough memory; this bounded check scores at most 256 predictions and is unavailable for Ling.

## Verification and limitations

Focused native checks cover real Git/import/conflicts/restoration/ZIP, permissions, worker success/failure/cancellation and actual MCP/sandbox boundaries. The relevant Python suite and physical miniature MoE conversion/resume journeys were run. See [the verification report](https://github.com/0xZKnw/mlxl3/blob/main/docs/desktop-v2-beta5-verification.md) for exact counts, commands, retained failures and scope.

Swift source proof is unavailable; CrossHair was attempted but is not installed. No Rust executable code changed. Tests and finite checks are not a complete proof. Public multi-billion-parameter end-to-end conversions, every model architecture, every tooltip/gesture and a complete reproduction of the reported display bug remain unverified. Package/source hashes and actual GitHub CI status are recorded with the final release artifacts.

## Artifacts

Built from clean source [`f200014a4e68b698c0a1a1b840e3411f3f329c2c`](https://github.com/0xZKnw/mlxl3/commit/f200014a4e68b698c0a1a1b840e3411f3f329c2c). The final read-only DMG passed production updater preflight **beta4/build35 → beta5/build36** and actual frozen conversion/native-quality, source-preservation, Hub-metadata and worker-exit checks. No new performance or public-model quality claim is made. Publication before GitHub CI completion was authorized; local results do not imply a fully green CI.

| File | Bytes | SHA-256 |
|---|---:|---|
| MLXL3-Desktop-v2.0.0-beta.5-b36-Apple-Silicon.dmg | 244638706 | `a17a55575b6e99a9d8b4c6ca7b8aec8fbba854fbe2eb72d9a0a0b48952b736be` |
| MLXL3-v2.0.0-beta.5-validation.json | 2362 | `eeb376e94ca6a6c648b6da42779b182cf164a6b46eb278d3800251e14ba4f984` |
| MLXL3-v2.0.0-beta.5-verification.tar.gz | 1961505 | `85e9db598d59d001fa6373a65a4b51acef19bd5a9e74db005c767e9dcde37fc6` |
| MLXL3-Desktop-v2.0.0-beta.5-b36-Apple-Silicon.dmg.sha256 | 116 | `812a5669c2ba908a5dc3ffcc9344e96fa82924c3fbbb2afd603287e52bb8c39b` |

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3 and the feature reports; [beamivalice / PonyExl3](https://github.com/beamivalice/PonyExl3) for the EXL3 conversion implementation and calibration data; [Apple MLX](https://github.com/ml-explore/mlx) and [MLX-LM](https://github.com/ml-explore/mlx-lm) for tensor execution and model loading; [Hugging Face](https://huggingface.co/) for Hub, Safetensors and tokenizer tooling; and [SwiftMath](https://github.com/mgriebling/SwiftMath) for equation rendering. Upstream licenses and calibration-data attribution are included in the bundle.
