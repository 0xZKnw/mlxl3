# MLXL3 Desktop 2.0.0-beta.21 · Refined surfaces and loading fixes

Desktop prerelease **v2.0.0-beta.21**, internal version **2.0.0 / build 52**. This is another **v2 beta**, with the published **MLX engine 2.0.9-beta.1** bundled unchanged.

## Changes

- Redesigned the menu bar widget around the active model, engine status, generation metrics and actual app memory. Its compact layout follows the Aurora theme in light and dark appearances.
- Refined the response appearance menu with neutral width controls, a clean text preview and a continuous text-size slider.
- Replaced the full-width MTP and DFlash comparison bars with compact glass controls.
- Refined shared glass surfaces into tinted native frosted material without bright white rims. New conversation, back buttons, model selection, toolbar controls and the Generation inspector now share this treatment.

## Fixes

- Switching **Chat → Agent** or **Agent → Chat** preserves an unloaded model, including when startup loading is disabled and after a manual or idle unload. Navigation does not silently retry a failed load. Explicit loading still works, and loaded models remain available across the two modes.
- Starting a new Chat from the Lab no longer creates two conversations when only Agent tasks exist.
- The menu bar popover resizes when its content changes, including longer model names. Failed engine states are identified correctly.
- Disabled controls visibly dim and do not gain active hover feedback. Their enabled state is read from the installed view when button styles are composed.
- Fixed the missing MLX package path in the macOS CI expert-route check. Mode-test failures now include the failing call and engine/MCP/server state, without extending their timeout.

## Performance

Aurora retains its existing low-frequency animation schedule and motion law. Popover backgrounds remain static; the widget retains its existing 1 Hz memory refresh. No inference throughput improvement is claimed, and model quantization, context settings and engine defaults are unchanged.

Two final runs of the optimized production window on Apple M5 measured **1.315 / 1.323% of one CPU core at idle**, **2.572 / 1.865% during simulated generation**, and **0.009 / 0.018% while hidden**. The rendering process used approximately **54–68 MiB** without an engine or model loaded. Each state had 3 seconds of settling and 7 seconds of sampling, on battery with nominal thermals. These are window-process measurements, with no synchronized baseline, throughput, attributable GPU or power claim. Both runs and their variation are preserved in the [measurement record](measurements/desktop-matte-20261113.json).

## Installation and compatibility

Requires **Apple Silicon and macOS 26.2 or later**. Select the **Desktop beta channel** in Settings, or open the DMG and drag MLXL3 Desktop to Applications. The application is ad-hoc signed and is not notarized by Apple. App and engine update channels remain independent; this prerelease does not replace the latest stable Desktop release.

## Validation and limits

- Python: **2,828 tests passed, 106 skipped, 155 subtests passed**.
- Desktop: real startup/manual loading and bidirectional mode navigation, idle unload, Agent workspaces and authenticated local server lifecycle; **404 preference assertions**, **120 idle-unload assertions** and **2,324 mode assertions** passed. Widget sizing, disabled-control pixels, light/dark settings surfaces and compact layouts were checked against rendered native views.
- Additional Desktop checks cover persistence/recovery, downloads, MTP/DFlash, imports, images, incremental rendering, response visibility, updater boundaries and Studios. A style edit invalidated a compilation during the first pass; final visual checks and the remaining production-script checks were run on frozen sources. The detailed review preserves the interrupted pass and the resumed scope.
- Rust CPU library: **434 passed, 10 ignored**. MLX-feature library: **461 passed, 241 ignored**. Strict Clippy and formatting passed for the checked configurations. C++ capability, cache-key and disk-reader checks passed with ASan/UBSan.
- Physical Apple M5 checks: LFM3 vision primitives against scalar oracles, Q4/F16/BF16 error recovery, and the bundled bridge on LFM3-1.7B 8 bpw and Qwen3.6-35B-A3B 2.49 bpw. Both bridges generated the requested **1/2/8-token** outputs, rejected a zero budget and exited cleanly. These short checks do not establish general model-quality or reference-logit parity.
- Kani verified the CPU token-total saturation property over symbolic `u64` inputs, **5 obligations passed**. A GQA harness exceeded the local verification budget; an existing remote grouped-route proof timed out. These properties remain unverified. Compiler checking and sampled rendering tests are not a formal proof of SwiftUI behavior.
- The base commit's hosted Desktop mode timeout was not reproduced locally. Its diagnostic was improved. Hosted results for the published commit must be read separately from local checks; see [GitHub Actions](https://github.com/0xZKnw/mlxl3/actions).

See the [detailed beta review](desktop-v2-beta-review-20261113.md) for commands, the split Desktop run, negative attempts and remaining limits. Physical Gemma/Ling models, full vision inference with image inputs, general numerical parity, GPU power and all context lengths are outside this campaign. The reduced-transparency fallback is implemented; a test-only environment injection was rejected by the SDK's read-only API, and system accessibility preferences were left untouched.

## Artifact and source

Verified artifact metadata is added after packaging. The bundled runtime is [engine-v2.0.9-beta.1](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.9-beta.1); no separate engine release is part of this update.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the design feedback, bidirectional loading report and original Aurora reference in [Locked](https://github.com/0xZKnw/locked). Thanks to [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, model format and math-rendering stack.
