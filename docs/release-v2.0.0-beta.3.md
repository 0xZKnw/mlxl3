# MLXL3 Desktop 2.0.0-beta.3 — build 34

This prerelease adds four workspaces above the conversation list: **Chat, Agent, Server and Lab**. It keeps MLXL3's existing visual identity and conversation storage. The stable update channel remains unchanged.

## New features

- **Agent:** a centered MLXL3 welcome screen and composer, a private folder for each task, persistent task history, and a working Finder shortcut. The local model can inspect, create and edit files through a built-in sandbox tool. Branches retain their Agent mode. Moving between Agent tasks replaces the tool connection while keeping the model resident.
- **Native macOS sandbox:** commands run through Seatbelt without Docker or a resident virtual machine. Personal file contents, networking and private IPC are denied. Tool configuration stays outside the writable task folder, and the Agent does not inherit the user's Chat MCP servers. Commands have a five-second deadline, a 32 KiB output limit, CPU/file/descriptor/process limits, and supervised descendant cleanup. Missing sandbox support fails closed.
- **Server:** start and stop the existing OpenAI-compatible API from Desktop, choose a local port, copy its base URL and API key, and inspect its logs. The service binds to `127.0.0.1`, requires a fresh key generated for each start, and takes exclusive ownership of the model; returning to Chat or Agent stops the service before loading the interactive engine.
- **Lab:** inspect the Mac/CPU/Metal GPU, total RAM, recommended GPU working set, macOS/runtime versions, UI and interactive-engine memory, thermal state and low-power mode. Completed Chat and Agent generations retain their model, quantization, settings, token counts, cached tokens, decode/prefill throughput, TTFT and available MLX/process memory metrics. Filter by model and export the recorded samples as JSON.

## Performance

This update introduces no inference kernel change and claims no new throughput or RAM improvement. The native sandbox avoids a continuously running VM by construction; its memory savings have not been benchmarked. Lab contains observations from actual generations, with their settings, rather than a controlled model ranking. Server requests are not included in Lab history.

## Installation and compatibility

Apple Silicon and macOS 26.2 or later. The application uses ad-hoc signing and is not notarized. Select **Beta** in Settings → Updates to receive this prerelease, or install its DMG. The existing models remain outside the application bundle. No standalone engine release is introduced by this Desktop update.

Old conversations open in Chat. New Agent and Lab metadata is optional, but older Desktop versions may discard it when saving; back up conversation storage before downgrading. Deleting an Agent conversation keeps its workspace files on disk.

## Verification and limitations

Built from executable source [`45697feb06cac08983683c9ad5c1b8afdb8e9d32`](https://github.com/0xZKnw/mlxl3/commit/45697feb06cac08983683c9ad5c1b8afdb8e9d32); bundle and native runtime identities match, without tracked modifications. The full Desktop regression passed locally before the final logo-only adjustment; final strict compilation and 1158 functional/pixel checks passed after it. The sandbox and protocol harnesses passed 24 and 17 checks, the bounded transport passed 27 tests, and the sandbox passed AddressSanitizer. A deliberately weakened policy was rejected by the file-access oracle. These are tests and finite checks, not a complete proof; a Swift source-level proof was unavailable.

The packaged application completed a real Qwen3.8 27B tool/file journey and an authenticated Server request on an Apple M5. Eleven packaging checks and the actual DMG updater preflight passed, including 15 signed compatibility/integrity cases. Intermediate failures and rejected logo integrations remain in the [verification archive](https://github.com/0xZKnw/mlxl3/releases/download/v2.0.0-beta.3/MLXL3-v2.0.0-beta.3-verification.tar.gz). **Published at the user’s request before GitHub CI completes.** On source `45697feb06cac08983683c9ad5c1b8afdb8e9d32`, conversion contracts passed; Desktop regression and Native checks were still running at publication preparation. The unchanged base’s earlier global Kani job ended on runner shutdown. CI is not claimed fully green.

## Artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| MLXL3-Desktop-v2.0.0-beta.3-b34-Apple-Silicon.dmg | 74120917 | `5a8110b9fb02f358938e588d68ea0cd0badabc372939f4daf151414b937bdb30` |
| MLXL3-v2.0.0-beta.3-validation.json | 1853 | `0a1a80872b62265c2847f2abb9734c951495be9e73fe51fd157783110d87d812` |
| MLXL3-v2.0.0-beta.3-verification.tar.gz | 699946 | `468e6381290d6445654bc2ac013861ac0d7637f60df292a237e2ab34aeacf48e` |
| MLXL3-Desktop-v2.0.0-beta.3-b34-Apple-Silicon.dmg.sha256 | 116 | `3c5b34554036024b381767b173c9439058dcf3e69ef8ca4f476d0ec75ca16b5a` |

The Agent is experimental. Its shell can use macOS system tools and files placed in its private workspace; it cannot download dependencies or freely access Homebrew, user environments or arbitrary host projects. The sandbox provides process restrictions on the same macOS kernel, with no hard aggregate RAM or disk quota. It is not a virtual machine. Command compatibility and model tool-calling reliability remain dependent on the selected model. Testing is not a complete security proof.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3 and the product direction, [OpenAI Codex](https://github.com/openai/codex) for the native-sandbox reference, [Apple](https://developer.apple.com/documentation/security/app-sandbox) for macOS sandbox enforcement, and [MLX](https://github.com/ml-explore/mlx) for local Apple Silicon inference. Existing checkpoint and speculative-decoding credits remain in the preceding engine release notes.
