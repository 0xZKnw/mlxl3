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

Validation results, exact source identity and artifact checksums will be recorded here after the frozen build is inspected.

The Agent is experimental. Its shell can use macOS system tools and files placed in its private workspace; it cannot download dependencies or freely access Homebrew, user environments or arbitrary host projects. The sandbox provides process restrictions on the same macOS kernel, with no hard aggregate RAM or disk quota. It is not a virtual machine. Command compatibility and model tool-calling reliability remain dependent on the selected model. Testing is not a complete security proof.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3 and the product direction, [OpenAI Codex](https://github.com/openai/codex) for the native-sandbox reference, [Apple](https://developer.apple.com/documentation/security/app-sandbox) for macOS sandbox enforcement, and [MLX](https://github.com/ml-explore/mlx) for local Apple Silicon inference. Existing checkpoint and speculative-decoding credits remain in the preceding engine release notes.
