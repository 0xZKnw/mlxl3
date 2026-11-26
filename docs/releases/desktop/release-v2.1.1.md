# MLXL3 Desktop 2.1.1

**The line counting failed and succeeded commands under an Agent answer is gone.**

Desktop **2.1.1 / build 89** bundles MLXL3 engine **2.0.38**, rebuilt from this source. The Rust sources are identical to [`engine-v2.0.38`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.38); the binary is not byte-identical to that archive because it records another source commit. The engine itself is not republished.

## Changed

- **No more `Commands · 1 failed · 4 succeeded` under an answer.** The same summary for test commands (`Test commands · …`) is removed too. Each command and its result is still shown in the steps of the turn.
- The **Review N files** button and the turn's document stay, when the turn changed files. A turn that only ran commands now shows nothing in that place.

Everything else is [Desktop 2.1.0](https://github.com/0xZKnw/mlxl3/releases/tag/v2.1.0): one conversation mode, summarized conversations, and the confirmation before deleting in your folder.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

**Windows:** the Windows app never had this line; its installer for this tag has the same content as 2.1.0.

## Verification and limits

_To be completed with the build._

## Source and download

_To be completed with the build._

## Acknowledgements

- [0xZKnw](https://github.com/0xZKnw) for the request and the change.
- [HENK0O](https://github.com/HENK0O) for the per-turn review this row belongs to ([#204](https://github.com/0xZKnw/mlxl3/pull/204)).
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
