# MLXL3 Engine 2.0.35

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. One fix: **Inkling tool calls whose JSON name was only the start of the tool name are read again**, instead of being refused one after the other.

## Fixed

- **Inkling 2 Mini, tools and Agent mode.** The model sometimes wrote the right tool name before the call (`sandbox.javascript`) and only its first part in the call's JSON (`"name":"sandbox"`). The engine refused the call with "Inkling tool call name differs from its header", the model repeated the same call, and the turn went nowhere. When the JSON name is a whole leading part of the name written before the call, that name now designates the tool and the call runs.
- **A bare tool name** that a single offered tool carries (`javascript` for `sandbox.javascript`) now runs that tool. It used to be answered with "There is no tool named `javascript`". With two tools of that name, nothing is chosen and the model still gets the list.
- When the two names really differ, the call is still refused, and the message now says what differs: the "name" in the call's JSON and the tool name written before the call.

Nothing else changes since 2.0.34.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security. The bridge protocol is unchanged (1). [Desktop 2.0.31](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.31) bundles this engine.

## Verification and limits

- The refused call was taken as written from the conversation where it happened and is now a regression test. `cargo test` (feature `chat`): 573 passed, 0 failed, 13 ignored. Clippy with `-D warnings` passed for `chat` and `mlx,chat`. Five temporary mutants of the fix were all caught, including the parser as it was before.
- On the installed Inkling 2 Mini (3.33 bpw), a build with this fix read 12 real tool calls out of 12 through the bridge with the Desktop sandbox server.
- **Not explained:** why the model shortened the name. It did not happen again in 28 real calls (temperature 0.7 and 0), so the cause is not established; the report was at temperature 1 without top-p or top-k. This release makes that slip harmless; it does not prevent it.
- **Not verified:** the app's own system prompt and web tools at temperature 1, Kani, the `direct-metal` build.

## Artifacts

| File | Size | SHA-256 |
|---|---|---|
| `MLXL3-Engine-v2.0.35-arm64.tar.gz` | 69,298,589 bytes | `6202179127f3ccf8edb8ee4221a3100a7856643911534b2973b76d5a40b1c37d` |
| `MLXL3-Engine-v2.0.35-arm64.tar.gz.sha256` | 100 bytes | |

Source: `a0d6a2e66` (tag `engine-v2.0.35`). The archive was extracted into a new directory: `mlxl3` passes a strict signature check and is arm64, its manifest reports 2.0.35 and `engine-v2.0.35`, and its five files are identical to those bundled in Desktop 2.0.31. The extracted binary, with no environment variable, then ran two Agent requests at temperature 1 on the installed Inkling 2 Mini: the bridge reported 2.0.35 from `a0d6a2e66`, and both `sandbox.javascript` calls were read and executed.

## Thanks

- [0xZKnw](https://github.com/0xZKnw) for the report with screenshots of the refused calls and of the model's reasoning about them.
- Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [minijinja](https://github.com/mitsuhiko/minijinja).
