# MLXL3 Engine 2.0.33

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. One change: **Inkling 2 Mini's speculative decoding (DFlash2, or DSpark when asked) now also runs when the request carries tools** — MCP tools in chat and Agent mode.

## Changed

- Until 2.0.32 a request with tools always decoded plainly, whatever `dspark` said. The speculative session now runs in tool rounds too, as the prompt lookup already did: every token still goes through the same stream filter and the same stops, and the verification rules are unchanged, so tool calls and the sampling law are not affected.
- Nothing else changes: `"dspark": false` still turns speculation off, an omitted field follows the DFlash2 drafter when it is installed.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security. The bridge protocol is unchanged (1). [Desktop 2.0.29](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.29) bundles this engine.

## Verification and limits

- **One real tool round** on the installed Inkling 2 Mini with the Desktop sandbox server, speculation by default: the model called `sandbox.javascript` (`return list().length`), got `3` and answered "The project has 3 files"; 16 speculative blocks ran in that request. A plain prompt with MCP enabled ran 39 speculative blocks, and the same prompt with `"dspark": false` ran none.
- `cargo fmt --check` passed. **The test suites and Clippy were not run for this release** (one condition changed since 2.0.32, published quickly at the maintainer's request); CI runs them on `main`.
- **Not verified:** long Agent tasks with many tool rounds, tool calls at temperature > 0 beyond this run, speed with tools (not measured), Kani.

## Artifacts

ARTIFACTS_PLACEHOLDER

## Thanks

- Thanks to [0xZKnw](https://github.com/0xZKnw) for spotting, with screenshots, that the drafter stayed idle as soon as tools were enabled.
- [IncoAI](https://huggingface.co/incoai) for the [Inkling 2 Mini DFlash2 drafter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2). Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [minijinja](https://github.com/mitsuhiko/minijinja).
