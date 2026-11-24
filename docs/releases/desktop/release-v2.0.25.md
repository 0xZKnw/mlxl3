# MLXL3 Desktop 2.0.25

**Agent mode can write a whole program in one call, and a refused call says why.**

Desktop **2.0.25 / build 78** bundles MLXL3 engine **2.0.31**: the same files as [`engine-v2.0.31`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.31).

## Fixes

### A whole program in one call

Until 2.0.24, the code of one `javascript` call was limited to 32 KiB. A real Agent task hit this limit. The model spent 20 minutes writing a whole Python game in one call: a step of about 45,000 generated tokens, thinking included. It got "JavaScript sandbox refused or unavailable.", nothing was written, and no reason was given.

- **Code of one call:** up to **1 MiB** (was 32 KiB). That is more than 250,000 tokens, so a model does not reach it in one step.
- **`write()` and `read()`:** up to **1 MiB** per file (was 256 KiB).
- **`run()` commands:** up to **256 KiB** (was 16 KiB), enough for a long heredoc.
- **MCP server:** the server and the JavaScript kernel accept requests of that size. Before, a request over 192 KiB ended the MCP server.
- **`list()`:** unchanged. It still stops at 256 KiB of paths and says so.

### Errors that say what happened

- **Code over the limit:** the model now reads:
  - the size of its code and the limit;
  - that nothing was run;
  - how to write a long file in parts.
- **Other failures:** the model reads that the JavaScript sandbox stopped or could not start, and that it restarts on the next call. It is told to check what the call did before running it again. Until 2.0.24, every failure got the same message.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The Windows app is unchanged in this release.

## Unchanged

- Models, conversations, projects, Agent projects and settings.
- The engine, the Agent sandbox's permissions and everything else are as in [2.0.24](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.24).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Agent sandbox checks:** 346 passed.
  - They include a 175 KB Python program written in one call through the JavaScript kernel.
  - They check code of 1 MiB + 1 byte, refused with its size and "Nothing was run".
  - They check that a write over 1 MiB is refused.
- **MCP protocol checks:** 77 passed (73 in 2.0.24), on the real MCP server:
  - code of exactly 1 MiB runs;
  - code over 1 MiB is refused with its size, without ending the server, for accented text, plain text and control characters;
  - a 90 KB Python program, accents and quotes included, is written in one call;
  - 1 MiB of accented text is written and read back.
- **Mutations:** each old limit and the old message were restored temporarily, 5 changes in all, and the checks detected every one. With the old 32 KiB limit, the 1 MiB and whole-program checks fail.
- **Studios checks:** 60 passed.
  - One check still expected the sandbox from before 2.0.23, where a command could not read the Agent task's metadata. The 2.0.24 notes list it as failing.
  - It now checks that commands can read that metadata but cannot change or remove it.
- **CI:** the reviewed inventory of MLX call sites now includes the Agent loop and the 2.0.24 interface text. These new sites are iterator calls, `cfg` gates and text that mentions "mlx", not MLX calls. This inventory made `main`'s regression checks fail since 2.0.23.
- **This DMG:**
  - `hdiutil verify` valid, `codesign --deep --strict` passed;
  - version 2.0.25 / build 78, arm64, built with the macOS 26.5 SDK;
  - `build-info` has commit `7866a2f7e` with no tracked changes;
  - the bundled engine reports 2.0.31 (`engine-v2.0.31`, commit `e117e02e8f7b`), and its five runtime files are identical to the published engine archive;
  - the 77 MCP protocol checks passed against this DMG's app executable, which is the server the app ships.
- **The failing call, reproduced:** one call writes a Python file of 114,780 bytes.
  - The 2.0.24 app answers "JavaScript sandbox refused or unavailable." and writes nothing.
  - This DMG's app writes the file.
  - The 2.0.24 app also fails the new protocol checks: a request over 192 KiB ends its MCP server.
- **Not verified:**
  - the game task run again on a real model;
  - the time and memory of a call near 1 MiB with a real model (not measured);
  - the full Desktop suite (`scripts/check-desktop.sh`), which was not run (`scripts/check-studios.sh` was);
  - the app driven by hand.

## Source and download

- Source: [`7866a2f7e`](https://github.com/0xZKnw/mlxl3/commit/7866a2f7e2a903a5241c44d87d67cabb5710cf8d) (tag `v2.0.25`).
- `MLXL3-Desktop-v2.0.25-b78-Apple-Silicon.dmg`: 255,299,611 bytes, SHA-256 `59b12693a222f5a70ffb67f84c6391a121185fe7f32fa2b7faaf54db9e9d73ff`.
- `MLXL3-Desktop-v2.0.25-b78-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.31, the same files as [`engine-v2.0.31`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.31).

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for reporting the refused call with screenshots of the task, and for asking for the limit to go.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
