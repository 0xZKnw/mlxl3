# MLXL3 Desktop 2.0.33

**Agent no longer stops a long task by itself: the 5-minute cut is gone, a stopped turn keeps what its tools did, and the sandbox comes back after a Stop.**

Desktop **2.0.33 / build 86** bundles MLXL3 engine **2.0.37**: the same files as [`engine-v2.0.37`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.37).

## Fixed

### Agent

- **No more "Agent has not responded for 5 minutes".** The app stopped an Agent turn after five minutes without any event from the engine, which happened while a long context was being read. That limit is removed: an Agent turn now ends only when it finishes, when you press Stop, or when the engine process ends. There is no time limit left in the app.
- **The context being read shows its progress.** With engine 2.0.37, `Reading context · N / M tokens` is also shown with Inkling 2 and its drafter, and with Qwen and its MTP head. These two paths used to stay silent for the whole read.
- **A stopped turn remembers its work.** After Stop, or after an engine error, the tool calls that had finished stay attached to the interrupted answer and are given back to the model with your next message. Before, the model only saw its visible text and had forgotten which files it had written and what its commands had returned. This needs engine 2.0.37; with an older engine the behaviour is unchanged.
- **The sandbox comes back after a Stop.** Pressing Stop while a command was running (or a call that timed out) left every later tool call answering `MCP server unavailable: sandbox` until the model was reloaded. With engine 2.0.37 the tool server restarts at the next call. It is a new process: background jobs started before the Stop are gone.

### Windows

- The Agent's 10-minute limit per turn and its 2-minute silence limit are removed.
- The same tool-server fix as on the Mac: a server lost after a stopped or failed call restarts at the next call, and an error answered by a server no longer disconnects it.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security. The Windows installer is built and attached by the `windows-desktop` workflow after publication.

## Verification and limits

- **No test suite was re-run for this release.** The checks cited here were run on the sources of [#210](https://github.com/0xZKnw/mlxl3/pull/210) before it was merged, on an Apple M5, without loading a model.
- Mac app, compiled with warnings as errors: the Studio modes check (2,460 checks, with a fake engine extended for a turn stopped or failed after a tool call) and the completion notifications check (17). The full Desktop suite was not run.
- Engine: see the [engine notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.37) (587 library tests passed, strict Clippy on three variants).
- Windows: UI tests 80/80; the Windows core tests (109 passed) were run **on the Mac only**.
- **Not tried with a real model**: a long context read with the progress text, a task resumed after Stop on Inkling 2 Mini or Qwen, the real sandbox restarting after a Stop during a command.
- **Known limit, unchanged:** if the engine has not confirmed a Stop after 3 seconds, the app still unloads the model. This can happen during a long context read; the stopped turn then keeps no tool history.

## Source and download

- Source: [`7e66b29cc`](https://github.com/0xZKnw/mlxl3/commit/7e66b29cc) (tag `v2.0.33`).
- `MLXL3-Desktop-v2.0.33-b86-Apple-Silicon.dmg`: 254,321,584 bytes, SHA-256 `a242d470e957b305c98b54bbc606c97095d3767a535f8096273f225d22145048`.
- `MLXL3-Desktop-v2.0.33-b86-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.37, the same files as [`engine-v2.0.37`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.37).
- This DMG: `hdiutil verify` valid, `codesign --deep --strict` passed on the app, version 2.0.33 / build 86, `build-info` has commit `7e66b29cc` with no tracked changes and MLX 0.32.2; the five runtime files are identical to the published engine archive. The app was not launched for this release.

## Acknowledgements

- [0xZKnw](https://github.com/0xZKnw) for the Agent fixes ([#210](https://github.com/0xZKnw/mlxl3/pull/210)).
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) for the tool design the Agent follows.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
