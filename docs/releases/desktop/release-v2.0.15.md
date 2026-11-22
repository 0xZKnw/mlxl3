# MLXL3 Desktop 2.0.15

**Agent mode can work with one JavaScript tool, the conversation no longer jumps while a long answer streams, code opens full screen, and a new menu sets the reasoning level of models that support it.**

Desktop **2.0.15 / build 68** bundles MLXL3 engine **2.0.25**.

## New

### One JavaScript tool for Agent mode

Agent mode can now give the model a single `javascript` tool instead of four separate tools. The model writes a short program that calls the project functions, so **one call can explore, edit and test**: read several files and search at once, or apply an edit and run the tests. This is the "code mode" style of recent agent harnesses ([CodeAct](https://arxiv.org/abs/2402.01030), [Codex](https://github.com/openai/codex); [Unharnessed](https://github.com/unharnessed-agent/Unharnessed) does the same with Python).

- **Functions:** `list()`, `read(path)`, `search(regex, path?)`, `write(path, text)`, `edit(path, find, replace)` (the text to find must occur exactly once) and `run(command)`. Only the functions your Sandbox permissions allow are offered.
- **Results:** the model sees its `console.log` output, the value it returns, and on failure the error with its line, or the call that failed and the calls completed before it.
- **Same sandbox as before.** The program runs in macOS's JavaScriptCore, interpreter only, inside a compute-only sandbox: it cannot open files, use the network or start processes. Every function goes back through the existing sandboxed commands, so file access, network refusal and permissions are unchanged.
- **Limits per call:** 10 seconds, 200 function calls, 32 KiB of output, files up to 256 KiB.
- **Default, and reversible.** New and existing Agent projects use the JavaScript tool. Sandbox (the lock button in Agent Studio) → **Separate tools** restores the four previous tools, with the same instructions as before.

### Reasoning level selector

Models whose chat template supports a reasoning level now show a menu in the composer, next to **Tools**:

- **Inkling 2 Mini:** No thinking, Low, Medium, High (default), Extra high, Max.
- **Qwen3.8 27B and Qwen3.8 Flash Next:** No thinking, Low, Medium, Extra high (default).
- The levels are read from each model's own template, so a model offers only what it supports. Models without levels, such as Qwen3.6, show no menu.
- The choice is kept per model. The default level sends nothing, so prompts at the default are identical to 2.0.14.
- **No thinking** really turns thinking off. The "Instruct · non-thinking" sampling preset only changes sampling.
- The menu needs engine 2.0.25 or later, bundled here.

### Full-screen code and previews

- Code blocks in conversations have a full-screen button. The full-screen window keeps the Code / Preview switch for HTML.
- The Preview tab of Agent Studio has the same button.
- Esc, ⌘W or leaving full screen returns to the conversation. While the preview is full screen, the inline preview pauses, so only one page runs.

## Fixes

### The conversation no longer jumps while an answer streams

While a long answer streamed, the view following it could jump up and down several times a second. There were two causes:

- every 50 ms batch of text started an animated scroll that showed its target for one frame, then slid back from the previous position;
- the height of older messages was re-estimated as they left the screen, which moved the whole conversation by thousands of points.

The view now stays pinned to the bottom natively, the 12 most recent messages are laid out in full, and it re-pins without animation after content or window size changes. In the check that reproduced the bug, a long answer below four long answers made the view jump up 57 to 76 times. It now makes **0 jumps**, and the view is 0 points from the bottom throughout. Scrolling up still stops the following.

The view no longer glides for 120 ms after each batch: it stays at the bottom, and the token fade is unchanged.

## Unchanged

- Everything else is as in [2.0.14](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.14).
- Conversations, settings, models and Agent projects are kept. Agent projects saved by earlier versions open with the JavaScript tool and their permissions intact.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Desktop checks** (`scripts/check-desktop.sh`, the full suite, Command Line Tools toolchain with the macOS 26.5 SDK): passed, including:
  - Agent sandbox: 73 checks on the real sandbox: the JavaScript functions, errors, exact text round trips (CRLF, quotes, Unicode), a non-UTF-8 file never rewritten, limits, permissions enforced by the host, and isolation. No JavaScriptCore shell helper is reachable; the candidate names are extracted from this Mac's `jsc` binary.
  - Agent MCP protocol: 53 checks through the real MCP helper, including cross-harness refusal, oversized code refused without ending the helper, and cleanup when the MCP parent dies.
  - Studio modes: 2,324 checks. Studios: 58. Workspace: 44. Python: 39.
  - Streaming: the new scroll-follow check (10 and 34 messages) and the existing long-visibility check (30 OCR frames, including a window resize) pass.
- The checks were also rerun after the reasoning selector was added: preferences (392 assertions, including the selector), the full response UI suite and full-screen code.
- **Mutations caught:** removing the inline-preview pause, or the oversized-code guard, fails the new checks. The scroll-follow check fails on 2.0.14.
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum;
  - after mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.15 / 68;
  - its `build-info.json` names source `9c36fa4ebbbb`, with no uncommitted changes, and engine 2.0.25;
  - the bundled engine files are byte-identical to the engine 2.0.25 archive;
  - the app binary from the DMG passed the 53 Agent MCP protocol checks, the JavaScript tool included, and its built-in chat timeline and MCP preference checks;
  - the bundled engine passed the reasoning level bridge check on a 230 MB Inkling 2 test checkpoint. The [engine 2.0.25 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.25) describe one stall that was not reproduced.
- **Not verified:**
  - The JavaScript tool was not run with a real model. There is no measurement of whether models complete tasks better with it than with the four tools. Published research is mixed for small models, which is why the four tools remain one click away.
  - Physical scroll gestures (wheel, trackpad, inertia) were not automated. The native full-screen animation was not exercised by the automated checks.
  - The bundled engine was not run on a full model: `scripts/smoke-release.py` needs one, and another model was in use on the release machine.
- **Known limits, unchanged in this release:**
  - The engine allows 5 consecutive tool rounds per answer.
  - The engine gives an MCP call 12 seconds, which is why a JavaScript call stops at 10 seconds. A separate-tools command set above about 11 seconds still times out.
  - `python3` from the Command Line Tools cannot start inside the sandbox.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. It contains none of these changes, which are in the macOS app.

## Source and download

- Compiled source `9c36fa4ebbbb254acad97c2f6c118e05db1a4afb`: `main` at `442625ec0` (engine 2.0.24) plus the engine change, the Desktop changes and the versions.
- `MLXL3-Desktop-v2.0.15-b68-Apple-Silicon.dmg` · **251,319,612 bytes**, SHA-256 `bba5d3fb3d646af37d10f7ed7929666ed7b8df543e208fcf4842d8bb4129f8ac`.
- `MLXL3-Desktop-v2.0.15-b68-Apple-Silicon.dmg.sha256` · 110 bytes, SHA-256 `6f34a4c76eff0bc39c9a0cd53e89607762057c5cd72c01e95402f6b85bd95daf`.
- Bundled engine 2.0.25 (`mlxl3` SHA-256 `e97a0892da1735860d47e596cc0a3748f325d5010b45778655be3d46a0b05714`, source `9c36fa4ebbbb`). The Swift app was built with the Command Line Tools toolchain and its macOS 26.5 SDK.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the single-tool harness idea, the scroll bug report, and the full-screen and reasoning selector requests.
- Thanks to the projects that shaped the single-tool design: [Unharnessed](https://github.com/unharnessed-agent/Unharnessed), [OpenAI Codex](https://github.com/openai/codex) code mode and the [CodeAct](https://arxiv.org/abs/2402.01030) paper.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini and to the [Qwen team](https://huggingface.co/Qwen) for Qwen3.8, whose templates define the reasoning levels.
- Thanks also to the projects behind the runtime and the app:
  - Apple JavaScriptCore, which runs the Agent programs;
  - [ExLlamaV3](https://github.com/turboderp-org/exllamav3), for the EXL3 format and quantizer;
  - [Apple MLX](https://github.com/ml-explore/mlx), for the runtime;
  - [PonyExl3](https://github.com/beamivalice/PonyExl3), for the Metal EXL3 port;
  - [SwiftMath](https://github.com/mgriebling/SwiftMath), for math rendering.
