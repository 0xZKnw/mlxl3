# MLXL3 Desktop 2.0.20

**Agent mode's single JavaScript tool gets a persistent kernel and background jobs: what the model reads or computes stays in memory between calls, and long commands keep running instead of being stopped after a few seconds.**

Desktop **2.0.20 / build 73** bundles MLXL3 engine **2.0.28**, unchanged: same engine sources as [`engine-v2.0.28`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.28), rebuilt with this commit.

## New

Agent mode still gives the model a single `javascript` tool. It now works like a persistent REPL, the design of [Unharnessed](https://github.com/unharnessed-agent/Unharnessed), with background commands as in [opencode](https://github.com/anomalyco/opencode).

### A persistent JavaScript kernel

- **The kernel lives as long as the task's sandbox server.** Before, each call started a new JavaScript engine and only `state` survived.
- **`$` keeps any value between calls:** text read from files, parsed data, `Map`s and the model's own helper functions. The model is told to keep what it reads there instead of reading it again.
- **Restarts are announced.** If a call has to be stopped (time or memory limit), the kernel restarts. The result says that `$` was emptied, and `state`, saved on disk, keeps its value.
- **Permissions apply to every call.** Turning off writing or commands in Agent Studio removes those functions at the next call.

### Background jobs

- **New functions:**
  - `spawn(command)` starts a background job: a server, a build, an install or a long test run.
  - `jobs()` lists them.
  - `tail(id, lines)` shows the latest output.
  - `wait(id, seconds)` waits for the end.
  - `kill(id)` stops a job.
- **Limits:** up to 4 jobs at once, 15 minutes each.
- **`run()` no longer stops long commands.** A command still running after the task's command wait (5 s by default) keeps going as a background job. `run()` returns its output so far, `status: null` and the job's number.
- **The end of a job is announced** in the next result: `[job 3 finished: status 0 after 42 s] npm test`.
- **Cleanup.** Jobs stop when the task's sandbox server stops. Jobs left behind by a kernel that was killed are stopped by the next one.

### Smaller additions

- **`glob(pattern)`** finds files, for example `glob("src/**/*.ts")`. A pattern without `/` matches file names anywhere.
- **A lone expression is its own result.** `view("app.py")` returns its value without `return`.
- **Syntax warnings after writes.** Writing or editing a `.py`, `.sh` or `.swift` file reports syntax errors as a warning. The file is still written.

The tool definition is 684 tokens and the Agent prompt 301 (Qwen3.8 tokenizer, Internet on), up from 588 and 251.

## Fixes

- **The sandbox's `pip3` and `pip` wrappers were rewritten at every command.** They are now written once, and restored only if they change.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The Windows app is unchanged since 2.0.19.

## Unchanged

- Engine 2.0.28, the Agent sandbox's rules, models, conversations, settings and Agent projects.
- Everything else is as in [2.0.19](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.19).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Agent sandbox checks: 350 passed.** The 31 new checks cover:
  - `$` and a helper function kept between calls;
  - permissions applied per call;
  - restarts after a time limit or an `await` that never resolves;
  - jobs: start, output so far, wait, end notices given once, `run()` continuing in the background, the 4-job limit, `kill` and the end of the kernel;
  - `glob`;
  - shell and Python syntax warnings.
- **Agent protocol checks: 73 passed**, run against this DMG's app binary. They include:
  - `$` and a background job kept across two MCP calls;
  - one kernel per server;
  - a killed kernel announced, with its orphaned job stopped by the next kernel and its JavaScript engine stopping by itself;
  - the end of the server ending the kernel and its jobs.
- **12 mutants detected.** These are deliberately broken versions of the kernel, the jobs and the server; the checks catch each one.
- **A bug found and fixed before release.** When its supervisor died, a kernel's JavaScript engine read an empty line and looped on it. An empty line now ends the kernel, the kernel's CPU limit is 300 s, and new kernels stop orphaned ones. A protocol check reproduces the bug on the previous build.
- **GPU check: indicative only.** It was run on Inkling 2 Mini at temperature 1, with the defaults of a new task, and three runs are not enough to measure a success rate.
  - Results: bug fix and rename both passed in 5 tool rounds; the question was answered correctly. The pre-registered question check rejected the correct answer because of its exact wording, as in earlier releases: 2/3 by that check, 3/3 with the corrected one.
  - No engine error, unreadable call or tool error, and no JavaScript engine left running afterwards.
  - These short tasks did not need `$` or background jobs.
- **This DMG:**
  - `hdiutil verify` valid, `codesign --deep --strict` passed;
  - version 2.0.20 / build 73;
  - `build-info` has commit `b88dd6179` with no tracked changes;
  - the bundled engine reports 2.0.28, is arm64, has SHA-256 `86e0b818…2067`, and ships MLX 0.32.2's `mlx.metallib`.
- **Not run:** the full Desktop suite. The app was compiled, and the checks above ran on this code.
- **Limits:**
  - The model learns that a job has ended in its next result; the engine does not wake it up by itself.
  - `$` is emptied whenever the engine or the task's sandbox server restarts.

## Source and download

- Source: [`b88dd6179`](https://github.com/0xZKnw/mlxl3/commit/b88dd6179715360135bb7306982787bfd8d7e4ab) (tag `v2.0.20`).
- `MLXL3-Desktop-v2.0.20-b73-Apple-Silicon.dmg`: 243,899,203 bytes, SHA-256 `8a091951960a2ebf648609937b9332f97e21771a90589b48bd5df33aa3f729d9`.
- `MLXL3-Desktop-v2.0.20-b73-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.28, the same sources as [`engine-v2.0.28`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.28), rebuilt at this commit.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the request for a real single-tool harness like Unharnessed, and for spotting that the Mac was overloaded, which exposed the kernel's input loop.
- Thanks to [Unharnessed](https://github.com/unharnessed-agent/Unharnessed) for:
  - the persistent single-tool REPL;
  - the kernel-reset notice;
  - background commands with handles and end notices;
  - the advice to keep what was read in variables.
- Thanks to [opencode](https://github.com/anomalyco/opencode) for:
  - shell commands that move to the background instead of being killed;
  - `glob`;
  - diagnostics after edits;
  - code mode's last-expression result.
- Thanks to Thinking Machines for Inkling 2 Mini, used for the GPU check.
- Thanks also to the projects behind the runtime and the app:
  - Apple JavaScriptCore, which runs the kernel;
  - [ExLlamaV3](https://github.com/turboderp-org/exllamav3);
  - [Apple MLX](https://github.com/ml-explore/mlx);
  - [PonyExl3](https://github.com/beamivalice/PonyExl3);
  - [SwiftMath](https://github.com/mgriebling/SwiftMath).
