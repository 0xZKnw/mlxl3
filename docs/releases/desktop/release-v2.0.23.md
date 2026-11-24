# MLXL3 Desktop 2.0.23

**Agent mode with DeepSeek Harness's sandbox, the whole output of long commands, and summarized steps instead of a full context.**

Desktop **2.0.23 / build 76** bundles MLXL3 engine **2.0.31**: the same files as [`engine-v2.0.31`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.31).

## New

### An open sandbox, like DeepSeek Harness

Agent commands now run under the same kind of macOS sandbox as [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness): **everything is allowed except writing outside the project copy, the command's private home and `/tmp`**. Writes outside those folders are refused by the kernel, whether direct, through a symbolic or hard link, a removal or a rename.

Things that used to fail now work:

- reading any file, for example `~/.gitconfig`, `~/Library` preferences and caches, or SSH keys for `git` over SSH;
- system services and local sockets, for example `defaults`, Docker and `ssh-agent`;
- writing to `/tmp`, stopping your own processes with `pkill`, and new sessions (`setsid`).

Unchanged:

- The agent still works on an **isolated copy** of the project, and nothing reaches your files before you review the changes.
- The JavaScript interpreter of the `javascript` tool still has no file, network or process access of its own; everything goes through its functions.
- Turning **Internet off** still refuses network connections.

**Security trade-off** (the same as DeepSeek Harness): with Internet on, which is the default, the agent can read secrets and send them elsewhere. It could do so if a web page or a file it reads tricks it. Commands can also act outside the copy through other apps' services, such as AppleScript. Turn Internet off for projects where this matters.

### The whole output of long commands

When a command's output is too long to show whole (over 32 KiB, of which the first 8 KiB and the last 24 KiB are shown), the whole output is saved in the copy, under `.mlxl3-scratch/outputs/`. The result names the file, so the model can read the middle of a long build or test log with `view(path, offset)` or `search(regex, path)` instead of running the command again.

- These files are never part of the reviewed changes.
- They are not listed by `list()`.
- Only the 16 newest are kept, at most 64 MiB each.
- The file is created without following links, so a link planted in that folder is never written through.

### From engine 2.0.31

- **Summarized steps:** when old tool output cannot be shortened enough, the model summarizes the earlier steps of the task, instead of the turn stopping with a full context. The status line shows "Summarizing…" while this happens.
- **Unknown tools:** a call to a tool the model was not given gets the list of tools it can call, instead of an error that ended the turn in chat with project tool restrictions.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The Windows app is unchanged since 2.0.16.

## Unchanged

- Models, conversations, projects, Agent projects and settings.
- Everything else is as in [2.0.22](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.22).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Engine 2.0.31:** see its [release notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.31) for the loop and summary checks.
- **Agent sandbox checks:** 345 passed, with real sandboxed commands, the JavaScript harness, jobs and the new output files.
  - They cover reads outside the copy, refused outside writes, `/tmp`, standard streams, Mach services, `setsid`, and the network following the Internet setting.
  - With live network checks before the output-file change, 340 passed, including real `curl`, `git clone` and `pip`.
  - The MCP protocol checks passed: 73, or 74 with the network.
  - Temporary mutations were tried and all detected: 5 sandbox and 5 output-file mutations.
- **App build:** release build with the macOS 26.5 SDK.
- **This DMG:** DMG_CHECKS
- **Not verified:**
  - an Agent task on a real model that exercises the new sandbox and output files end to end;
  - the full Desktop suite, which was not run again (CI runs it);
  - the app driven by hand.

## Source and download

DOWNLOADS

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for asking to bring DeepSeek Harness's sandbox level and robustness to the Agent mode.
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (MIT), by DeepSeek: its macOS sandbox profile (`sandbox-local`) and its spill policy for long tool output are the models for this release.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
