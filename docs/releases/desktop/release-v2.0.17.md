# MLXL3 Desktop 2.0.17

**Agent mode gets an open sandbox and a sturdier JavaScript tool: developer tools work, commands can read your Mac (never your secrets) and use the Internet, there is no more 10-minute limit, and edits that small models get slightly wrong now land.**

Desktop **2.0.17 / build 70** bundles MLXL3 engine **2.0.27**, unchanged: same engine sources as [`engine-v2.0.27`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.27), rebuilt with this commit.

## New

### An open Agent sandbox

The sandbox stays, but it is no longer closed off. Agent tasks can now do real development work.

- **Developer tools work.** `python3`, `pip3`, `cc`/`clang`, `make`, `git` (commits included) and `swiftc` run inside the sandbox, with Homebrew and `/usr/local` on the `PATH`. Caches and temporary files go to a private home folder beside the task's copy, never into your project.
- **Commands can read your Mac, except your secrets.** Reading is allowed everywhere except:
  - keys and credentials: `~/.ssh`, `~/.gnupg`, cloud, Git, npm and PyPI credentials, the GitHub CLI and other agents' data;
  - shell histories;
  - `~/Library`: keychains, browser data and cookies, Mail, Messages, app data and the app's own conversations. Its developer folders stay readable: Developer, Android, Java, Python, Fonts.
  - `.env` files outside the task's copy;
  - the task's own folder, where its checkpoints and permissions live.

  Links, hard links and copies cannot reach a protected file.
- **Writing is still limited** to the task's copy and its private home folder. Nothing outside them can be changed, and your originals change only after you review the copy.
- **Internet is on by default for new tasks.** Commands reach the network directly: `pip install`, `npm install`, `git clone`, `curl`.
  - Web search and `fetch`/`download` follow the same switch. Their host list now defaults to all hosts (`*`), and local and private addresses are still refused.
  - Unix sockets outside the copy, such as `ssh-agent` or Docker, stay closed.
  - Turn Internet off for a task in the **Sandbox** panel of Agent Studio. Tasks saved by 2.0.16 keep their saved setting.
- **No more 10-minute limit.** An Agent answer now stops only when you stop it, or after 5 minutes without any activity.

### A sturdier JavaScript tool, still a single tool

Agent mode keeps its single `javascript` tool, the design that OpenAI Codex's code mode and DeepSeek's harness also use. Its functions were reworked from what those harnesses, SWE-agent, OpenHands, Claude Code and Aider found about small models:

- **`view(path, offset, limit)`** shows numbered lines, 200 at a time, and says how to see the rest.
- **`edit` tolerates near misses.** It still tries the exact text first. It then compares whole lines while ignoring:
  - trailing spaces, then indentation;
  - `N| ` prefixes copied from `view`;
  - typographic quotes and dashes.

  Windows line endings (CRLF) are kept. Pass `true` to replace every occurrence.
- **Edit errors say how to fix the call.**
  - Several matches: their line numbers.
  - Text not found: the closest lines of the file.
  - Replacement already present: "this edit may already be done".
- **`edits(path, [[find, replace], …])`** applies several changes to one file, all or nothing.
- **Errors quote the failing line** of the program.
- **Misshaped programs run.** A program sent as a function it never calls, as `(async () => {…})()`, or as `{"code": …}` now runs instead of doing nothing.
- **Commands are non-interactive.**
  - No pager, colors, prompts or update notices: `PAGER=cat`, `NO_COLOR=1`, `CI=1` and more.
  - `run()` output drops terminal color codes and the progress lines a `\r` overwrote.
- **A `.json` file that a write or edit breaks** produces a warning.
- **Loop notes.** When the same program returns the same result a third time, or the same error comes back three calls in a row, the result tells the model to change its approach.

The tool definition grows from 525 to 588 tokens and the Agent prompt from 216 to 251 (Qwen3.8 tokenizer, Internet on).

## Fixes

- **Python and pip failed in the sandbox** with `xcrun: error: unable to load libxcrun … file system sandbox blocked open()`. The developer folders are now readable.
- **Commands that start helper processes failed.** Python, clang and make start them with `posix_spawn`, which the sandbox refused. It is now allowed, and every process a command starts is still followed and stopped when the command ends, including one that leaves its process group.
- **`git commit` printed "setsid failed"** and Git had no identity. The private home folder now has a Git identity and keeps Git maintenance in the foreground.
- **Long Agent tasks stopped at 10 minutes** with "10-minute Agent limit reached".

## Security notes

- The protection covers the locations listed above. A secret you keep somewhere else, such as a token file in a project folder, can be read by commands.
- With Internet on, a server started by a command can listen on every network interface, and be reached from your local network, while that command runs. It is stopped when the command ends.
- macOS may ask, on behalf of MLXL3 Desktop, for access to Documents, Desktop or Downloads if a command reads those folders.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The changes above are in the macOS app; the Windows app is unchanged since 2.0.16.

## Unchanged

- Engine 2.0.27, models, conversations, settings and Agent projects.
- Everything else is as in [2.0.16](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.16).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Agent sandbox checks: 313 passed**, run against the real Seatbelt sandbox. They use a throwaway home folder holding a dozen fake secrets, and cover:
  - secrets refused, by any path and through links or copies;
  - writes outside the copy refused;
  - network access only with Internet on;
  - outside Unix sockets refused;
  - each new function and error message.

  With the live network: **270 passed**, including HTTPS with `curl`, `git clone` and `pip download`.
- **Agent protocol checks: 66 passed** through the real MCP server, including:
  - command networking for both harnesses;
  - the loop note on the third identical call.
- **Full Desktop suite passed**, including Studio modes (2,324) and completion notifications (17).
- **21 mutants detected.** These are deliberately broken versions of the sandbox profile and the new functions; the checks catch each one.
- **GPU check: indicative only.** It was run on Inkling 2 Mini at temperature 1, with the defaults of a new task (Internet on), and three runs are not enough to measure a success rate or an effect on the number of tool rounds.
  - Results: bug fix and rename both passed; the question was answered correctly. The pre-registered question check rejected the correct answer because of its exact wording, as in the previous release: 2/3 by that check, 3/3 with the corrected one.
  - No engine error, unreadable call or tool error. The model used `view`, `edit` and `edits`.
- **Release build.** The checks above ran on this code before the version change; only the version and the documentation changed since, so the suites were not run again on the release commit.
- **This DMG:**
  - `hdiutil verify` valid, `codesign --deep --strict` passed;
  - version 2.0.17 / build 70;
  - `build-info` has commit `bef59cdaa` with no tracked changes;
  - the bundled engine reports 2.0.27, is arm64, has SHA-256 `e1a1d444…47b2`, and ships MLX 0.32.2's `mlx.metallib`.
- Limits are listed under **Security notes**.

## Source and download

- Source: [`bef59cdaa`](https://github.com/0xZKnw/mlxl3/commit/bef59cdaae344b431b103134233e19c791c0937f) (tag `v2.0.17`).
- `MLXL3-Desktop-v2.0.17-b70-Apple-Silicon.dmg`: 251,506,425 bytes, SHA-256 `6a8b1d382aca7006d88ecdfb2b5131f013c5735ed2b60f6f809cb3f3e48b5f28`.
- `MLXL3-Desktop-v2.0.17-b70-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.27, the same sources as [`engine-v2.0.27`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.27), rebuilt at this commit.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the sandbox bug reports with screenshots, the 10-minute limit report, the open-sandbox decision and the harness request.
- Thanks to [Unharnessed](https://github.com/unharnessed-agent/Unharnessed), the reference for a harness that trusts the model with real tools.
- The harness changes follow published work:
  - [OpenAI Codex](https://github.com/openai/codex): code mode with a single JavaScript tool, and `apply_patch`'s matching cascade.
  - [DeepSeek's harness](https://github.com/deepseek-ai/deepseek-harness): a single `run_code` tool, repeat reminders and paginated reads.
  - [Hugging Face smolagents](https://github.com/huggingface/smolagents): the failing line in errors, and no repeated calls.
  - [OpenHands](https://github.com/OpenHands/software-agent-sdk): stuck detection and edit snippets.
  - [SWE-agent and mini-swe-agent](https://github.com/SWE-agent): the windowed file viewer and non-interactive commands.
  - [Claude Code](https://code.claude.com/docs/en/tools-reference): numbered reads and unique edits.
  - [Aider](https://github.com/Aider-AI/aider): closest lines when an edit does not match.
- Thanks to GitHub for [octocat/Hello-World](https://github.com/octocat/Hello-World) and to PyPI for `six`, used by the live network checks, and to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini, used for the GPU check.
- Thanks also to the projects behind the runtime and the app:
  - Apple JavaScriptCore, which runs the Agent programs;
  - curl, which makes the app's requests;
  - [ExLlamaV3](https://github.com/turboderp-org/exllamav3);
  - [Apple MLX](https://github.com/ml-explore/mlx);
  - [PonyExl3](https://github.com/beamivalice/PonyExl3);
  - [SwiftMath](https://github.com/mgriebling/SwiftMath).
