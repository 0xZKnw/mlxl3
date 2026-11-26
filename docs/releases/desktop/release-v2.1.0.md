# MLXL3 Desktop 2.1.0

**One conversation mode: Chat and Agent are merged. Every conversation is now an Agent task, long conversations are summarized instead of hitting the context limit, and the Agent asks before deleting anything in your folder.**

Desktop **2.1.0 / build 88** bundles MLXL3 engine **2.0.38**: the same files as [`engine-v2.0.38`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.38).

## New

### Chat and Agent are one mode

- **The sidebar no longer has a Chat row.** It lists Agent, Decision, Server and Image; ⌘1 opens Agent. A plain question is simply a task in which the model calls no tool.
- **Your existing chats are kept.** They open as tasks without a folder (each works on its own private copy, as Agent tasks without a folder always did). Nothing is deleted or rewritten in them; they are saved as tasks the next time the app saves.
- **Your MCP servers serve the Agent.** With **Tools → Enable MCP tools** on, the servers of your `mcp.json` are offered to the model next to the sandbox tools and web search. A server of yours that fails to start no longer blocks the Agent; a server named `sandbox` or `exa` is replaced by the app's own.
- **What changes for a plain conversation:** every request now carries the tool definitions (about 1,300 to 1,600 prompt tokens on Inkling 2 Mini, measured statically earlier, not re-measured for this release), and the model may decide to use a tool. JSON-schema answers are not available in this mode.

### Long conversations are summarized

- **When earlier turns fill the context, the model summarizes them.** Until now only tool outputs were trimmed; the text of old questions and answers stayed, and the turn ended on "Context limit reached". With engine 2.0.38, once the context is more than three-quarters full and nothing of the current turn can be given up, the oldest turns are replaced by a summary the model writes itself. The cut is always before one of your messages; the most recent turns that fit in a fifth of the context are kept word for word, and your last message always is.
- **You keep the whole conversation on screen.** Only what is sent to the model changes: the summary, then the messages from the first one kept. The summary is saved with the conversation, so the next turns and the next launch do not summarize again.
- While it happens the answer shows `Summarizing N earlier messages to free context`, then `Context summarized: …`. If the summary cannot be used (empty, cut, or not shorter), the messages are kept as they were.
- A branch keeps the summary only if it still contains the message the summary starts at; otherwise it sends its full history again.

### The Agent asks before deleting in your folder

- **In a task that works directly in one of your folders**, a command that deletes or overwrites with no way back waits for your answer: **Allow**, **Always allow for this task**, or **Refuse**. Without an answer within one minute the command is not run, and the model is told so.
- **What is asked:** `rm`, `rmdir`, `unlink`, `shred`, `srm`, `truncate`, `dd … of=`, `find … -delete`, `mkfs`, `diskutil erase…`, `git reset --hard`, `git clean`, `git checkout --` / `.` / `-f`, `git restore`, `git stash drop` / `clear`, `git push --force`, `git branch -D`, `git filter-branch`.
- **The model cannot answer for you:** the request is written outside the folders its commands can write to. Each button answers the command it shows, never another one asked meanwhile.
- **Agent Studio → Sandbox → Ask before deleting** turns the question off or on for a task.
- **Limits:** detection is by command text. A script (Python, Node, a Makefile…) that deletes files itself is not seen, and neither is the `javascript` harness. Tasks without a folder never ask: they work on a private copy.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

**Windows:** the Windows app does not have these three changes (it still has Chat and Agent, and no folder mode).

## Verification and limits

_To be completed with the build._

## Source and download

_To be completed with the build._

## Acknowledgements

- [0xZKnw](https://github.com/0xZKnw) for the single mode, the summary of earlier turns and the confirmation before deleting.
- [HENK0O](https://github.com/HENK0O) for the task folders, per-turn review and Agent Studio this builds on ([#204](https://github.com/0xZKnw/mlxl3/pull/204)).
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), whose summary of a long turn the summary of earlier turns extends.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
