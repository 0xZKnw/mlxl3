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

**Windows:** `MLXL3-Desktop-v2.1.0-Windows-x64-setup.exe` is attached, built by CI from this tag. It is the first Windows installer since 2.0.32 and carries the Windows fixes announced in [2.0.33](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.33) (no time limit on an Agent turn, a tool server restarts after a stopped or failed call). The Windows app does not have the three changes above: it still has Chat and Agent, and no folder mode.

## Verification and limits

Checks run on these sources on an Apple M5, before publication.

- **One real-model run, on the files of this release** (the bundled engine and the app's own sandbox server): Inkling 2 Mini EXL3 3.33 bpw, temperature 1, 16,384-token context, a fabricated conversation of six long exchanges followed by a question. The engine summarized ten earlier messages (prompt 13,139 → 4,331 tokens), answered, and the next turn, rebuilt the way the app does, was sent with 4,671 prompt tokens and answered. The summary kept the two facts planted in the first two messages and both answers gave them. One run, one model, one conversation: it shows the path works, not how good summaries are in general.
- Studio modes check with a fake engine: 2,534 checks (summary stored, applied to the next request and to branches, invalid events ignored; the approval question shown, answered, dropped on Stop, never answered for another command; earlier chats opened as tasks; the user's MCP servers merged into the Agent registry).
- Agent sandbox check: 621 passed, and 153 through the production entry of the sandbox server (commands asked, allowed, refused, unanswered; no question outside a user's folder).
- The full Desktop suite passed except its last step, which could not start (`pytest` missing from the system Python); that step's two scripts were run with the project's Python and passed (65 and 96 checks).
- Engine: `cargo fmt --check`; Clippy with `-D warnings` on `mlx,chat`, `chat` and `direct-metal,chat`; `cargo test --no-default-features --features chat`: 593 passed, 0 failed, 13 ignored (library).
- 22 temporary mutants of the new guards (9 engine, 6 sandbox server, 7 app) were all detected.
- A defect found while reading the final diff was fixed before this release: the approval alert answered the *current* request and treated its own dismissal as a refusal, so depending on the order in which the system closes an alert, **Allow** could have been recorded as a refusal, or applied to a command asked after the one displayed.
- **Not verified:** real clicks in the app (the alert, the menus, the sidebar: their logic is tested through the app model and one rendered screenshot was inspected; the app of this DMG was not launched); a summary with Qwen or any model other than Inkling 2 Mini; a conversation with images or tool history across the cut on a real model; one of your MCP servers being slow or stuck when the Agent starts; the speed of a plain conversation now that it carries tool definitions. Kani was not run.
- CI: **the CI of this source had not finished when this release was published** (profile compatibility passed; Rust, regression, Apple compatibility and EXL3 workflows still running). On the previous `main` commit the Windows, Rust, Apple compatibility, EXL3 and profile workflows passed, and `Regression checks` failed on a save-panel check of the image history that also fails on unrelated branches of the hosted runner and passes locally. No Windows file changed since that commit.

## Source and download

- Source: [`20473fd12`](https://github.com/0xZKnw/mlxl3/commit/20473fd12) (tag `v2.1.0`).
- `MLXL3-Desktop-v2.1.0-b88-Apple-Silicon.dmg`: 255,366,284 bytes, SHA-256 `b831ce22ee9b64d56f9fcb52cfd7ebb91d2385027bb0893e023c36113b5ec4a8`.
- `MLXL3-Desktop-v2.1.0-b88-Apple-Silicon.dmg.sha256`: 109 bytes.
- `MLXL3-Desktop-v2.1.0-Windows-x64-setup.exe`: 25,917,330 bytes, with its `.sha256` (108 bytes); built and attached by the `Windows Desktop` workflow (its three jobs passed). Not downloaded or installed for these notes.
- Bundled engine: MLXL3 engine 2.0.38, the five files of [`engine-v2.0.38`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.38) (compared byte for byte with the extracted archive).
- This DMG: `hdiutil verify` valid, `codesign --deep --strict` passed on the app of the mounted DMG, version 2.1.0 / build 88, `build-info` has commit `20473fd12` with no tracked changes, MLX 0.32.2 and engine 2.0.38 (`engine-v2.0.38`). The app binary of the DMG is the one whose sandbox server was used in the real-model run.

## Acknowledgements

- [0xZKnw](https://github.com/0xZKnw) for the single mode, the summary of earlier turns and the confirmation before deleting.
- [HENK0O](https://github.com/HENK0O) for the task folders, per-turn review and Agent Studio this builds on ([#204](https://github.com/0xZKnw/mlxl3/pull/204)).
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), whose summary of a long turn the summary of earlier turns extends.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
