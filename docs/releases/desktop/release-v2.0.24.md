# MLXL3 Desktop 2.0.24

**A quiet activity log: tool calls read as one-line actions, reasoning folds itself when it is done, and nothing pulses anymore.**

Desktop **2.0.24 / build 77** bundles MLXL3 engine **2.0.31**: the same files as [`engine-v2.0.31`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.31).

## New

### Tool calls you can read at a glance

Each tool call is now one line that says what happened and to what, instead of a full-width card with the raw function name, its server and a status at the far end of the row.

- **Agent mode:** "Ran JavaScript `const files = list();`" (its first statement), "Read `src/stats.pl`", "Wrote `src/stats.pl`", "Ran `sh test.sh`".
- **Failures:** a failed command shows its exit status, for example "Command `sh test.sh` **Exit 1**". The icon and the outcome are in a muted red that stays readable in light and dark mode.
- **MCP tools in Chat:** a readable name, the query and the server, for example "Web search “mlx prompt cache” · exa".
- **Details on click:** a click opens what the tool was given and what it returned. JavaScript shows as written, a written file shows its content as text, and other arguments show as indented JSON. Long sections have **Show all** and **Copy**.
- **Agent Studio:** its "Agent actions" list uses the same labels.

### Live status without blinking

- The pulsing dot beside "Thinking" and under every streaming answer is gone.
- The step that is running now carries a soft light sweep on its label ("Thinking", "Running `sh test.sh`", context reading). The sweep stops when the step ends. With Reduce Motion on, the label stays still.
- Before the first token, the reply says "Preparing response". Between a finished tool and the next step, it says "Preparing the next step".

### Reasoning that folds itself

- While the model thinks, its reasoning is shown live in a scrolling area that follows the text.
- When the model moves on, it folds to "Thought for 4 s". A click opens it again, and it then stays the way you chose.
- The duration of each reasoning block and tool call is saved with the conversation. Conversations from earlier versions show "Thought" without a duration.

### Less clutter

- Consecutive reasoning and tool lines are grouped tightly, so a ten-step Agent task fits on one screen.
- "Context processed · 0 s" lines are no longer kept after each round. Context reading stays visible while it runs, and afterwards only if it took a second or more or was interrupted.
- Message actions (copy, edit, regenerate, read aloud, branch) appear on hover, as one row of identical icon buttons with tooltips. The latest reply keeps its actions visible.
- A user message's actions appear beside its bubble, so they no longer reserve an empty gap below it.
- Document sources are compact chips instead of system buttons.
- A one-line code block is sized to its line instead of a box twice as tall.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The Windows app is unchanged in this release.

## Unchanged

- Models, conversations, projects, Agent projects and settings. Existing conversations open with the new layout.
- The engine, sampling and the Agent sandbox: everything else is as in [2.0.23](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.23).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Timeline checks** (`--check-chat-timeline`) passed. They cover:
  - tool labels in French and English, server prefixes, exit statuses, clipped long queries, and what the details show;
  - the durations of reasoning and tool calls, including after an interruption and after a relaunch;
  - the grouping of activity lines, over every sequence of up to five steps (1,365 sequences).
  - Temporary mutations were tried and detected: a missing reasoning clock, and a broken server prefix (detected after a case was added for a prefixed MCP file tool).
- **Desktop suite** (`scripts/check-desktop.sh`, macOS 26.5 SDK): everything passed except three checks that fail in the same way on the previous `main` (`6d1b8df16`, before this change):
  - `desktop-experience`: the Graphite home texture is not found at the bottom of the off-screen render. With only that block disabled in a copy of the check, its other 3,212 assertions pass.
  - `studios-check`: a sandboxed `cat` of the Agent project metadata is not refused.
  - `aurora-shader`: the check no longer compiles, since `StudioAurora.swift` needs `Theme.swift` (2.0.22).
- **Response UI:**
  - native copy and controls, and long streaming through the bridge;
  - scroll following while a reply streams: **0 jumps** over 10 and 34 messages.
- **Screenshots** of the preview fixtures: Graphite dark and light, Aurora ASCII, an Agent task running, finished and thinking live, and an expanded tool call.
- **App build:** release build with the macOS 26.5 SDK.
- **This DMG:**
  - `hdiutil verify` valid, `codesign --deep --strict` passed;
  - version 2.0.24 / build 77, arm64, built with the macOS 26.5 SDK;
  - `build-info` has commit `597d9fe1a` with no tracked changes;
  - the bundled engine reports 2.0.31 (`engine-v2.0.31`, commit `e117e02e8f7b`), and its five runtime files are identical to the published engine archive.
- **Not verified:**
  - hover and the folding animation, which were not driven by automated clicks;
  - the CPU and GPU cost of the light sweep during a real generation (not measured; it redraws only the running line, at 30 frames per second);
  - an Agent task on a real model with the new layout, end to end;
  - the app driven by hand.

## Source and download

- Source: [`597d9fe1a`](https://github.com/0xZKnw/mlxl3/commit/597d9fe1aded14468378c0eb7d3789dc753dd22c) (tag `v2.0.24`).
- `MLXL3-Desktop-v2.0.24-b77-Apple-Silicon.dmg`: 252,712,348 bytes, SHA-256 `8f1c8a259a2dd9b2ee65d8cc0e875fb243d757bd35dc63cddac3e8d4465b551a`.
- `MLXL3-Desktop-v2.0.24-b77-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.31, the same files as [`engine-v2.0.31`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.31).

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for pointing at the tool calls and the blinking dots, and for asking for an app that looks less generated.
- Design references for this release:
  - Refero's [anti-AI-slop guide](https://github.com/referodesign/refero_skill) and Laith0003's [UX skill ban list](https://github.com/Laith0003/ux-skill/wiki/Anti-AI-slop-ban-list.md): no cards without interaction, colour only for state, motion only while something changes.
  - Bob's [Making long agent conversations scannable](https://timetobuildbob.github.io/blog/making-long-agent-conversations-scannable): one line per step.
  - 21st.dev's [typing indicator guide](https://21st.dev/blog/react-typing-indicator-components): say what is running instead of animating dots.
  - The [AI SDK Elements](https://elements.ai-sdk.dev/components/shimmer) shimmer and [shinychat](https://shiny.posit.co/blog/posts/shinychat-tool-ui) tool cards.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
