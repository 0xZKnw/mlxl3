# MLXL3 Desktop 2.0.26

**A redesigned sidebar where every place has a name, and image generation inside the main window.**

Desktop **2.0.26 / build 79** bundles MLXL3 engine **2.0.31** unchanged: the same files as [`engine-v2.0.31`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.31).

## New

### A redesigned sidebar

The sidebar was rebuilt around one idea: every place has a visible name, and everything lines up on one left edge.

- **Modes as labelled rows.** Chat, Agent, Decision and Server are rows with an icon and a name, instead of five small tabs with 10-point captions. Hover a row to see its shortcut. **⌘1…⌘5** switch modes from the View menu, even with the sidebar hidden.
- **Image** sits with the modes when this Mac can generate images (Apple M5 and macOS 26.3 or later). It opens the new Images page described below.
- **Search at the top.** The search field spans the sidebar and searches titles and messages. **⌘K** focuses it and Escape clears it.
- **New conversation** sits at the top right, next to Downloads, on the same line as the window controls (⌘N is unchanged). In Agent mode it creates a new task.
- **One history list.** Pinned conversations come first, then recent ones. A long title fades out at the edge instead of ending on "…", and the full title appears on hover. An untouched "New conversation" is shown dimmer so it does not read as one more chat.
- **"…" on every conversation.** Hovering a row shows a menu to pin, unpin or delete it. Right-click offers the same actions.
- **The list title is the project picker.** "Conversations" (or "Tasks" in Agent mode, or the project's name) opens the project menu, with **Manage projects…** at the end. The "No project" row is gone.
- **A labelled footer** replaces the four unlabelled icons. **Models** is a labelled row. **…** holds Lab, Projects and profiles, and MCP servers. The gear opens Settings, and the update button still appears when an update is available.
- Only Chat and Agent highlight a conversation. In Decision, Server and Lab, no row looks selected.
- Hover feedback is instant. Rows no longer animate when the pointer passes over them.

The Graphite, Minimal, Aurora ASCII and Aurora styles keep their own colours. Aurora styles keep the accent mark on the selected conversation.

### Agent: folder and Internet in the composer

In Agent mode, the task folder and the Internet switch now sit in the composer, next to Tools, as icons. Their tooltips still explain what they do. Agent Studio and the HTML preview stay under the composer.

### Image generation in the main window

Contributed by [HENK0O](https://github.com/HENK0O) in [#191](https://github.com/0xZKnw/mlxl3/pull/191).

- **Images** is now a page of the main window, like Models or Settings, instead of a separate window. It holds the model picker and the generation settings next to the preview and the prompt.
- **Back** returns to the previous screen and keeps the draft or the running job. Reopening Images shows its progress or the finished PNG, with a shortcut to reveal it in Finder.
- Progress messages tell file downloads apart from denoising steps. The initial teapot is labelled as a model example.
- The Window menu's **Generate an image** opens the same page.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The Windows app is unchanged in this release.

## Unchanged

- Models, conversations, pins, projects, Agent projects and settings.
- The engine. Engine changes merged on `main` after 2.0.31 are **not** in this release.
- Everything else is as in [2.0.25](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.25).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

<!-- DMG_CHECKS -->

## Source and download

<!-- DOWNLOADS -->

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the sidebar review: dropping the date groups, the account-like footer row and the hidden image button, moving Lab into "…", and putting the Agent folder and Internet switch in the composer.
- Thanks to [HENK0O](https://github.com/HENK0O) for the Images page ([#191](https://github.com/0xZKnw/mlxl3/pull/191)).
- Design references: Apple's Human Interface Guidelines (sidebars, segmented controls), Nielsen Norman Group's research on icon labels, and Linear's 2024 and 2026 interface redesign notes.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
