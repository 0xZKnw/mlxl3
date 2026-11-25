# MLXL3 Desktop 2.0.34

**Agent can now work directly in one of your folders: choose it at the top of the window, and find your tasks grouped by folder in the sidebar.**

Desktop **2.0.34 / build 87** bundles MLXL3 engine **2.0.37**, rebuilt from this source. The Rust sources are identical to [`engine-v2.0.37`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.37); the binary is not byte-identical to that archive because it records another source commit. The engine itself is not republished.

## New

### Agent works in your folder

- **Choose a folder at the top.** In Agent mode the header has a folder button: your recent folders, **Choose a folder…**, **No folder (private copy)** and **Show in Finder**. **Tools → Work in a folder…** opens the same choice.
- **Changes are made directly in that folder.** There is no copy and no review step before applying any more for these tasks: files change while the turn runs, and the model is told so. The changes of each turn can still be reviewed from its answer, and Agent Studio keeps its versions.
- **The sandbox still confines writes.** Commands can write only to the chosen folder and to `/tmp`. They can read the whole Mac, as before.
- **Nothing is added beside your project.** The task's history, permissions and its commands' caches stay in the app's data.
- **Refused folders:** `/`, your home folder itself, and any folder that holds MLXL3's own data.
- **One folder per task.** An empty task takes the folder you choose. A task that already has messages keeps its folder: choosing another one starts a new task there. A new task (⌘N) continues in the current folder.

### Sidebar grouped by folder

- In Agent mode, tasks are listed under the folder they work in. Groups open and close and stay that way; **+** on a group starts a task in that folder; tasks without a folder come last. Two folders with the same name also show their parent folder. A search shows every match, whatever is closed.
- An empty task is now called **New task**.

## Changed

- **Work in a folder…** no longer imports a copy of the project to review before applying. Tasks that already work on such a copy keep working, including **Review** and **Apply**; new ones use the folder itself. Choose **No folder (private copy)** for a task that should stay away from your files.

## Good to know

- **Restore** in Agent Studio rewrites the real folder to an earlier version (a version is saved just before).
- The per-turn history covers at most 4,000 files or 64 MiB. Symbolic links, files of a nested git repository, and `.git`, `node_modules`, `.venv`, `build`, `dist`, `target` and `.env` files are left out of it.
- A command whose output is cut writes the full output to `.mlxl3-scratch/` **inside** your folder.
- Deleting a task never touches its folder. A folder that has been moved or whose volume is ejected is reported, never recreated.
- Windows does not have this feature.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security. The Windows installer is built and attached by the `windows-desktop` workflow after publication.

## Verification and limits

- **No test suite was re-run for this release.** The checks below were run on the feature's sources before they were pushed to `main`, on an Apple M5, without loading a model.
- Agent sandbox check: 551 passed, and 144 through the production entry of the sandbox server (commands use the assigned private home, write inside the folder, are refused beside it, and nothing is created beside the folder).
- Studio modes check with a fake engine: 2,499 checks, 39 of them new (refused folders, the server pointed at the real folder, a turn sent with a symbolic link present, per-turn review, branching and deleting without touching the folder, a missing folder reported).
- Tasks on a private copy: 96 checks passed; completion notifications: 17.
- 8 temporary mutants of the new guards were all detected.
- **Not verified:** any run with a real model; the folder picker and real clicks in the menus and the sidebar (their logic is tested through the app model and one rendered screenshot was inspected); a folder on an external or network volume; a very large repository; the time each turn spends indexing a large folder. The full Desktop suite and CI were not run before publication.

## Source and download

- Source: [`8edbbb065`](https://github.com/0xZKnw/mlxl3/commit/8edbbb065) (tag `v2.0.34`).
- `MLXL3-Desktop-v2.0.34-b87-Apple-Silicon.dmg`: 255,033,893 bytes, SHA-256 `0650ce8235c37e0a48a65052fdae6a85d1992a2db2d33126fe403de5dad29d22`.
- `MLXL3-Desktop-v2.0.34-b87-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.37 rebuilt from this source (same Rust sources as `engine-v2.0.37`).
- This DMG: `hdiutil verify` valid, `codesign --deep --strict` passed on the app of the mounted DMG, version 2.0.34 / build 87, `build-info` has commit `8edbbb065` with no tracked changes, MLX 0.32.2 and engine 2.0.37 (`engine-v2.0.37`). The app was not launched for this release.

## Acknowledgements

- [0xZKnw](https://github.com/0xZKnw) for the folder mode and the grouped sidebar.
- [HENK0O](https://github.com/HENK0O) for the task folders, per-turn review and Agent Studio this builds on ([#204](https://github.com/0xZKnw/mlxl3/pull/204)).
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness), whose default `workspace-write` mode (work in the real folder, writes confined to it) this follows.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
