# MLXL3 Desktop 2.0.18

**A simpler Agent screen: one compact bar under the composer, with Agent Studio, the task's folder and a one-click Internet switch. The Agent sandbox's developer tools also run without extra noise or delay when Xcode is selected.**

Desktop **2.0.18 / build 71** bundles MLXL3 engine **2.0.27**, unchanged: same engine sources as [`engine-v2.0.27`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.27), rebuilt with this commit.

## New

### A simpler Agent screen

- **One compact bar under the composer**, in a task and on the New task screen:
  - **Agent Studio**;
  - the task's folder;
  - its HTML previews: a button for one file, a menu when there are several;
  - on the right, an **Internet** switch.
- **Internet in one click.** The switch turns Internet on or off for the current task: commands (`pip`, `npm`, `git clone`), `fetch` and web search. It is the same setting as in Agent Studio's **Sandbox** panel. A globe means on, a crossed-out network icon means off.
- **Less text.**
  - The bar above the composer and the notice below it are gone. They still said "Tools offline" and "Network blocked; commands limited to 5 s", which has not been true since 2.0.17.
  - "This Mac" is gone too.
  - Agent Studio no longer says "Offline, inside the isolated copy".

## Fixes

- **Developer tools in the Agent sandbox were slow and noisy with Xcode selected.** With Xcode as the developer folder, every `python3`, `git`, `cc` or `swiftc` call went through an `xcrun` shim. The shim started `xcodebuild` inside the sandbox each time, and its warnings ("DVTFilePathFSEvents…", "DARWIN_USER_CACHE_DIR…") were mixed into the output the model read.
  - The sandbox now puts the selected developer folder's own tools first on the `PATH` and sets `SDKROOT`. This works for Xcode and the Command Line Tools.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. The Windows app is unchanged since 2.0.16.

## Unchanged

- Engine 2.0.27, the Agent sandbox's rules and the JavaScript tool of 2.0.17, models, conversations, settings and Agent projects.
- Everything else is as in [2.0.17](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.17).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

TO FILL

## Source and download

TO FILL

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the screenshots of the outdated Agent texts and the request for a minimal Agent screen.
- Thanks to the GitHub Actions macOS runner, whose Xcode setup exposed the `xcrun` noise in the sandbox.
- Thanks also to the projects behind the runtime and the app:
  - Apple JavaScriptCore;
  - curl;
  - [ExLlamaV3](https://github.com/turboderp-org/exllamav3);
  - [Apple MLX](https://github.com/ml-explore/mlx);
  - [PonyExl3](https://github.com/beamivalice/PonyExl3);
  - [SwiftMath](https://github.com/mgriebling/SwiftMath).
