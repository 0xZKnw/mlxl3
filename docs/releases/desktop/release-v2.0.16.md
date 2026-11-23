# MLXL3 Desktop 2.0.16

**Agent mode can use the Internet through the app, Inkling 2 Mini can finally use its tools, and Agent works longer and more reliably: 24 tool rounds, a larger budget, long commands, and the code it runs shown in the conversation.**

Desktop **2.0.16 / build 69** bundles MLXL3 engine **2.0.27**.

## New

### Internet for Agent, through the app

Each Agent task now has an **Internet (through the app)** switch in the **Sandbox** panel of Agent Studio. It is **off by default**.

- **Web search** through Exa, the search service already built into the app.
- **`fetch(url)`** reads a page or a file as text, up to 2 MiB.
- **`download(url, path)`** saves a file into the task's copy, up to 100 MiB. Test material goes under `.mlxl3-scratch/`, which never appears in the changes to review. For example, a GitHub repository:

  ```js
  download("https://codeload.github.com/OWNER/REPO/zip/refs/heads/main", ".mlxl3-scratch/repo.zip");
  run("unzip -oq .mlxl3-scratch/repo.zip -d .mlxl3-scratch");
  ```

How it stays safe:

- The sandbox itself still has **no network**. The app makes each request on the model's behalf.
- Only **HTTPS GET** requests, only to the **hosts in the task's list**. The default list is GitHub (`github.com`, `codeload.github.com`, `raw.githubusercontent.com`, `objects.githubusercontent.com`, `api.github.com`), PyPI and npm. `*.domain` allows subdomains, and `*` allows everything, which is not recommended.
- Every redirect is checked again before it is followed, at most 5.
- Each name is resolved, and the connection goes to the checked address. **Local and private addresses are always refused**, IPv4 and IPv6: your Mac, your network, router pages and cloud metadata addresses.
- Downloads are written by the sandbox, so a link pointing outside the project cannot be used to write elsewhere.
- The model is told that content from the Internet is data, never instructions.

### Inkling 2 Mini uses its tools

With engine 2.0.27, Inkling 2 Mini gets the tools of Agent mode and of MCP, and its calls run. Before, it wrote calls as plain text and nothing happened.

### A longer, sturdier Agent

- **24 tool rounds** per answer instead of 5. At the limit, the model answers from what it has instead of the answer ending with an error.
- **32,768 tokens** per Agent answer instead of 4,096, reasoning included. Reasoning models used up the old budget before acting.
- **Long commands.** A JavaScript call can now last as long as the command limit set in the Sandbox panel, plus 3 seconds, instead of stopping at 10 seconds. Commands of the separate tools can use their whole limit, up to 30 seconds.
- **The code that ran is shown.** Each tool row in the conversation shows the program the model ran, with a button to copy it.
- **`state` keeps values between JavaScript calls** of a task, up to 256 Ki characters, so the model does not have to reread files or write notes into your project.
- An unreadable tool call is sent back to the model to fix, instead of ending the answer.

### DSpark off by default for Inkling 2 Mini

At temperature 1, the app's preset, DSpark did not beat plain decoding. It is now off unless you turn it on, and its drafter is loaded only when it is on. A choice you already saved is kept.

## Fixes

- **`search()` reported nothing when it failed.** An invalid pattern, such as `fetchUser(`, or a missing folder returned an empty list. It now reports the error, with a hint about extended regular expressions. Files with Windows line endings (CRLF) returned no match at all; they now return their matches.
- **`run()` stopped talkative commands.** A command that printed more than 32 KiB, a test suite for example, was stopped half-way, and reported status 0. It now runs to its end, keeps the first 8 KiB and the last 24 KiB of its output, and reports its real status (137 when it is stopped).
- **Long results lost their end.** Results over 32 KiB kept only their start. They now keep their start and their end, where summaries and errors are.
- **`list()`** returns exact file names, including unusual ones, and a partial list with a warning for very large projects, instead of failing.
- The Agent instructions no longer announce a 30-second limit to a call that had 10.
- A failed Agent answer still records a checkpoint of the files it changed.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. It now offers the soft aurora of the 2.0 betas as a background. The Agent and Internet changes above are in the macOS app.

## Unchanged

- Everything else is as in [2.0.15](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.15).
- Conversations, settings, models and Agent projects are kept. Agent projects saved by earlier versions open with Internet off.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **On a real model.** Inkling 2 Mini, as installed by Desktop, ran Agent tasks through the bundled engine's source with the app's Agent request and the real sandbox: 8 JavaScript runs with **no engine error, no unreadable call and no call markup in an answer**. It fixed a bug (2/2), renamed a function across 5 files (2/2), read a file from GitHub with `fetch` (2/2) and counted TODOs correctly (2/2). The round limit ended in an answer.
- **Internet:**
  - Checked without network: 96 checks on addresses (local, private and reserved, IPv4 and IPv6), host lists, URLs, redirects through a fake transport (other host, `http`, a private address, a loop), and downloads, where a link out of the project is refused by the kernel.
  - Checked live: `fetch` of a GitHub file, and `download` of the `octocat/Hello-World` archive through the `github.com` → `codeload.github.com` redirect, unzipped in the sandbox. The app binary from the DMG also fetched the file through the real MCP path.
- **Desktop checks** (`scripts/check-desktop.sh`, the full suite, Command Line Tools toolchain with the macOS 26.5 SDK): DESKTOP_RESULT. They include:
  - Agent sandbox: 228 checks, 230 with the live network.
  - Agent MCP protocol: 63 checks, 64 live.
  - Studio modes: 2,325. Studios: 59. Python: 39.
- **Mutations caught:** for `search()`, a pipe to `head` and line splitting by character; for `run()`, the early stop, reading the status before the process is reaped, and start-only output; `state` not saved; `list()` splitting by line; private addresses accepted; redirects not checked again; a download written outside the sandbox.
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum;
  - after mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.16 / 69;
  - its `build-info.json` names source `ed3898889399`, with no uncommitted changes, and engine 2.0.27;
  - the bundled engine files are byte-identical to the engine 2.0.27 archive;
  - the app binary from the DMG passed the 64 Agent MCP protocol checks, the live `fetch` included, and its chat timeline and MCP preference checks;
  - the bundled engine ran Inkling 2 Mini's tool calls and DSpark on demand on the installed model (see the [engine 2.0.27 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.27)).
- **Not verified:**
  - On a real model: the separate tools, and Qwen models with the new loop. The release run was shortened on request.
  - Whether models complete more tasks with the longer loop than before: there is no before/after comparison.
  - The Sandbox panel's new controls were checked by the build and the Studio checks, not by looking at them.
- **Known limits:**
  - Package managers (`npm install`, `pip install`) still have no network inside `run()`. `download` can fetch their archives.
  - Content from the Internet can still try to steer the model. The model is told to treat it as data, and nothing touches your files before you review it.
  - With `*` in the host list, a misled model could send data to any site. The default list is limited to GitHub, PyPI and npm.

## Source and download

- Compiled source `ed38988893997613216b7176a193481cbc0a72e1`: `main` at `d5b5d31e7` merged with the Agent work, and the versions.
- `MLXL3-Desktop-v2.0.16-b69-Apple-Silicon.dmg` · **253,258,227 bytes**, SHA-256 `0b3c6e8107b14bdc57d90a89690029175fd9ef2bf457e95fc538f4caf12fcc77`.
- `MLXL3-Desktop-v2.0.16-b69-Apple-Silicon.dmg.sha256` · 110 bytes, SHA-256 `dc97c586b0c36f7861aa405f81300aee09f7b9e893ece55ba3342cb4328e572c`.
- Bundled engine 2.0.27 (`mlxl3` SHA-256 `eeddf28cde64018bce8efad02bde85cb856ff4eb6c4c12dc0afa241a1ba766f2`, source `ed3898889399`). The Swift app was built with the Command Line Tools toolchain and its macOS 26.5 SDK.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the Internet idea, the Inkling tool-call report, the harness review requests and the DSpark decision.
- Thanks to [Unharnessed](https://github.com/unharnessed-agent/Unharnessed), the reference harness for this work: its turn loop, output handling and persistent state shaped the 24-round loop, the start-and-end output and `state`.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini, to [Exa](https://exa.ai) for web search, and to GitHub for the [octocat/Hello-World](https://github.com/octocat/Hello-World) repository used by the live network check.
- Thanks also to the projects behind the runtime and the app: Apple JavaScriptCore, which runs the Agent programs; curl, which makes the app's requests; [ExLlamaV3](https://github.com/turboderp-org/exllamav3); [Apple MLX](https://github.com/ml-explore/mlx); [PonyExl3](https://github.com/beamivalice/PonyExl3); [SwiftMath](https://github.com/mgriebling/SwiftMath).
