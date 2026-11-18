# MLXL3 Desktop 2.0.7

**A Decision mode for Liquid d1 decision models, and a one-click DFlash2 draft for Qwen3.8-27B.**

Desktop **2.0.7 / build 60** bundles MLXL3 engine **2.0.16**.

## New

### Decision mode

A fifth tab, **Decision**, sits beside Chat, Agent, Server and Lab. It runs [Liquid d1](https://huggingface.co/LiquidAI/d1-2B) decision models, which answer typed questions about a state in one forward pass, with no generated text, and give calibrated probabilities.

- **Playground**
  - **State.** Type text or JSON, or leave it empty and use media alone.
  - **Media.** Add pictures (PNG, JPEG, WebP; other formats are converted) and audio clips with the file picker, by drag and drop, or by recording from the microphone. Clips can last up to 5 minutes; video is not supported.
  - **Questions.** Build questions of every type: yes/no, choice, multiple choice, score, read text, number or date, estimate, box and boxes. You can also ask a question aloud by giving an audio clip as its instructions.
  - **Answers.** Probability bars for each option, the read span highlighted in the state, boxes drawn on the picture, and the raw JSON. The panel also shows the tokens read and the time taken.
- **Server**
  - **Local decision API.** One click starts `POST /v1/decide` on the chosen port, for this Mac or the local network. A new API key is generated at each start.
  - **Copy buttons.** You can copy the URL, the key, or a ready `curl` request for the current Playground draft.
- **Models**
  - The mode lists registered decision models (a config with `d1_attention`), and **Add folder…** adds an EXL3 export from disk.
  - d1-Omni takes text, pictures and audio; d1-2B takes text and pictures.
  - d1 EXL3 exports are not on the Hub yet, so there is no download in the app.
- **Memory**
  - Loading a decision model first stops the chat server and unloads the chat model.
  - Leaving the mode stops the decision engine and the decision server.

### Direct DFlash2 draft download for Qwen3.8-27B

- In **Generation settings → DFlash2**, a **Download the Qwen3.8-27B draft** button now fetches the pinned [incoai/Qwen3.8-27B-Splash](https://huggingface.co/incoai/Qwen3.8-27B-Splash) draft (about 1 GB, revision and file checksums verified by the engine) and enables DFlash2 with it. Before, the card offered only **Choose a local draft…**, unless you knew that the toggle also downloaded the draft.
- Other packed DFlash2 targets get a **Download the compatible draft** button.
- The draft buttons now use the same toolbar style as **Compare speeds**.

## Unchanged

- Chat, Agent, Server and Lab behave as before. In the Decision mode, the chat model menu and the MTP and DFlash2 toolbar controls are hidden.
- Conversations, settings and models are kept.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Decision mode check** `tests/decision-studio-check.swift`, 55 checks against a fake engine, added to `scripts/check-desktop.sh`:
  - request drafts: option and question order, validation messages;
  - model discovery and saved folders;
  - the `decide-bridge` lifecycle: answers, engine errors, a crash with recovery, a model that fails to load, audio refused on d1-2B;
  - audio import (AIFF to 16-bit WAV, WAV passes through);
  - `decide-serve` arguments, readiness and stop.
- **Real models, in a test build of the app:**
  - a Playground decision with d1-Omni-2B 8 bpw showed the answers, with 123 tokens read and none generated;
  - the Server tab started `decide-serve`: `/health` answered and `/v1/models` refused a request without the key; the server then stopped;
  - with Qwen3.8-27B loaded, the release build showed the new **Download the Qwen3.8-27B draft** button. The download itself was not run: it uses the engine's existing pinned `dflash-draft --target` path.
- **Engine.** The decision results are in the [engine 2.0.16 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.16). d1-2B passes all 12 fixtures. d1-Omni-2B gives identical answers on all 18, but three precision gates failed; they are documented there.
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum;
  - after mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.7 / 60;
  - its `build-info.json` names source `40440b0bc2a5`, with no uncommitted changes, and engine 2.0.16;
  - the bundled `mlxl3` is **byte-identical** to the published engine 2.0.16 binary (SHA-256 `e14a2317…0cffb2`).
- **Not verified:**
  - the full `scripts/check-desktop.sh` suite: only the new Decision check ran locally, because a pre-existing deprecation warning in `VoiceController.swift` fails `-warnings-as-errors` with the local macOS 27 SDK;
  - hosted CI on this commit had not completed when this was written;
  - SwiftUI behavior has no formal proof;
  - the Windows app has no Decision mode yet.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. It contains no Decision mode or DFlash2 changes.

## Source and download

- Compiled source `40440b0bc2a51adfb7c7e5ca327a34e1767c833e`.
- `MLXL3-Desktop-v2.0.7-b60-Apple-Silicon.dmg` · **248,127,829 bytes**, SHA-256 `f76d7b2205ed6be5bf43b35465d9c549d8007cf1f67f7ae485ea87f7edfd96c1`.
- `MLXL3-Desktop-v2.0.7-b60-Apple-Silicon.dmg.sha256` · 109 bytes, SHA-256 `c575f4384389613367729618a04c528936142edae3fa7c9bd9d5122a5fbaff51`.
- Bundled engine 2.0.16 (`mlxl3` SHA-256 `e14a2317765debbc9ef3a67269fbf426e700acfc9affa9ada00b770fd90cffb2`). The Swift app was built with the Command Line Tools macOS 26.5 SDK.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the feature requests, the design feedback and the release decision. Thanks to [Liquid AI](https://huggingface.co/LiquidAI) for the d1 and LFM3 models, [IncoAI Splash](https://github.com/incoai/splash) for the Qwen3.8-27B DFlash2 draft, and [Qwen](https://huggingface.co/Qwen) for the Qwen models. Thanks also to [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [MTPLX](https://github.com/youssofal/mtplx) and [SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, model format, speculative decoding research and math rendering.
