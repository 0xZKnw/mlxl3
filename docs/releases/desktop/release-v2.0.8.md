# MLXL3 Desktop 2.0.8

**Download Liquid d1 from the recommended models, load it in the Decision mode, and a simpler Decision page.**

Desktop **2.0.8 / build 61** bundles MLXL3 engine **2.0.17**.

## New

### d1 and d1-Omni in the recommended models

- **Models → Hugging Face → Recommended** has a new **Decision** family with two pinned downloads:
  - [d1](https://huggingface.co/0xzknw/d1-2B-EXL3-8bpw) (`0xzknw/d1-2B-EXL3-8bpw`, revision `0e99f8d8`): text and pictures, about 2.7 GB;
  - [d1-Omni](https://huggingface.co/0xzknw/d1-Omni-2B-EXL3-8bpw) (`0xzknw/d1-Omni-2B-EXL3-8bpw`, revision `54ded3cf`): text, pictures and audio, about 3.2 GB.
- These are our 8 bpw EXL3 conversions of [LiquidAI/d1-2B](https://huggingface.co/LiquidAI/d1-2B) and [LiquidAI/d1-Omni-2B](https://huggingface.co/LiquidAI/d1-Omni-2B), now published on Hugging Face with their license, a modification notice and the upstream model card.
- When a download ends, the message says to load the model in the Decision mode.

### Decision models load in the Decision mode

- A downloaded d1 model appears in the Decision mode's model menu right away.
- In **My library**, decision models are labelled **Decision** and their button is **Open in Decision**: it switches to the Decision mode and loads the model.
- Decision models no longer appear in the chat model menu or in the startup model setting, and the app never picks one as the chat model. Before, a downloaded d1 model could be selected for chat, where it cannot answer.

### A simpler Decision page

- **Header.** The model menu, its inputs (text, pictures, audio) and a **Load** button sit in one row. Once loaded, an eject button frees the memory. The menu also offers **Download more d1 models…** and **Add folder…**.
- **No model yet.** The page shows the two recommended d1 models; one click opens their download page.
- **Input.** One text box takes text or JSON: JSON is detected on its own, so the Text/JSON switch is gone. Pictures and sounds are added with **Picture or sound**, **Record** or drag and drop.
- **Questions.** Each question fits in one card: name, type, question, microphone for a spoken question, and the options when the type needs them.
- **Examples.** The Examples menu fills in a support ticket, an invoice (text, number and date reads) or an order written as JSON.
- **Decide** stays on ⌘↩ and loads the model if needed.

## Fixes

- A JSON state is now sent exactly as written. Before, the app re-encoded it with its keys sorted alphabetically, so d1 read the fields in a different order from the one typed (d1 reads object keys in order, as its reference does). Numbers also keep their spelling (`2.50` stays `2.50`).

## Unchanged

- Chat, Agent, Server and Lab behave as before. The Server tab of the Decision mode is unchanged.
- Conversations, settings and models are kept. A d1 folder added by hand in 2.0.7 stays listed.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **In the release build of the app** (a copy with its own bundle identifier, registry and models folder, so nothing touched an existing library):
  - with no decision model, the Decision page showed the two recommended d1 models; clicking d1 opened its Hugging Face page at revision `0e99f8d8` (2.73 GB, `lfm3_vl` supported);
  - **Download** fetched the model; its `model.safetensors` and `calibration.json` are byte-identical to the validated export;
  - **My library** labelled it **Decision**; **Open in Decision** switched modes, loaded it and showed **Loaded**;
  - **Decide** on the support ticket example: refund yes 99.7%, billing 69.8%, urgency medium 62.6%, 123 tokens read, 0 generated, 1,631 ms. These match the engine command line with calibration (and differ from a folder without `calibration.json`: billing 77.7%);
  - the Chat mode showed **Choose a model**: the d1 model, although the only one in the library, was neither selected nor loaded.
- **Decision mode check** `tests/decision-studio-check.swift`: 78 checks (55 before) against a fake engine, now also covering JSON detection, the JSON state sent as written (key order, number spelling, one bridge line), the three examples, 2.0.7 drafts, question naming, the pinned recommended revisions and decision-model routing.
- **Engine.** See the [engine 2.0.17 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.17): both repositories downloaded with the engine give files identical to the validated exports and decisions identical to them, text and audio.
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum;
  - after mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.8 / 61;
  - its `build-info.json` names source `6f8e4b3d2ee6`, with no uncommitted changes, and engine 2.0.17;
  - the bundled `mlxl3` is **byte-identical** to the published engine 2.0.17 binary (SHA-256 `7c6aefb8…9ec2`).
- **Not verified:**
  - the d1-Omni download inside the app (it was downloaded and checked with the engine command line only);
  - the full `scripts/check-desktop.sh` suite: only the Decision check ran locally, because a pre-existing deprecation warning in `VoiceController.swift` fails `-warnings-as-errors` with the local macOS 27 SDK;
  - hosted CI on this commit had not completed when this was written;
  - SwiftUI behavior has no formal proof.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. It contains none of these changes: the Windows app has no Decision mode.

## Source and download

- Compiled source `6f8e4b3d2ee604c3cb2fffa60c65dba7c6ee07f1`.
- `MLXL3-Desktop-v2.0.8-b61-Apple-Silicon.dmg` · **253,757,185 bytes**, SHA-256 `a5b6a5c98bc83ccce3061dc49c22eb70a76983c3ac998f096e4470564c4b42c5`.
- `MLXL3-Desktop-v2.0.8-b61-Apple-Silicon.dmg.sha256` · 109 bytes, SHA-256 `11cae8ae8b8240326dd20462567fe4ed4ccbcb0741a485e4aea188edec22a208`.
- Bundled engine 2.0.17 (`mlxl3` SHA-256 `7c6aefb85f45bd19a12ba12e1cb2f86636682f4da3a22148763bbf07239f9ec2`). The Swift app was built with the Command Line Tools macOS 26.5 SDK.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the feature request and the release decision. Thanks to [Liquid AI](https://huggingface.co/LiquidAI) for d1-2B, d1-Omni-2B and the LFM3 architecture, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format and converter, and [Hugging Face](https://huggingface.co) for hosting the models. Thanks also to [Apple MLX](https://github.com/ml-explore/mlx), [PonyExl3](https://github.com/beamivalice/PonyExl3), [MTPLX](https://github.com/youssofal/mtplx) and [SwiftMath](https://github.com/mgriebling/SwiftMath) for the runtime, the Metal EXL3 port, speculative decoding research and math rendering.
