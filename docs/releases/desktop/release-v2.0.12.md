# MLXL3 Desktop 2.0.12

**A better Inkling 2 Mini at the same size, a DSpark switch, and an engine that no longer slows down or grows during long answers.**

Desktop **2.0.12 / build 65** bundles MLXL3 engine **2.0.21**.

## New

### Inkling 2 Mini R3

- **Same download, better quality.** The recommended Inkling 2 Mini now installs revision `0a455840` of [`0xzknw/Inkling-2-Mini-EXL3-3.33bpw`](https://huggingface.co/0xzknw/Inkling-2-Mini-EXL3-3.33bpw). It is still 14.8 GB, with a **17% lower KL divergence** against the original model (0.0280 against 0.0336) and higher top-1 agreement (94.66% against 94.22%).
- **How it was improved.** The new version spends the same bits differently: 4 bits for the down projection of every expert, 3 bits for gate and up.
- **What you need to do.** Download the model again from the recommended models.
  - The previous version stays on your disk until you remove it.
  - Desktop 2.0.11 keeps downloading the previous version.

### DSpark

- **A DSpark switch.** When a model ships a DSpark drafter, as Inkling 2 Mini does, the inspector shows an **Accélération DSpark / DSpark acceleration** card with a switch. DSpark is on by default and the choice is remembered.
- **DSpark statistics.** Under each answer, the metrics now show the accepted drafts and the tokens each verified block delivered, for example `64/120 · DSpark 53% · 2.60 tok/block`.
- **DSpark adapts to the text.** Engine 2.0.21 picks the block width from the acceptance it measures during the answer.
  - On long answers sampled at temperature 1, DSpark used to be about 25% slower than plain decoding; it is now close to plain decoding there.
  - On short and greedy answers it stays faster.

### Memory

- In engine 2.0.20 the **engine RAM** shown under each answer grew by 2–4 GB during a long Inkling 2 answer. Engine 2.0.21 limits MLX's buffer cache to 1 GiB while Inkling 2 is loaded; the growth is now about 1.5 GB with DSpark and 0.4 GB without.

## Performance

Apple M5 MacBook, 24 GB, AC power; Inkling 2 Mini R3; see the [engine 2.0.21 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.21) for every run.

| Answer | Plain decoding | DSpark in 2.0.11 | DSpark in 2.0.12 |
|---|---|---|---|
| Long essay sampled at temperature 1 | ≈ 18.2 tok/s | ≈ 13.8 tok/s | ≈ 16.7–19.6 tok/s |
| Short greedy answers | ≈ 19.6 tok/s | ≈ 26.5 tok/s | ≈ 26.9 tok/s |

Engine memory during a 1,200-token answer with DSpark: +3.6 GB in 2.0.11, +1.5 GB in 2.0.12 (+0.4 GB with DSpark off).

## Limits for Inkling 2 Mini

- **Memory.** The model keeps about 15 GB resident and was run on a 24 GB Mac only.
- **No image input** and **no MCP tools**, as in 2.0.11.
- **Prompt cache.** The prompt is not reused between turns, so a second turn evaluates the whole conversation again.

## Unchanged

- Everything else is as in [2.0.11](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.11).
- Conversations, settings and models are kept.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Desktop checks** (`scripts/check-desktop.sh`, Command Line Tools toolchain, Swift 6.4.2, with the macOS 26.5 SDK): both halves that CI runs passed, including speculative sampling (10,029 assertions, with the new DSpark checks: `dspark_supported` decoded and unknown on older engines, the `dspark` request field encoded on and off, the DSpark acceptance summary and its refusal of missing or inconsistent counters) and the recommended-models UI.
- **Engine at the bridge**, with the released binary: Inkling 2 Mini R3 answered greedy and sampled prompts and a second turn; a request with DSpark off runs without DSpark statistics. Other models give token hashes identical to engine 2.0.20. Details are in the engine notes.
- **Model on Hugging Face:** the six large files of revision `0a455840` match the export's sizes and SHA-256; the model card and `quantization_config.json` are byte-identical to the local copies; the engine's hub reports the revision as supported (14,790,037,062 bytes). A full download of the new revision was not repeated on this Mac.
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum;
  - after mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.12 / 65;
  - its `build-info.json` names source `c1f371b93ef1`, with no uncommitted changes, and engine 2.0.21;
  - its four runtime files are byte-identical to those of the engine 2.0.21 archive (`mlxl3` SHA-256 `3ed1d4e7…0386`).
- **Not verified:** the DSpark switch and the new model were not used by hand in the shipped app (the switch, request field and statistics were checked with the test harness, and the model at the bridge). SwiftUI behavior has no formal proof.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. It contains none of these changes: the Windows app uses its own engine and does not run Inkling 2 Mini.

## Source and download

- Compiled source `c1f371b93ef19432b71438d3fd374ca82929a092`.
- `MLXL3-Desktop-v2.0.12-b65-Apple-Silicon.dmg` · **254,976,397 bytes**, SHA-256 `d52df49ada1f436f8d4e9c39aaa38836882d5752fd2b6e52efc170ac16bd1730`.
- `MLXL3-Desktop-v2.0.12-b65-Apple-Silicon.dmg.sha256` · 110 bytes, SHA-256 `7acad0587f861dbdffc8ec95237d0fae77274464f0b3dae3df211be55c261d22`.
- Bundled engine 2.0.21 (`mlxl3` SHA-256 `3ed1d4e79d6abbf2700f7bf7b6b64eaf63d22e272ee1cc774b793550e0084386`, source `c1f371b93ef1`). The Swift app was built with the Command Line Tools toolchain (Swift 6.4.2) and its macOS 26.5 SDK.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the reports from the app (DSpark not available, slow decoding, growing memory), the request to add DSpark to the app, and the release decision.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini and its DSpark drafter.
- Thanks also to [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [Apple MLX](https://github.com/ml-explore/mlx), [PonyExl3](https://github.com/beamivalice/PonyExl3) and [SwiftMath](https://github.com/mgriebling/SwiftMath), for the EXL3 format and quantizer, the runtime, the Metal EXL3 port, and math rendering.
