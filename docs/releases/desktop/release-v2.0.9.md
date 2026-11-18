# MLXL3 Desktop 2.0.9

**A DFlash2 panel like the MTP one, DFlash2 and Tune that follow your sampling settings, and a Tune that keeps the fastest mode.**

Desktop **2.0.9 / build 62** bundles MLXL3 engine **2.0.19**.

## New

### DFlash2 in the toolbar, like MTP

- The header's **Tune DFlash2** button is replaced by a **DFlash2 On/Off** entry, built like the **MTP** one.
  - It shows a spinner while the draft is prepared or tuned.
  - It opens the same settings as the generation inspector: the enable switch, the state, the mode (Auto, 2 tokens, 7 tokens), **Compare speeds**, the results, and the draft options.
  - The draft options are the Qwen3.8-27B draft download and choosing a local draft, under **advanced options**.
- The inspector's DFlash2 card shows the same panel.

### DFlash2 and Tune follow your sampling settings

- **Before.** DFlash2 always forced greedy decoding: temperature, top-k, top-p and penalties were locked while DFlash2 was on.
- **Now.** With engine 2.0.19, which announces it supports sampling, DFlash2 uses your settings, as MTP already did.
  - With a preset above temperature 0, such as Qwen3.8's Thinking and Instruct, DFlash2 samples.
  - With temperature 0 and no penalty, it stays greedy.
- **Tune** measures with the same settings, and its results say "Measured with your sampling settings".
- With an older engine, DFlash2 stays greedy as before.

### Tune keeps the fastest mode

- In 2.0.8 with engine 2.0.18, Tune could mark **Auto** "Output differs from baseline" and keep **7 tokens**, even when Auto was faster. The divergence was a tie between two equally likely tokens.
- Tune now accepts such near ties and says so under the row: "Near tie accepted · gap …".

## Performance

Qwen3.8-27B on an Apple M5 MacBook with 24 GB (details in the [engine 2.0.19 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.19)):

- **Greedy (temperature 0).** With Tune now keeping Auto, DFlash2 uses the tree: +21 to +25% on prose and +5% on code over the chain that "7 tokens" ran (engine 2.0.18 measurements). Speed is about 25 tok/s on prose and 19 on code.
- **Sampling (Qwen3.8 Thinking preset).** DFlash2 now samples instead of forcing greedy output, at about 19–20 tok/s.
  - The tree is 12–13% faster than a sampled chain.
  - The bounded relaxation adds about 4% more accepted tokens per block.
  - Sampling stays slower than greedy because fewer draft tokens are accepted. For the highest speed, set the temperature to 0.

## Fixes

- The MTP toolbar button no longer shows a spinner, nor the DFlash2 progress, while DFlash2 is being tuned.

## Unchanged

- Chat, Agent, Decision, Server and Lab behave as before. Conversations, settings, models and saved Tune results are kept.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Desktop checks** (`scripts/check-desktop.sh`, full suite on the release source, Xcode toolchain with the macOS 26.5 SDK): every check passed, including:
  - speculative sampling: 10,006 assertions over 1,944 capability and intent states. They cover DFlash2 sampling with and without the engine flag, a sampled DFlash2 generation and a Tune request sent with the chat settings, and decoding of the sampled and near-tie rows;
  - MTP and DFlash2 layout: 48 toolbar and panel cases, French and English, 200–320 points;
  - the DFlash2 and MTP Tune checks;
  - 39 Python tests and the end-to-end checks (419 s).
- **Engine.** See the [engine 2.0.19 notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.19): the bundled binary passed the bridge checks, and both DFlash2 Tunes, greedy and sampled, keep Auto on Qwen3.8-27B.
- **Release DMG:**
  - `hdiutil verify` reported a valid checksum;
  - after mounting, the app passed `codesign --verify --deep --strict` and reports 2.0.9 / 62;
  - its `build-info.json` names source `2d74600037a8`, with no uncommitted changes, and engine 2.0.19;
  - the bundled `mlxl3` is **byte-identical** to the published engine 2.0.19 binary (SHA-256 `127c6691…783a`).
- **Not verified:**
  - the new DFlash2 panel and the sampled Tune were not exercised by hand in the shipped app; they were checked through the Desktop test suite and at the engine bridge;
  - hosted CI on this commit had not completed when this was written;
  - SwiftUI behavior has no formal proof.

## Windows

The Windows installer is built and attached by the `windows-desktop` workflow after publication. It contains none of these changes: the Windows app uses its own engine and has no DFlash2.

## Source and download

- Compiled source `2d74600037a860a2f08f4a8f3ec4470686e31ae8`.
- `MLXL3-Desktop-v2.0.9-b62-Apple-Silicon.dmg` · **247,880,317 bytes**, SHA-256 `0d4d5c8d59a1382e613e570cdfc584a7e5340fae659fc3ed797031d25a4976be`.
- `MLXL3-Desktop-v2.0.9-b62-Apple-Silicon.dmg.sha256` · 109 bytes, SHA-256 `865f21460e8bf591abb2466c4c69ecea7d94d9ac9e6d2202cb5619ef0ecf23c2`.
- Bundled engine 2.0.19 (`mlxl3` SHA-256 `127c6691e727fd7e5379dca21dd42ec81f10dbf0bdbaa9d79ae6a9f16c2f783a`). The Swift app was built with the Xcode toolchain and the Command Line Tools macOS 26.5 SDK.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the requests (the DFlash2 panel like MTP, Tune with sampling, the near-tie rule) and the release decision.
- Thanks to [incoai Splash](https://github.com/incoai/splash) for the DFlash2 draft, and for RFlash, which inspired the relaxed sampled verification.
- Thanks to [TensorFold](https://github.com/ashhart/TensorFold) for the best-first selector tree, and to [Qwen](https://huggingface.co/Qwen) for Qwen3.8-27B.
- Thanks also to [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3) and [SwiftMath](https://github.com/mgriebling/SwiftMath), for the runtime, the EXL3 format and converter, the Metal EXL3 port, and math rendering.
