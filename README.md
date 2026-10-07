# MLXL3 — EXL3 inference on macOS

**An open-source EXL3 inference engine for macOS and Apple Silicon, built with Rust, MLX and custom Metal kernels.**

[MLXL3 website](https://mlxl3.0xzknw.tech/)
· [Download MLXL3 Desktop](https://github.com/0xZKnw/mlxl3/releases/latest)
· [Validation scope](docs/v1-validation.md)
· [Optimization journal](opti.md)
· [Third-party notices](THIRD_PARTY_NOTICES.md)

MLXL3 runs supported EXL3 language models locally on an Apple Silicon Mac.
To run an EXL3 model on macOS, install MLXL3 Desktop, download or import a
compatible checkpoint, then load it and start a conversation. The desktop
distribution includes its native runtime; no Python or Homebrew installation
is required. Model weights are downloaded separately.

The project ships three pieces that share the same native engine:

- **MLXL3 Desktop**, a SwiftUI chat application with model management, Markdown,
  saved conversations, MCP tools and live performance metrics;
- **`mlxl3`**, a streaming terminal client and model-management CLI;
- **the Rust/Metal runtime**, which reads EXL3 checkpoints directly and executes
  them through MLX plus architecture-specific Metal kernels.

The downloadable DMG is self-contained: it includes the app, Rust engine, MLX
runtime and Metal assets. It deliberately contains no model weights.

> [!IMPORTANT]
> The current DMG is ad-hoc signed, not Developer ID signed or notarized. On
> first launch, macOS may require **Open Anyway** in **System Settings → Privacy
> & Security**. Do not disable Gatekeeper globally.

## Engine v1.4.2 candidate

An explicitly prepared **compact Q4 MTP draft projection** can accelerate
Qwen3.6-35B-A3B EXL3 without changing its target vocabulary or greedy results.
On M5, MTP2 decode improved **8.2% on the code prompt** and **11.6–15.3% in
the tuner** against engine1.4.1. The extra projection and ID map use **87.77 MiB**;
the full Q4 prototype used 272.81 MiB and regressed on code, so it is not a
general speed recommendation. These measurements use fixed token budgets,
not complete-task accuracy evaluations, and do not establish gains for other models.

MTP target-layer graphs are also submitted progressively on the measured
Qwen3.6-35B-A3B geometry on **Apple M5**. With the stock head, MTP2 code
decode improved **7.16–7.80%** versus the same 1.4.2 binary with submission
disabled; complete time fell **6.27–6.80%**, confirmed in opposite pass orders.
This result is separate from the compact-head gain above. The default applies
to MTP only; `MLXL3_QWEN_PIPELINE=0` disables it. Value `1` experimentally
extends submission to all Qwen trunk paths. Other geometries and devices keep
the previous default. See [protocols and limits](docs/engine-v1.4.2-pipeline.md).

A separate prompt-lookup experiment can draft continuations of repeated code
with `MLXL3_MTP_LOOKUP=1`. It is **off by default**: the initial copy speed
signal was not reproduced within the control-drift limits. The full target
verifies its proposals and the runtime repairs exact caches before continuing.
It falls back to the neural draft after a full rejection. See the
[Sushi/TensorFold review](docs/measurements/engine-v1.4.2/sushi-tensorfold-deepening.json)
for examined mechanisms and numerical compatibility limits.

Desktop sources also fix Send/Tune becoming disabled during background update
checks or downloads and refresh the composer when update state changes.
Installation still blocks inference. This UI fix requires a new Desktop build;
an independent engine update alone does not deliver it.

Protocols, rejected experiments, tests and limits are in the
[optimization journal](opti.md) and [validation evidence](docs/measurements/engine-v1.4.2/validation.json).

## v1.4.0

MTP automatically selects and downloads the pinned head for **Qwen3.8-27B**
or **Qwen3.6-35B-A3B**. Switching models unloads the previous head; the global
ON/OFF preference remains in force even with saved Tune MTP results. Desktop
build23 includes engine1.4.0; the independent engine update requires Desktop
1.4.0. [Release notes](docs/release-v1.4.0.md) and
[validation/measurements](docs/mtp-dense-1.4.0-validation.md).

## MTP and independent updates

Desktop now uses native **MTP** for Qwen3.5/3.6 checkpoints with a matching
MLX affine 4-bit/group64 head. The Qwen3.6-35B-A3B head downloads on demand,
from a pinned revision with full SHA-256 verification. Version 1 uses one
proposal per verification block in greedy mode (temperature 0 / top-k 1,
repetition penalty 1); other settings use ordinary decoding. Proposed tokens
are delivered only after target verification. DFlash is retained as a CLI
experiment and is disabled in Desktop on relaunch.

The app and engine have independent GitHub release channels: `vX.Y.Z` for
DMGs and `engine-vX.Y.Z` for `MLXL3-Engine-vX.Y.Z-arm64.tar.gz`. Engine updates
are validated against manifest/protocol/OS/app versions, file hashes, Mach-O
architecture and signatures, then activated through an atomic pointer outside
the app bundle. A managed engine that fails before ready is disabled and the
bundled engine is selected on the next load. Both channels can be updated in
one installation action. Publish engine releases with `--latest=false` so
older Desktop versions keep receiving app DMGs from GitHub's latest endpoint.

M3 branding fills the icon canvas, and whole-message copy is below each
message. See [release notes](docs/release-v1.2.0.md) and
[verification and limitations](docs/desktop-v1.2.0-validation.md).

## Platform and model support

The release build requires an **Apple Silicon Mac (M1–M5) running macOS 26.2 or
newer**. Development and performance validation currently happen on M5. M1–M4
use compatible Metal paths where M5 TensorOps are unavailable, but they have not
received the same physical performance campaign.

The native loader currently accepts these `config.json` model types:

| Family | Config type | Current scope |
| --- | --- | --- |
| Liquid AI LFM2 / LFM2 MoE | `lfm2`, `lfm2_moe` | Text chat, recurrent state, EXL3 dense and routed experts |
| Qwen 3.5 / 3.6 / 3.8 | `qwen3_5`, `qwen3_5_moe` | Text chat, hybrid Gated DeltaNet/attention, dense and MoE |
| Gemma 4 | `gemma4` | Text chat and native tool-call parsing; no image/audio input |
| Ling 3 / Bailing V3 | `bailing_hybrid` | Text chat and Ling-native MCP tool calls |

EXL3 is a file format, not a promise that every EXL3 repository is compatible.
The app inspects architecture, tensor inventory, expert count, shapes and kernel
constraints before loading. Unsupported layouts fail visibly instead of falling
back to a different numerical path. Mapped MoE projections at K=7 are currently
unsupported; ordinary dense K=7 projections remain supported.

## Measured performance

These are physical-engine measurements, not estimates. Results from different
rows use different workloads and must not be combined.

| Model and workload | Decode | Prefill / TTFT | Notes |
| --- | ---: | ---: | --- |
| Qwen3.6-35B-A3B EXL3 2.49 bpw, greedy, 48 generated tokens | **47.545 tok/s** latest median | Captured prefill 0.153–0.154 s | Apple M5, three alternating runs |
| Same target with experimental lossless DFlash2 | **74.645 tok/s** steady-state confirmation | Draft-context setup 0.006–0.007 s | **+43.8%** paired after the shared MoE route-preparation gain, five useful head rows/proposals per block, 86.7% accepted, exact sequences |
| Ling 3.0 Tiny EXL3 4 bpw, 84-token prompt / 128-token generation | **102.873 tok/s** best control median | **105.524 tok/s**, 796.23 ms TTFT | M5, battery-powered diagnostic campaign |
| LFM2.5-8B-A1B EXL3 3.10 bpw, historical 12-run warm campaign | **65.8 tok/s** paired median | **113.9 tok/s** on a 51-token prompt | 4.02 GB peak MLX allocation |

The DFlash2 number measures delivered output tokens from a complete draft →
select → exact target verify → accept → state commit loop. The optimized commit
costs about **1 ms**, down from 127–132 ms for restore-and-recompute. The target
sequence and all **80 recurrent/KV state arrays** were compared with ordinary
greedy execution for retained widths 1 through 8. Desktop v1.1.0–v1.1.3 exposed
this path for Qwen3.6-35B-A3B; v1.2.0 replaces it with MTP. The historical
DFlash CLI experiment requires separate draft weights; the benchmark rate is not a guarantee
of in-app speed or of performance on another Mac.

Peak MLX allocation is not the model file size, process RSS or total macOS
physical footprint. Unified memory also holds compiled graphs, caches, recurrent
state, scratch buffers, the UI and the operating system.

Full protocols, negative results and hardware conditions are preserved in
[`opti.md`](opti.md). That journal is authoritative when a headline number and
an older document disagree.

## Install MLXL3 Desktop

1. Download the latest `MLXL3-Desktop-…-Apple-Silicon.dmg` from
    [GitHub Releases](https://github.com/0xZKnw/mlxl3/releases/latest).
2. Open the DMG and drag **MLXL3 Desktop** to **Applications**.
3. Launch it. If Gatekeeper blocks the ad-hoc-signed build, approve this specific
   app in **Privacy & Security**.
4. Open **Models**, then either search Hugging Face or import an existing EXL3
   folder. Select the desired branch, tag or quantization before downloading.
5. Load the model and start a conversation.

### Optional native MTP for Qwen3.5/3.6/3.8

In **Generation → MTP**, enable the switch. For Qwen3.8-27B (~228 MiB) or
Qwen3.6-35B-A3B (~453 MiB), MLXL3 downloads and verifies the matching affine
4-bit head. Switching models cancels obsolete preparation and loads the
matching head before chat becomes available; OFF releases head and caches.
Downloads resume after cancellation. Other Qwen3.5-family models require a matching
MLX 4-bit/group64 head selected through the folder button; mismatched layouts
are rejected. The DMG contains neither target nor MTP weights.

MTP applies greedy settings and checks every proposal against the target.
The first request can take longer while Metal compiles kernels. A prefix
checkpoint retains both target state and MTP KV when the conversation's
encoded prefix and chunk boundary match. DFlash remains available to CLI
experiments; its former Desktop toggle has been replaced.

#### Experimental compact draft preparation

Preparation uses the development Python/MLX environment and supports unbiased
Default/MCG EXL3 heads. The native runtime reads the resulting files directly.
Automatic head downloads keep the stock projection. For the tested Qwen3.6
tokenizer, save the pinned [TensorFold draft ID list](https://github.com/ashhart/TensorFold/blob/cb2ebf0540f42604e2759b2ddef497861e928248/src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt)
as `draft-vocab.txt`, then use new disposable output folders:

```sh
python scripts/experimental-draft-head.py /path/to/EXL3-target /tmp/mtp-full
ln -s /absolute/path/to/MTP-head/config.json /tmp/mtp-full/config.json
ln -s /absolute/path/to/MTP-head/model.safetensors /tmp/mtp-full/model.safetensors
python scripts/experimental-draft-head.py /tmp/mtp-full /tmp/mtp-compact --draft-vocab /path/to/draft-vocab.txt
```

Select `/tmp/mtp-compact` with the MTP folder button and run **Tune MTP**.
The ID map is sorted, validated and padded to whole 64-row tiles; only draft
scores are pruned. Every proposed token is still verified by the full target.
`conversion.json` reports conversion, not validated parity or speed. The full
Q4 projection is an intermediate; original checkpoints are read only, and
existing output files or symlinks are refused.

No Python, Homebrew, Hugging Face CLI or separate MLX installation is required
for the DMG. Managed weights are stored under:

```text
~/Library/Application Support/io.mlxl3.desktop/Models
```

Models imported from Documents, Downloads or an external disk can trigger the
normal macOS Files & Folders permission dialog. Managed downloads avoid that
permission. The library can reveal a checkpoint in Finder, repair a moved path,
remove only its registration, or move its managed files to the Trash.

### Desktop behavior

- Return sends; Control-Return inserts a newline.
- **Attach** (Command-Shift-O) or drag PDF/text files onto the composer. Preview
  extracted text or remove a file before sending. TXT, Markdown, CSV, JSON and
  source code are supported in UTF-8 or UTF-16 with a BOM. PDF extraction keeps
  page numbers; scanned or password-protected PDFs require preparation first.
  Up to eight files can be attached per message (20 MiB per file, 256 KiB of
  extracted text per file, 512 KiB total). Their text is included in the model
  context and saved with the conversation, so the originals can be moved later.
  The model's context limit still applies; oversized prompts fail visibly.
- Reasoning, tool calls and final answers stream as separate visual phases.
- Markdown, tables, LaTeX and syntax-highlighted code render incrementally.
- Code blocks have a one-click copy action.
- Conversations and partial long generations are saved atomically.
- Stop cooperatively cancels the current generation while retaining the model.
- Eject releases the model and its Metal memory without deleting its files.
- The menu-bar panel shows physical footprint, active model, context and the
  latest decode/prefill/TTFT measurements.
- Settings provide French/English UI, context sizing, sampling controls and
  build-aware GitHub updates.

Conversation data lives at:

```text
~/Library/Application Support/io.mlxl3.desktop/conversations.json
```

The context setting is saved per model. `0` means the model-declared maximum,
or 32,768 when the checkpoint declares none. **Save and reload model** clears
engine cache, not chat history. Oversized prompts are rejected rather than
silently truncated.

## CLI

The bundled runtime is native and uses the same engine as Desktop. A source
build produces `target/release/mlxl3-rs`; installed builds may expose it as
`mlxl3`.

```bash
mlxl3 list
mlxl3 inspect /absolute/path/to/model
mlxl3 register my-model /absolute/path/to/model
mlxl3 run my-model
mlxl3 remove my-model
```

`mlxl3 run` opens a streaming multi-turn chat. Use `/clear` to reset the current
conversation and `/exit` to quit. A one-shot prompt is also supported:

```bash
mlxl3 run my-model --prompt "Explain speculative decoding simply." --max-tokens 256
```

Set `--max-tokens 0` to continue until EOS or the context limit. The interactive
CLI is greedy; Desktop exposes temperature, top-k and repetition penalty through
its native bridge.

### Hugging Face catalogue and downloads

Search, inspect and download EXL3 repositories without leaving the CLI:

```bash
mlxl3 hub search "Ling 3 EXL3"
mlxl3 hub details owner/repository
mlxl3 hub download owner/repository --revision 4bpw
```

Use `--folder` when a repository contains a selected variant in a subdirectory.
Downloads are pinned to a resolved commit and only fetch the selected variant.
Interrupted jobs are resumable:

```bash
mlxl3 hub pending all
mlxl3 hub resume DOWNLOAD_ID
mlxl3 hub discard DOWNLOAD_ID
```

Private or gated models use the locally saved Hugging Face token. Accept the
repository license on Hugging Face first.

## MCP tools and privacy boundary

Inference, caches and Metal kernels stay on the Mac. Network activity is a
separate, explicit boundary:

- Hugging Face is contacted only for catalogue/auth/download actions;
- GitHub is contacted for update checks and release downloads;
- remote MCP servers receive the tool arguments sent to them;
- local stdio MCP servers run with the current user's permissions and can make
  their own network requests.

MCP is off on first launch. The composer switch persists your choice across
restarts. Exa is preconfigured but is not contacted until MCP is enabled. When
enabled, search queries and fetched URLs are sent to Exa; model inference still
runs locally.

Configure additional local MCP servers with the CLI:

```bash
mlxl3 mcp add filesystem npx -y @modelcontextprotocol/server-filesystem "$HOME/Documents"
mlxl3 mcp list
mlxl3 mcp check --json
```

The shared configuration is `~/.config/mlxl3/mcp.json` and follows the common
`mcpServers` shape:

```json
{
  "version": 1,
  "mcpServers": {
    "exa": {
      "url": "https://mcp.exa.ai/mcp",
      "enabled": true
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"],
      "enabled": true
    }
  }
}
```

Commands are launched directly, without a shell. Browser OAuth and legacy
HTTP+SSE endpoints are not supported. Only configure MCP processes and remote
servers you trust.

## How the engine is structured

```text
SwiftUI Desktop / native CLI
             │
             ▼
Rust runtime: registry · tokenizer · templates · streaming · MCP
             │
             ▼
Model runtime: LFM2 · Qwen · Gemma 4 · Ling 3
             │
             ▼
MLX graph ops + MLXL3 Metal kernels
             │
             ▼
Serialized EXL3 weights in unified memory
```

The runtime implements EXL3 trellis packing/unpacking, all three procedural
codebooks, fused QMV for token decode, serialized QMM for prefill, grouped
QKV/gate-up projections, routed MoE execution and architecture-specific
attention/recurrent kernels. Ordinary inference never reconstructs a full dense
copy of each EXL3 weight.

ExLlamaV3 is the EXL3 format and numerical reference. MLXL3 is an independent
Apple-Silicon runtime, not an ExLlamaV3 fork. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for exact provenance and
licenses.

## Build from source

### Standalone Rust checks

Rust 1.89 or newer is required:

```bash
cargo build --release --locked
cargo test --locked
./target/release/mlxl3-rs --help
```

This build covers registry/checkpoint/codec tooling without linking MLX.

### Native inference build

Inference links to MLX 0.32.2 but does not embed Python. A local wheel is the
simplest way to supply the headers, dynamic libraries and metallib while
developing:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,bundle]"
export MLXL3_MLX_ROOT="$PWD/.venv/lib/python3.12/site-packages/mlx"
export MACOSX_DEPLOYMENT_TARGET=26.2
cargo build --release --locked --features mlx,chat
./target/release/mlxl3-rs run /absolute/path/to/exl3-model --max-tokens 256
```

Python supplies the optional development/conversion environment only. The
compiled CLI and distributed Desktop app do not start or embed Python.

### OpenAI-compatible API

Build with `--features mlx,chat`, then serve a registered EXL3 model on
loopback. The process keeps one model resident and exposes text-only Chat
Completions (including SSE streaming), model listing and a health endpoint:

```bash
./target/release/mlxl3-rs serve my-model --port 8000 --api-key local-secret
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer local-secret' -H 'Content-Type: application/json' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

The server binds to `127.0.0.1` by default. A non-loopback host requires
`--api-key`. OpenAI tool calls, image/audio inputs, stop sequences and output
constraints are rejected because the native bridge does not support them.

### Desktop and DMG

Building the app requires Swift, Xcode Command Line Tools, a compatible macOS
SDK, Rust and the MLX development files above:

```bash
./scripts/build-macos-app.sh
open "dist/MLXL3 Desktop.app"

./scripts/build-macos-dmg.sh
```

The second script creates an ad-hoc-signed, self-contained Apple Silicon DMG in
`dist/`.

## Verification and benchmarks

The default checks are intentionally small enough for CI:

```bash
cargo fmt --all -- --check
cargo clippy --locked --all-targets --features chat -- -D warnings
cargo test --locked --features chat
cargo kani --lib --no-default-features
python -m pytest -q
scripts/check-desktop.sh
```

Kani verifies bounded pure-Rust safety and parser/shape properties. It cannot
prove MLX, Metal shaders or their FFI. Those paths use exact physical-GPU
differential tests against reference operations and imposed-token model runs.
Passing either class of check is not a proof that the entire application has no
bugs.

To reproduce the current Qwen DFlash2 end-to-end campaign, place the target and
draft package at the paths named by the ignored test, then run:

```bash
export MLXL3_MLX_ROOT="$PWD/.venv/lib/python3.12/site-packages/mlx"
MLXL3_DFLASH_TOKENS=48 MLXL3_DFLASH_REPEATS=3 MLXL3_DFLASH_PROPOSALS=5 \
  cargo test --release --features mlx,chat \
  benchmarks_dflash_end_to_end_greedy -- --ignored --nocapture
```

The test alternates ordinary greedy and DFlash2, compares every emitted token,
and reports acceptance plus draft/target/commit timing. Results are hardware,
power, temperature, prompt and checkpoint dependent.

## Optional EXL3 conversion

The installed app consumes EXL3 checkpoints; it does not include a quantizer.
The developer environment retains the Python/PonyExl3 conversion workflow and
exact Metal trellis-search optimizations for K=2 through K=8, including 2 bpw.
These kernels accelerate conversion without changing the calibration recipe or
float32 error metric.

- [LFM local conversion guide](docs/lfm26-local-quantization.md)
- [Ling local conversion guide](docs/ling-local-quantization.md)
- [Metal quantization measurements](docs/metal-quantization-optimization.md)
- [CUDA-to-Metal kernel inventory](docs/kernel-port.md)

## Repository map

```text
apps/MLXL3Studio/   SwiftUI Desktop application
native/src/         Rust runtime, loaders, model implementations and protocols
native/shaders/     Custom Metal inference kernels
src/mlxl3_quantizer Optional developer-only EXL3 conversion helpers
scripts/            Build, packaging, release and validation tools
docs/               Audits, release notes and focused investigations
opti.md              Append-only optimization experiments and decisions
```

Local model weights, build products and benchmark artifacts are not committed.

## Known limits

- Apple Silicon/macOS only for native inference and Desktop.
- No multimodal Gemma input.
- DFlash2 is opt-in, greedy-only and validated for Qwen3.6-35B-A3B on M5;
  the separate draft weights are not bundled.
- Performance on M1–M4 is not inferred from M5 measurements.
- The release is not notarized.
- MCP processes are trusted external tools, not a sandbox.
- Markdown/LaTeX rendering aims for robust chat output, not complete browser or
  TeX compatibility.

## License

MLXL3 is released under the [MIT License](LICENSE). Components and algorithms
adapted from upstream projects retain their respective notices in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and `LICENSES/`.
