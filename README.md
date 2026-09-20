# MLXL3

**Native EXL3 inference for Apple Silicon, built with Rust, MLX and custom Metal kernels.**

[Download MLXL3 Desktop v1.0.3](https://github.com/0xZKnw/mlxl3/releases/latest)
· [Validation scope](docs/v1-validation.md)
· [Optimization journal](opti.md)
· [Third-party notices](THIRD_PARTY_NOTICES.md)

MLXL3 runs supported EXL3 language models locally on Apple Silicon. The project
ships three pieces that share the same native engine:

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
| Qwen3.6-35B-A3B EXL3 2.49 bpw, greedy, 48 generated tokens | **48.325 tok/s** median | Captured prefill 0.152–0.155 s | Apple M5, three alternating runs |
| Same target with experimental lossless DFlash2 | **54.456 tok/s** median | Draft-context setup 0.008 s | **+12.7%**, 61.9% accepted, exact 48-token sequence in all runs |
| Ling 3.0 Tiny EXL3 4 bpw, 84-token prompt / 128-token generation | **102.873 tok/s** best control median | **105.524 tok/s**, 796.23 ms TTFT | M5, battery-powered diagnostic campaign |
| LFM2.5-8B-A1B EXL3 3.10 bpw, historical 12-run warm campaign | **65.8 tok/s** paired median | **113.9 tok/s** on a 51-token prompt | 4.02 GB peak MLX allocation |

The DFlash2 number measures delivered output tokens from a complete draft →
select → exact target verify → accept → state commit loop. The optimized commit
costs about **1 ms**, down from 127–132 ms for restore-and-recompute. The target
sequence and all **80 recurrent/KV state arrays** were compared with ordinary
greedy execution for retained widths 1 through 8. DFlash2 is currently an
**experimental benchmark path** and is not automatically enabled by Desktop or
the CLI.

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
MLXL3_DFLASH_TOKENS=48 MLXL3_DFLASH_REPEATS=3 \
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
- No automatic DFlash2 activation yet; its current integration is an exact
  physical benchmark path.
- Performance on M1–M4 is not inferred from M5 measurements.
- The release is not notarized.
- MCP processes are trusted external tools, not a sandbox.
- Markdown/LaTeX rendering aims for robust chat output, not complete browser or
  TeX compatibility.

## License

MLXL3 is released under the [MIT License](LICENSE). Components and algorithms
adapted from upstream projects retain their respective notices in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and `LICENSES/`.
