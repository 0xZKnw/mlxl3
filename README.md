<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/MLXL3Studio/Resources/AppIcon.png">
    <img src="apps/MLXL3Studio/Resources/LogoDark.png" alt="MLXL3 logo" width="128">
  </picture>
</p>

<h1 align="center">MLXL3</h1>

<p align="center">Local inference on Apple Silicon.</p>

<p align="center">
  <a href="https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.1">Download Desktop v2.0.1</a> ·
  <a href="https://mlxl3.0xzknw.tech/">Website</a> ·
  <a href="docs/README.md">Documentation</a> ·
  <a href="https://discord.gg/9hU46NVNnY">Discord</a>
</p>

MLXL3 runs EXL3 language and vision models on your Mac. A native SwiftUI app
brings together **Chat, Agent, Server and Lab**, backed by a Rust engine using
MLX and custom Metal kernels. The same engine powers the command-line client
and OpenAI-compatible HTTP API.

**Desktop 2.0.1 · Engine 2.0.10 · Apple Silicon · macOS 26.2+ · MIT**

![MLXL3 Desktop v2 with its ASCII aurora, conversation history and composer](docs/desktop/assets/mlxl3-ascii.png)

## What’s new in v2

The stable v2 brings together all **21 betas**, from the new interface to
project workspaces, vision, conversion and independent engine updates.

| Area | Included in v2 |
| --- | --- |
| **A new Desktop** | Animated violet, rose and amber ASCII aurora, native glass controls, a redesigned menu bar widget and smooth conversation transitions. |
| **Chat** | Streaming responses and reasoning, Markdown, code, math, HTML previews, attachments, vision, conversation branches, search and export. |
| **Agent Studio** | Isolated project copies, file editing, diffs, Git checkpoints, restore, ZIP export and reviewed changes applied to the original project. |
| **Models** | Recommended EXL3 checkpoints, model cards, resumable downloads, local imports and Quantization Studio. |
| **Server and Lab** | An OpenAI-compatible endpoint, live server logs, recorded generation metrics, charts and JSON export. |
| **Personalization** | System, Light and Dark appearances, Iris and custom accents, reading controls, project profiles, startup models and idle unloading. |

The ASCII characters transform within fixed cells as light flows across the
main window and widget. Motion responds to generation and follows Reduce
Motion. Settings, Models, MCP, Projects and Agent Studio share the same visual
language.

**New in 2.0.1:** the management screens follow the v2 design, generation
animation settles at twice the resting speed, and the Iris/amber/rose code
palette highlights embedded CSS, JavaScript, TypeScript and JSON inside HTML.
Chat and Agent Studio use the same syntax rules.

Read the [v2.0.0 release notes](docs/releases/desktop/release-v2.0.0.md) and
[v2.0.1 patch notes](docs/releases/desktop/release-v2.0.1.md) for the complete
changes, verification and known limits.

## Install and run your first model

Install the latest **stable Desktop and its bundled CLI**:

```sh
curl -fsSL https://mlxl3.0xzknw.tech/install.sh | bash
```

Or download the [Desktop DMG](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.1)
and drag **MLXL3 Desktop** into Applications. The distribution includes the
engine, MLX libraries, Metal assets and conversion environment. No separate
Python, Homebrew or MLX installation is required. Download model weights
separately and allow enough unified memory for the chosen checkpoint.

1. Open **Models → Hugging Face**, or import a complete local EXL3 folder.
2. Choose a model and quantization, download it, then load it from the model selector.
3. Start a conversation. Use **+** for attachments and tools, or switch to Agent, Server or Lab.

The app is **ad-hoc signed, not notarized**. If macOS blocks first launch,
approve this specific app in **System Settings → Privacy & Security**.
Agent project features also require Git from Apple's Command Line Tools.

Desktop **2.0.1 / build 54** bundles [engine 2.0.10](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.10).
Desktop and engine have separate Stable/Beta update channels in Settings.
Engine updates validate their identity, compatibility, hashes and architecture.
A managed engine that fails before startup can fall back to the bundled runtime.

## Desktop workspaces

### Chat

- Stream answers and separate reasoning with Markdown, tables, syntax-colored code and rendered math.
- Edit, regenerate or branch messages. Pin and search saved conversations, then export them.
- Attach text documents and PDFs with extracted text, source excerpts and page references. Scanned PDFs require prior text extraction.
- Send PNG, JPEG or WebP images with **LFM3-VL** using the picker, drag and drop or Command-V. Saved history retains image bytes after the original is removed.
- Open completed HTML in the inline **Code / Preview** view. Supported module CDNs, import maps and project CSS/JavaScript work in an isolated WebKit preview.
- Add MCP tools, structured JSON replies, project instructions, document retrieval, dictation or reply speech when needed.

Image input is limited to **eight images / 8 MiB per conversation**, with at
most 64 million source pixels per image. Text-only models cannot continue an
image conversation. Documents and images can be attached together.

### Agent

Each task works in an isolated project copy. **Agent Studio** combines a file
browser, code editor, diffs, Git history, checkpoints and ZIP export. Review a
revision before applying it to the original folder. Conflicting external edits
and changes made after review are refused.

File reading, writing and command execution have separate permissions. Native
macOS sandboxing confines the built-in tools to the task workspace. Commands
default to a five-second limit, adjustable from 1–30 seconds. External MCP
servers have their own permissions and trust boundary.

### Server and Lab

**Server** keeps one model resident behind an OpenAI-compatible API. Choose the
context and port, copy the endpoint and key, and inspect logs. Loopback access
is the default. Optional LAN access uses a generated bearer key over HTTP.
Desktop passes the key to the engine through a private pipe.

**Lab** displays recorded generations with decode and prefill rates, latency,
token counts, memory and runtime details. Compare samples or export JSON.
These are observations from actual sessions, not controlled benchmarks.

## Models and compatibility

EXL3 is a quantization format. A checkpoint also needs a supported architecture,
tensor layout, configuration, tokenizer and every weight shard. A single
`.safetensors` file is insufficient.

| Runtime family | Configuration type | Supported input / execution |
| --- | --- | --- |
| LFM2 / LFM2.5, dense and MoE | `lfm2`, `lfm2_moe` | Text, convolution, recurrent state and routed experts |
| LFM3-1.7B | `lfm3` | Text, gated attention, sliding window and seven-tap convolutions |
| LFM3-VL-2B | `lfm3_vl` | Text and images, vision encoder and projector |
| Qwen 3.5 / 3.6 / 3.8 | `qwen3_5`, `qwen3_5_moe` | Text, hybrid attention and dense/MoE models |
| Qwen 3.8 Flash Next | `qwen4_exp` | Text, SSD expert streaming and a bounded cache |
| Gemma 4 | `gemma4` | Text and native tool-call parsing |
| Ling / Bailing | `bailing_hybrid` | Text, routed experts and native tool-call parsing |

The Desktop catalogue includes these pinned exports:

| Checkpoint | Projection bpw | Repository |
| --- | ---: | --- |
| LFM3-1.7B Heretic NX | 8 | [0xzknw/LFM3-1.7B-Heretic-NX-EXL3-8bpw](https://huggingface.co/0xzknw/LFM3-1.7B-Heretic-NX-EXL3-8bpw) |
| LFM3-1.7B original | 8 | [0xzknw/LFM3-1.7B-EXL3-8bpw](https://huggingface.co/0xzknw/LFM3-1.7B-EXL3-8bpw) |
| LFM3-VL-2B | 8 | [0xzknw/LFM3-VL-2B-EXL3-8bpw](https://huggingface.co/0xzknw/LFM3-VL-2B-EXL3-8bpw) |
| LFM2.5-24B-A2B | 4 | [0xzknw/LFM2.5-24B-A2B-EXL3-4bpw](https://huggingface.co/0xzknw/LFM2.5-24B-A2B-EXL3-4bpw) |
| Qwen3.6-35B-A3B | 2.49 | [UnstableLlama/Qwen3.6-35B-A3B-exl3-2.49bpw](https://huggingface.co/UnstableLlama/Qwen3.6-35B-A3B-exl3-2.49bpw) |
| Qwen3.8-27B | 2.00 | [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3) |
| Qwen3.8 Flash Next | 2.05, SSD | [turboderp/Qwen3.8-Flash-Next-exl3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3) |

LFM3 text and vision recommendations select **8 bpw** by default. For vision,
use the calibrated 8-bpw export. The 4-bpw vision export is uncalibrated, and
4.49 bpw has not met the native feature-parity threshold. Projection bpw
excludes embeddings, output heads, scales and metadata. Model licenses apply
separately from MLXL3's license.

### MTP and DFlash

Compatible Qwen targets expose **MTP** or experimental **DFlash2** in Generation
settings. Matching heads and drafts are downloaded separately with integrity
checks. The modes are mutually exclusive and verify proposals against the
target before accepting tokens. MTP is remembered per model. Flash Next uses
its integrated head.

The tuner records a configuration for the current target, head, runtime,
context and Mac. A speculative path can increase memory use or run slower.
Hardware gates also apply. See the [Apple Silicon support matrix](docs/operations/memory-hardware/apple-silicon-stability-20261030.md)
and [engine release notes](docs/releases/engine/README.md).

### Safetensors-to-EXL3 conversion

**Models → Quantization Studio** inspects a local source or pinned Hugging Face
revision. Simple mode offers **3/4/6 bpw**. Expert mode adds calibration, head
precision, excluded modules and optional held-out quality checks. Conversion
unloads the interactive model and writes to a separate output directory.

The bundled adapters cover **LFM2, LFM2-MoE and Ling/Bailing**. Inference
support does not imply conversion support for LFM3, Qwen or Gemma. Follow the
[LFM conversion guide](docs/modeles/lfm/lfm26-local-quantization.md),
[LFM2.5 MoE guide](docs/modeles/lfm/lfm25-moe-quantization.md) or
[Ling conversion guide](docs/modeles/ling/ling-local-quantization.md).

## Native command-line inference

The installer adds `mlxl3` to your shell. For a DMG installation, use the bundled
executable directly:

```sh
export MLXL3_BIN="/Applications/MLXL3 Desktop.app/Contents/Resources/runtime/mlxl3"
"$MLXL3_BIN" runtime-info
"$MLXL3_BIN" register lfm3 /absolute/path/to/LFM3-1.7B-EXL3-8bpw
"$MLXL3_BIN" run lfm3 --prompt 'Explain how a ring buffer works.' --max-tokens 128
```

A complete checkpoint directory can also be passed directly. For vision:

```sh
"$MLXL3_BIN" run /absolute/path/to/LFM3-VL-2B-EXL3-8bpw \
  --image /absolute/path/to/photo.png \
  --prompt 'Describe this image.' --max-tokens 128
```

Use `mlxl3 hub` to browse, download or resume EXL3 models, and `mlxl3 mcp` to
manage tool servers. Downloads resolve immutable revisions, validate file
sizes and hashes, and publish only complete variants. Private or gated
repositories require a saved token and any necessary license acceptance.

## OpenAI-compatible local API

Start a server with a registered model:

```sh
"$MLXL3_BIN" serve lfm3 --host 127.0.0.1 --port 8000 --api-key local-secret
```

Then send a request from another terminal:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer local-secret' \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm3","messages":[{"role":"user","content":"Explain a mutex."}],"stream":true}'
```

Endpoints include `/v1/models`, `/v1/chat/completions`, `/v1/capabilities`,
`/health` and `/ready`. Chat Completions supports JSON and SSE streaming.
LFM3-VL accepts local base64 image data URLs. JSON object and schema response
modes validate generated output without grammar-constrained decoding.

Query `/v1/capabilities` for the active engine's scope. OpenAI-style `tools`
calls, custom stop sequences and audio input are unsupported. A non-loopback
host requires an API key. The CLI also accepts `--api-key-stdin` for private
key delivery.

## Local data and network access

Inference and conversation storage run locally. Managed models live in
`~/Library/Application Support/io.mlxl3.desktop/Models`. Imported checkpoints
can stay on another disk. History embeds image attachments. Keep backups
before downgrading to an older history format.

Hub actions contact Hugging Face. Update checks and downloads contact GitHub.
Optional MCP servers can receive tool arguments and make their own network
requests. MCP is off by default, and local stdio servers run with the user's
permissions. The shared configuration is `~/.config/mlxl3/mcp.json`.
Dictation and reply speech use macOS services and their permissions.

## Runtime and performance

Rust owns checkpoint validation, tokenization, templates, generation, streaming
and model state. EXL3 projections are evaluated from their quantized
representation without first exporting a dense BF16 model. **The shipped v2
backend is MLX**, including Flash Next. Independent native Metal and hybrid
backends remain in the source tree and are excluded from the v2 packages.

Performance depends on the checkpoint, quantization, context, prompt, output
budget, acceleration mode, Mac and memory pressure. Decode, prefill, time to
first token, total latency and memory measure different parts of the workload.

Recorded campaigns include:

- [Six-model Apple M5 / 24 GiB baseline](docs/measurements/v2-models-20261111/), with exact weights, source revisions, timings and quality checks.
- [Qwen MTP and DFlash2 observations](docs/measurements/v2-speculative-20261111/), including memory-pressure changes and comparison limits.
- [Flash Next SSD streaming](docs/performance/flash-dflash/flash-next-ssd-implementation.md) and [LFM2.5 measurements](docs/modeles/lfm/lfm25-measurements-2026-10-14.md).

These reports retain their original protocols and revisions. They are not fresh
benchmarks of Desktop 2.0.1 or guarantees for every Mac. The
[optimization journal](opti.md) preserves experiments, rejected approaches and
integration status. Physical release validation used Apple M5. M1–M4 and
macOS 26.2 did not receive the same hardware validation campaign.

## Build and verify

Development requires **Rust 1.89+**, **Swift 6.2+**, **Python 3.12**, Apple
Silicon and compatible Xcode Command Line Tools. The release environment pins
**MLX 0.32.2**. Python supplies build and conversion dependencies, while chat
inference uses the Rust engine.

```sh
git clone https://github.com/0xZKnw/mlxl3.git
cd mlxl3
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,bundle]' -c packaging/constraints-macos.txt
export MLXL3_BUILD_PYTHON="$PWD/.venv/bin/python"
export MLXL3_MLX_ROOT="$PWD/.venv/lib/python3.12/site-packages/mlx"
export MLXL3_ENGINE_BACKEND=mlx
scripts/build-macos-dmg.sh
```

The packaging script builds the engine, app and converter, then writes the DMG
to `dist/`. Set `MLXL3_MACOS_SDK` to select an installed compatible SDK.

Portable CPU/chat checks:

```sh
cargo fmt --check
cargo clippy --locked --all-targets --no-default-features --features chat -- -D warnings
cargo test --locked --no-default-features --features chat
```

Additional checks with the development dependencies installed:

```sh
cargo kani --lib --no-default-features
PYTHONPATH=src:apps/MLXL3Studio/Vendor/PonyExl3 .venv/bin/python -m pytest -q
MLXL3_TEST_PYTHON="$PWD/.venv/bin/python" scripts/check-desktop.sh
```

Kani verifies bounded properties of Rust helpers, not MLX, Metal, concurrency
or the complete application. CPU tests do not qualify physical GPU behavior.
See [CONTRIBUTING.md](CONTRIBUTING.md) for MLX checks, regression tests and PR
guidance, and the [release validation](docs/releases/release-v2.0.0-validation.md)
for checks actually run on the shipped v2 release.

## Documentation and community

- [Documentation index](docs/README.md) — engine, Desktop, model support, operations and measurements.
- [Desktop releases](docs/releases/desktop/README.md) and [engine releases](docs/releases/engine/README.md) — features, artifacts and validation limits.
- [Contributing](CONTRIBUTING.md) — setup, checks and focused changes. English and French contributions are welcome.
- [Issues](https://github.com/0xZKnw/mlxl3/issues/new/choose) and [Discord](https://discord.gg/9hU46NVNnY) — bug reports and usage questions.
- [Security](SECURITY.md) and [Code of Conduct](CODE_OF_CONDUCT.md) — private vulnerability reporting and community guidelines.

## Repository and credits

| Path | Contents |
| --- | --- |
| `apps/MLXL3Studio/` | SwiftUI application, resources and native UI checks |
| `native/src/`, `native/shaders/` | Rust engine and Metal kernels |
| `src/mlxl3_quantizer/` | Conversion adapters and calibration helpers |
| `scripts/`, `benchmarks/`, `tests/` | Build tools, measurement campaigns and regression checks |
| `website/` | Landing page, interactive ASCII aurora and Desktop installer |
| [`docs/`](docs/README.md), [`opti.md`](opti.md) | Documentation, numerical evidence and experiment history |

Created by [0xZKnw](https://github.com/0xZKnw). Thanks to
[Lucas Bauduin / HENK0O](https://github.com/HENK0O) for appearance, startup,
pinned conversations, reading controls and per-model MTP contributions.

Thanks to [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime,
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format,
[PonyExl3](https://github.com/beamivalice/PonyExl3) for conversion,
[MTPLX](https://github.com/youssofal/mtplx) and
[TensorFold](https://github.com/ashhart/TensorFold) for speculative decoding
references, and [SwiftMath](https://github.com/mgriebling/SwiftMath) for equation
rendering.

MLXL3 uses the [MIT license](LICENSE). Upstream components and model weights
retain their respective licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
for provenance and notices.
