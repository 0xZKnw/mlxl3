# mlxl3

`mlxl3` is an independent EXL3 inference and conversion engine for Apple Silicon.
The EXL3 CUDA implementation in ExLlamaV3 is the format and numerical source of
truth. The runtime is being implemented directly with MLX and custom Metal
kernels; PonyExl3 is kept only as a compatibility and performance comparison.

The working runtime now contains:

- bit-exact EXL3 trellis packing and unpacking on CPU;
- the three procedural EXL3 codebooks (`default`, `mcg`, and `mul1`);
- JIT-compiled Metal pack and fused unpack/decode kernels;
- a CUDA-style 128-thread tiled QMV kernel for autoregressive decode;
- automatic ragged QKV and gate/up fusion for compatible EXL3 projections;
- a mapped two-launch `SwitchGLU` path for selected MoE experts;
- vector and 32/64x64 simdgroup-matrix QMM kernels that keep weights serialized;
- a standard EXL3 safetensors loader on top of MLX-LM architectures;
- CPU-vs-Metal conformance tests for every bit width from 1 through 8.

The current end-to-end target is the official LFM2.5-8B-A1B EXL3 checkpoint at
3.10 bpw. MLXL3 loads all 2,179 quantized modules, including 2,112 routed expert
projections, and runs the model entirely through MLX/Metal. Ling conversion is
paused because a full no-training EXL3 conversion is much slower to iterate on.

Gemma 4 26B-A4B EXL3 is also supported for text chat. The loader maps its
individual experts to grouped Metal projections, applies GeGLU, respects padded
checkpoint dimensions, and loads its separately quantized output head. Gemma's
reasoning channels and generation-config end tokens are handled by both clients.
The local 3.54-bpw checkpoint has been checked with full short responses and an
expert-level reconstruction comparison; image/audio input is not implemented.
Gemma decode also groups padded gate/up projections and compiles its MLP/router
graphs. `benchmarks/benchmark_gemma_fusion.py` compares the optimized and reference
paths on identical tokens with fresh caches, alternating execution order and
checking full-vocabulary logits. This measures synchronized engine decode rather
than UI streaming throughput. The GeGLU rotation fusion remains disabled.
An experimental SIMD scaled-Hadamard kernel is available with
`MLXL3_SCALED_HADAMARD=1`. It preserves MLX's intermediate FP16 rounding and is
covered by exact comparison tests. It remains opt-in: five paired 192-token
Gemma CLI generations gave identical tokens but only a 2.2% median paired gain,
with substantial timing variation. `benchmarks/benchmark_gemma_cli.py` reproduces
that comparison. `benchmarks/profile_gemma_decode.py` counts and times isolated
stages with explicit synchronization; those diagnostic times are not GPU timings
and must not be compared directly with streaming throughput.

On this 10-core M5, the LFM fixture reaches a paired 12-run warm median of 65.8
generated tokens/s and a 4.02 GB peak allocation. The matching MLX 8-bit model
reaches 58.9 tokens/s and 9.04 GB. Prefill reaches 113.9 tok/s on a 51-token
prompt; larger-batch segmented GEMM remains future work.

## Local model CLI

After package installation, the per-user model registry is stored in
`~/.config/mlxl3/models.json`.

List locally registered models:

```bash
mlxl3 list
```

Start an interactive, streaming terminal chat by model name:

```bash
mlxl3 run lfm2.5-8b-a1b
```

Enter `/clear` to reset chat history and `/exit` to quit. Every response is
streamed with a magenta `Réflexion` section separated from the final `Réponse`,
then followed by TTFT, prefill tok/s, decode tok/s, token counts, and peak MLX
memory. Questions and final answers remain in context for the lifetime of the
interactive process.

For a single streamed response without entering the REPL:

```bash
mlxl3 run lfm2.5-8b-a1b "Explique-moi simplement pourquoi le ciel est bleu."
```

Register another standard EXL3 directory under any local name:

```bash
mlxl3 register MY_MODEL /absolute/path/to/exl3-model
```

Or download and register a public Hugging Face checkpoint in one command:

```bash
mlxl3 download UnstableLlama/Qwen3.6-35B-A3B-exl3-2.49bpw \
  --name qwen3.6-35b-a3b
```

Use `--revision 3.10bpw` when a repository stores BPW variants in branches.
Managed downloads live under
`~/Library/Application Support/io.mlxl3.desktop/Models` by default.
Those managed models never require access to Documents. For an existing model
stored in Documents, Downloads, or on an external disk, macOS can display its
native Files & Folders permission prompt when MLXL3 first loads the checkpoint.

Useful generation overrides are `--max-tokens`, `--temperature`, `--top-k`,
`--repetition-penalty`, and `--system`.

All prompt sizes stay on the serialized Metal QMM path; MLXL3 never reconstructs
a dense weight during ordinary inference. Cached chat prefixes use a memory-aware
512/1024/2048-token step, selecting wider and faster blocks when unified-memory
headroom allows it. The choice can be pinned for experiments with
`MLXL3_PREFILL_STEP_SIZE`.

## Native macOS app

MLXL3 Desktop is a native SwiftUI application for Apple Silicon Macs running
macOS 26.2 or newer. It uses the system
Liquid Glass materials and talks directly to the local MLXL3 Metal runtime. The
model stays resident while answers stream; reasoning, final answers, TTFT,
prefill/decode throughput, and peak memory are displayed separately. Context is
kept per conversation and saved atomically under
`~/Library/Application Support/io.mlxl3.desktop/conversations.json`, including
partial long generations. Press Return to send and Control-Return to add a line.
The eject button releases the active model and its Metal memory without removing
it from the local registry. Stop cooperatively cancels only the active generation,
keeping the model and stable conversation caches resident for the next prompt.
Output has no artificial token ceiling and stops on the model's end token, when
you press Stop, or at the unified-memory safety limit that keeps macOS responsive.
Long Markdown and reasoning streams are rendered in bounded chunks so completed
text is not reparsed for every new token. Fenced code blocks are highlighted as
they stream and include a one-click copy action for the complete block. A native
menu-bar panel keeps unified-memory usage, the loaded model, and the latest
generation performance visible even when the main window is closed.

The Settings page checks the latest public GitHub release at launch and on
demand. When a newer all-in-one DMG is available, it downloads in the background,
verifies GitHub's SHA-256 digest, then offers **Restart and install**. The engine
and native UI are versioned and updated together so their protocol stays in sync.

### MCP tools — Exa included, off by default

The **MCP switch in the message composer** enables/disables tools for the app.
It is **off on first launch**. Your choice is saved on this Mac and restored
after a relaunch, model change, or new conversation. Switching it does not reload
the model. During a generation or connection update the switch is temporarily
locked; turn it off before sending the next message to prevent tool use.

Exa web search and page fetching are preconfigured using its hosted endpoint
`https://mcp.exa.ai/mcp`. No Node installation or API key is required for Exa's
free tier (subject to Exa's rate limits). The signed DMG includes the native HTTP
client and certificate bundle. No MCP connection is opened while the master
switch is off. **When enabled, search queries and fetched URLs are sent to Exa**;
local model inference still runs on your Mac.

MLXL3 Desktop can also connect to local MCP servers over stdio, expose their tools to
the active model, execute tool calls, and show each call inline. Qwen tool
templates are used directly; models without a native tool template receive a
portable XML/JSON fallback prompt. Configure servers with the CLI:

```bash
mlxl3 mcp add filesystem npx -y @modelcontextprotocol/server-filesystem "$HOME/Documents"
mlxl3 mcp list
mlxl3 mcp check --json # explicitly connects to test tools, without loading a model
```

Or open **Generation settings → MCP → Configure** and edit the common
`mcpServers` JSON format stored in `~/.config/mlxl3/mcp.json`:

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

The per-server `enabled` flag controls which servers are eligible when the
master switch is on; it does not enable MCP globally. An explicit Exa entry
overrides the built-in preset. Set its `enabled` flag to `false` to exclude Exa
while using other servers. `mlxl3 mcp remove exa` does this too. Remote servers
support HTTPS Streamable HTTP (JSON/SSE responses); custom static `headers` are
supported. Browser OAuth and legacy HTTP+SSE endpoints are not supported.

Reload MCP from the same panel after editing (without reloading the model).
Commands are started directly,
without a shell, but enabled MCP servers and their tool descriptions are trusted
local code: only configure servers you trust and only grant the directories or
credentials they actually need.

Desktop enables grouped full-context attention for Gemma's FP16 d=512 decode
at 2048+ KV tokens. Other attention cases and architectures retain their usual
path. This avoids repeated KV reads across query heads; no speculative decoding
or extra quantization is used. Floating-point reduction order can differ.
Set `MLXL3_GEMMA_SDPA512=off` when launching Desktop to compare the reference
path; the standalone CLI remains opt-in. See the
[benchmark notes](docs/gemma-sdpa512-investigation.md).

Build the signed local application bundle and open it:

```bash
./scripts/build-macos-app.sh
open "dist/MLXL3 Desktop.app"
```

Release builds embed their own Python 3.12 interpreter, MLX/Metal libraries,
MLX-LM model implementations, tokenizers, Hugging Face downloader, and MLXL3
engine. Development builds still discover the editable project virtual
environment automatically. A different executable can be selected with
`MLXL3_EXECUTABLE=/path/to/mlxl3`.

### Install the standalone DMG

On an M1, M2, M3, M4, or M5 Mac running macOS 26.2 or newer:

1. Download
   `MLXL3-Desktop-v0.4.0-Apple-Silicon.dmg`, open it, and drag
   **MLXL3 Desktop** into Applications.
2. Launch the app. No Python, Homebrew, MLX, Hugging Face CLI, or Terminal setup
   is required. The current build is ad-hoc signed rather than Apple-notarized,
   so macOS may require right-clicking the app and choosing **Open** once.
3. Click **Ajouter un modèle EXL3**. Paste a public Hugging Face repository and,
   when needed, its BPW revision; alternatively import an existing EXL3 folder.
   The app downloads, registers, and loads the model automatically.

The DMG intentionally does not contain model weights: they are often several to
dozens of gigabytes and remain user-selected. M5-specific TensorOps are enabled
automatically where supported; earlier Apple Silicon GPUs use the compatible
Metal paths.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev,bundle]"
.venv/bin/pytest
```

Build the redistributable app and DMG with:

```bash
./scripts/build-macos-dmg.sh
```

The local model layout is intentionally ignored by Git:

```text
models/source/Ling-3.0-tiny-HF/
models/fixtures/LFM2.5-8B-A1B-EXL3-3.10bpw/
models/baselines/LFM2.5-8B-A1B-MLX-8bit/
```

See [`docs/kernel-port.md`](docs/kernel-port.md) for the CUDA-to-Metal inventory
and current implementation status.
