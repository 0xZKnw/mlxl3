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

Useful generation overrides are `--max-tokens`, `--temperature`, `--top-k`,
`--repetition-penalty`, and `--system`.

All prompt sizes stay on the serialized Metal QMM path; MLXL3 never reconstructs
a dense weight during ordinary inference. Cached chat prefixes use a memory-aware
512/1024/2048-token step, selecting wider and faster blocks when unified-memory
headroom allows it. The choice can be pinned for experiments with
`MLXL3_PREFILL_STEP_SIZE`.

## Native macOS app

MLXL3 Desktop is a native SwiftUI application for macOS 26. It uses the system
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
text is not reparsed for every new token. A native menu-bar panel keeps
unified-memory usage, the loaded model, and the latest generation performance
visible even when the main window is closed.

### Local MCP tools

MLXL3 Desktop can connect to local MCP servers over stdio, expose their tools to
the active model, execute tool calls, and show each call inline. Qwen tool
templates are used directly; models without a native tool template receive a
portable XML/JSON fallback prompt. Configure servers with the CLI:

```bash
mlxl3 mcp add filesystem npx -y @modelcontextprotocol/server-filesystem "$HOME/Documents"
mlxl3 mcp list
```

Or open **Generation settings → MCP → Configure** and edit the common
`mcpServers` JSON format stored in `~/.config/mlxl3/mcp.json`:

```json
{
  "version": 1,
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/Documents"],
      "enabled": true
    }
  }
}
```

Reload MCP from the same panel after editing. Commands are started directly,
without a shell, but enabled MCP servers and their tool descriptions are trusted
local code: only configure servers you trust and only grant the directories or
credentials they actually need.

Build the signed local application bundle and open it:

```bash
./scripts/build-macos-app.sh
open "dist/MLXL3 Desktop.app"
```

The app discovers the editable project virtual environment automatically. A
different executable can be selected with `MLXL3_EXECUTABLE=/path/to/mlxl3`.

### Install the current DMG release

The current `v0.1.0` DMG contains the native desktop interface, but it does not
yet bundle the Python/MLX inference runtime or a Hugging Face downloader. On an
Apple Silicon Mac running macOS 26:

1. Download
   [`MLXL3-Desktop-v0.1.0-Apple-Silicon.dmg`](https://github.com/0xZKnw/mlxl3/releases/download/v0.1.0/MLXL3-Desktop-v0.1.0-Apple-Silicon.dmg),
   open it, and drag **MLXL3 Desktop** into Applications. The app is currently
   ad-hoc signed rather than Apple-notarized, so the first launch may require
   right-clicking the app and choosing **Open**.
2. Install the local engine and Hugging Face CLI:

   ```bash
   brew install python@3.12 pipx hf
   pipx ensurepath
   pipx install --python "$(brew --prefix python@3.12)/bin/python3.12" \
     "git+https://github.com/0xZKnw/mlxl3.git"
   ```

3. Download a standard EXL3 model. This is the LFM checkpoint used by the
   project and known to work end to end:

   ```bash
   mkdir -p "$HOME/Models"
   hf download turboderp/LFM2.5-8B-A1B-exl3 \
     --revision 3.10bpw \
     --local-dir "$HOME/Models/LFM2.5-8B-A1B-EXL3-3.10bpw"
   ```

4. Add the downloaded directory to MLXL3's local registry:

   ```bash
   mlxl3 register lfm2.5-8b-a1b \
     "$HOME/Models/LFM2.5-8B-A1B-EXL3-3.10bpw"
   mlxl3 list
   ```

5. Launch **MLXL3 Desktop**. It discovers the registered model automatically.
   If the app was already open while registering it, quit and reopen the app.

The same `hf download` then `mlxl3 register` flow applies to other standard
EXL3 checkpoints whose model architecture is supported by the pinned MLX-LM
runtime. Bundling the runtime and model manager directly in the app is planned
for a future standalone release.

## Development

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The local model layout is intentionally ignored by Git:

```text
models/source/Ling-3.0-tiny-HF/
models/fixtures/LFM2.5-8B-A1B-EXL3-3.10bpw/
models/baselines/LFM2.5-8B-A1B-MLX-8bit/
```

See [`docs/kernel-port.md`](docs/kernel-port.md) for the CUDA-to-Metal inventory
and current implementation status.
