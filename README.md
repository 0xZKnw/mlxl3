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
- a mapped two-launch `SwitchGLU` path for selected MoE experts;
- a vectorized 2/4/8-row Metal QMM that keeps weights serialized and amortizes decode;
- a standard EXL3 safetensors loader on top of MLX-LM architectures;
- CPU-vs-Metal conformance tests for every bit width from 1 through 8.

The current end-to-end target is the official LFM2.5-8B-A1B EXL3 checkpoint at
3.10 bpw. MLXL3 loads all 2,179 quantized modules, including 2,112 routed expert
projections, and runs the model entirely through MLX/Metal. Ling conversion is
paused because a full no-training EXL3 conversion is much slower to iterate on.

On this 10-core M5, the LFM fixture reaches a three-run warm median of 55.5
generated tokens/s and a 4.02 GB peak allocation. The matching MLX 8-bit model
reaches 58.9 tokens/s and 9.04 GB. Prefill is currently 77.1 versus 122.5
tokens/s; a tensor-tiled EXL3 GEMM remains future work.

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
interactive process; no conversation is persisted between separate runs.

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
