# Native Rust port

Experimental work on `codex/rust-rewrite`. This is an incomplete migration,
not a replacement for the installed MLXL3 CLI or Desktop app. No model-level
speedup is established by changing languages.

## Current scope

- Native CLI: model registry, checkpoint inspection, codec oracle and streaming
  reasoning separation. Registry writes use the same file format and lock as
  the Python app; removing a registration retains the model files.
- CPU EXL3 codec: packing, tail-biting unpacking, permutations and all three
  procedural codebooks for K=1 through K=8.
- macOS: direct Metal ownership and reference EXL3 GPU kernels, without a
  Python interpreter. These are correctness references, not yet a complete
  optimized inference engine.
- Optional `mlx,chat` integration: native LFM2 dense/MoE and Qwen3.5 dense/MoE
  inference, grouped projections, recurrent/KV states, local Hugging Face
  tokenizer and Jinja chat templates, greedy streaming chat with
  per-conversation history and thinking separation.
- Resident JSON-lines bridge compatible with the SwiftUI event transport:
  loading/readiness, context usage, streaming phases, sampling controls,
  completion statistics, ping/shutdown and cooperative `SIGUSR1` cancellation.
- Native Hugging Face catalogue, resumable downloads and authentication, plus
  native stdio/HTTP MCP discovery, tool execution and multi-round bridge events.

The SwiftUI GUI now packages this Rust executable, `libmlx`, `libjaccl` and the
MLX metallib as a self-contained runtime. The full quantization pipeline has
not been migrated. Gemma and Ling architectures are not supported by the Rust
inference path yet. It rejects unsupported architectures explicitly and never
silently invokes Python. Existing Python/SwiftUI production remains available.

## Build and try the standalone CLI

Run these commands from the repository root. Install stable Rust with `cargo`,
`rustfmt` and `clippy`; the standalone build supports Linux and macOS and
does not require MLX or Python.

```sh
cargo build --release --locked
./target/release/mlxl3-rs --help
./target/release/mlxl3-rs list --json
./target/release/mlxl3-rs inspect /path/to/exl3-model
```

By default the registry is `~/.config/mlxl3/models.json`, or
`$MLXL3_HOME/models.json` when that variable is nonempty. Use a separate
registry to experiment without changing your app's library:

```sh
./target/release/mlxl3-rs --registry target/native-demo/models.json register demo /path/to/exl3-model
./target/release/mlxl3-rs --registry target/native-demo/models.json list
./target/release/mlxl3-rs --registry target/native-demo/models.json remove demo
```

## Tests

```sh
cargo fmt --check
cargo clippy --locked --all-targets --features chat -- -D warnings
cargo test --locked --features chat
cargo build --locked --features chat
python3 -m pip install numpy pytest
python3 native/check_parity.py --binary target/debug/mlxl3-rs
python3 -m pytest -q native/test_streaming_parity.py native/test_kernel_sources.py
```

The GitHub workflow `rust.yml` runs these standalone checks on Linux and
macOS. It compares CPU codec bit patterns and exact streaming fragments with
the Python source, and exercises checkpoint validation, isolated CLI commands
and concurrent registry updates. It does not download model weights.

Metal tests are explicitly ignored by default because hosted CI does not
validate physical GPU behavior. On a physical Apple Silicon Mac:

```sh
cargo test --locked gpu::tests -- --ignored
python3 native/check_parity.py --binary target/debug/mlxl3-rs --metal
```

The second command requires the repository's Python/MLX environment. These
are numerical correctness checks, not performance benchmarks or full-model
quality evaluations.

## Optional native MLX runtime

The experimental `mlx` feature links the C++ MLX runtime rather than embedding
Python. It requires a compatible local MLX distribution; this feature is not
enabled in the hosted standalone CI jobs.

Set `MLXL3_MLX_ROOT` to your MLX package directory. For an existing MLX wheel
installed in the repository's `.venv`, locate it without importing the GPU
runtime:

```sh
export MLXL3_MLX_ROOT="$(.venv/bin/python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("mlx").submodule_search_locations)))')"
export MACOSX_DEPLOYMENT_TARGET=26.2
cargo build --release --locked --features mlx,chat
./target/release/mlxl3-rs run /path/to/LFM2-EXL3 --max-tokens 256
./target/release/mlxl3-rs run /path/to/LFM2-EXL3 --prompt 'Bonjour !' --max-tokens 256
./target/release/mlxl3-rs run /path/to/LFM2-MoE-EXL3 --prompt 'Bonjour !' --max-tokens 256
./target/release/mlxl3-rs run /path/to/Qwen3.5-Dense-EXL3 --prompt 'Bonjour !' --max-tokens 256
./target/release/mlxl3-rs run /path/to/Qwen3.5-MoE-EXL3 --prompt 'Bonjour !' --max-tokens 256
```

Use `/clear` to reset the conversation, `/exit` to quit. Model registry names
work instead of paths. `--max-tokens 0` generates until EOS or the context limit;
the experimental default is 128. Desktop bridge sampling and MCP are native;
the interactive CLI remains greedy.

The MLX headers and dynamic library must both be **0.32.2**. Development builds
can link to that local path; `scripts/build-macos-app.sh` copies the required
libraries and metallib beside the Rust executable for a self-contained bundle.
The tested wheel requires macOS 26.2 or later and Apple Silicon. A Metal-capable
SDK is also needed to build the shim and compile shaders at runtime.

Prefill currently replays the rendered conversation **one token at a time**.
Prefix reuse, batched QMM and capacity-managed KV storage are not ported.
Displayed timings are diagnostic, not a claimed improvement over
production. No quantization pipeline or full app rewrite is complete yet.

## Numerical validation

On the development M5, 152 differential checks pass bit-for-bit: CPU codecs,
72 complete projections across K1–8 and all three codebooks, plus 42 ragged
projection groups across K1–6/8. The direct-Metal path separately passed 124
checks. These counts overlap in their CPU cases and should not be added.
LFM2.5-1.2B-Thinking and LFM2.5-2.6B EXL3 4bpw pass **every logit and every
recurrent/KV value** on eight imposed tokens each against the production Python engine, including QKV
grouping. Chat template strings, token IDs and decoding match Transformers
on a multilingual four-message conversation. LFM2.5-8B-A1B EXL3 3.10bpw also
passes every logit and cache on the same eight-token sequence. Qwen3.5-MoE
passes all 40 layers, 248,320 output logits and every recurrent/KV state
bit-for-bit on a three-token sequence; both MoE native chat/reset paths pass
local smoke tests. Qwen3.8-27B dense likewise passes 248,320 logits bit-for-bit
on three stateful tokens plus chat/reset smokes. These are bounded correctness
checks, not a complete model-quality evaluation or throughput benchmark.

```sh
PYTHONPATH=src .venv/bin/python native/check_parity.py --binary target/release/mlxl3-rs --mlx
PYTHONPATH=src .venv/bin/python native/check_model_parity.py /path/to/LFM2-EXL3 --binary target/release/mlxl3-rs
MLXL3_TOKENIZER_MODEL=/path/to/LFM2-EXL3 MLXL3_TEST_PYTHON="$PWD/.venv/bin/python" \
  cargo test --features chat tokenizer::tests::local_transformers_prompt_and_ids_match -- --ignored
```

See [opti.md](../opti.md) for rejected experiments, validation scope and actual
integration status. No installed CLI, Desktop app or release is replaced by
building this branch.
