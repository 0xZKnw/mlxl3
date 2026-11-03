# Native MLXL3 runtime

MLXL3 Desktop and the CLI use the native Rust/C++/MLX/Metal runtime. The
application does not start or embed Python for inference. This is the current
runtime, not an unmerged rewrite. Building it creates local artifacts; it does
not install an app, change model weights, or publish a release.

This document describes the source tree reviewed on 16 October 2026. A source
capability, a tested local package, a published engine and the installed app are
separate states. Use `mlxl3-rs runtime-info`, Desktop's loaded-engine identity and
the relevant release report to identify the executable actually used.
The [previous README is archived verbatim](../docs/native-runtime-readme-before-audit-2026-10-16.md).
The [audit status matrix](../docs/audit-api-lfm-status.md) records remaining work
and its acceptance conditions; planned features there are not current support.

## Components and capabilities

| Component | Current behavior and limits |
|---|---|
| Rust CLI | Registry, EXL3 checkpoint inspection, downloads, authentication, codec oracle, native text chat, resident bridge and HTTP service. Removing a registration retains the model files. |
| CPU codec | Trellis packing/unpacking, permutations and procedural codebooks for K1–K8; CPU operation does not require MLX. |
| EXL3 execution | Decode QMV, eligible small batches, grouped projections and TensorOps QMM selected by supported shapes/device. CUDA optimizations and unrelated Metal quantized formats are not interchangeable with this representation. |
| LFM2 / LFM2 MoE | Text inference with attention and short-convolution states, including the LFM2.5-24B-A2B 64-expert router. Dense LFM can use eligible multi-token prefill; MoE remains serial by default. An opt-in V2 K4/MCG microblock path has passed the registered numerical matrix; the two final timing orders did not meet stability criteria, so no speedup is claimed. LFM has no native MTP or DFlash2 support. |
| Qwen3.5/3.6/3.8 | Supported dense/MoE text layouts, attention/recurrent states, prefill paths, prefix checkpoints and optional matching MTP heads. Exact eligibility depends on the target/head configuration. |
| Gemma 4 / Ling 3 | Native text paths and architecture-specific tool-call handling. This does not imply image/audio support or universal checkpoint qualification. See the model validation reports. |
| Desktop bridge | JSON-lines readiness, capabilities, context usage, streaming phases, tools, completion statistics, configuration and cooperative cancellation. Model/runtime capabilities control acceleration eligibility. |
| MCP | Native stdio/HTTP discovery and tool execution through configured servers. This is distinct from HTTP Chat Completions tool calling. |
| Memory | Bounded prompt checkpoints (256 MiB default), conservative retained-component accounting and conditional exact compaction. The optional Memory saving policy releases saved checkpoints/unused buffers, not weights or history. |

LFM and Qwen expose logical context-memory profiles. These estimate persistent
states; they exclude retained backing, allocator caches, temporary work and other
processes. MLX peak allocation, process footprint and checkpoint file size are
different quantities. The initial `resident_gb` checkpoint field is a file-size
estimate; it is not a complete RAM measurement.

Desktop distinguishes saved MTP/DFlash preference from eligibility for the
selected model. An incompatible LFM keeps ordinary sampling editable and OFF
accessible. For an eligible greedy acceleration mode, the request uses 0/1/1
while preserving the user's ordinary temperature/Top K/repetition settings.
This correction is locally validated; its publication/install state is recorded
in the [LFM UI report](../docs/measurements/lfm-mtp-ui-20261016/README.md).

## Build and run

Run from the repository root. The Rust package is defined by the root
`Cargo.toml`, with sources in `native/`; there is no separate native manifest.
The CPU CLI builds on Linux/macOS without Python or MLX:

```sh
cargo build --release --locked
./target/release/mlxl3-rs runtime-info
./target/release/mlxl3-rs list --json
./target/release/mlxl3-rs inspect /path/to/exl3-model
```

The default registry is `~/.config/mlxl3/models.json`, or beneath `MLXL3_HOME`
when configured. Use an isolated registry for experiments:

```sh
./target/release/mlxl3-rs --registry /tmp/mlxl3-demo-models.json register demo /path/to/exl3-model
./target/release/mlxl3-rs --registry /tmp/mlxl3-demo-models.json list
./target/release/mlxl3-rs --registry /tmp/mlxl3-demo-models.json remove demo
```

Native inference requires Apple Silicon, macOS 26.2+, compatible Metal SDK tools,
and MLX headers/libraries **0.32.2**. Set `MLXL3_MLX_ROOT` to that MLX package:

```sh
MLXL3_MLX_ROOT=/path/to/site-packages/mlx MACOSX_DEPLOYMENT_TARGET=26.2 \
  cargo build --release --locked --features mlx,chat
./target/release/mlxl3-rs run /path/to/exl3-model --prompt 'Bonjour !' --max-tokens 256
./target/release/mlxl3-rs bridge /path/to/exl3-model --context-length 4096
```

Interactive `run` supports `/clear` and `/exit` and remains greedy. Its default
output budget is 128; `--max-tokens 0` continues until EOS or the context limit.
The Desktop bridge provides the broader sampling/MCP controls. Chat templates
and tokenization use local model files; unsupported layouts fail explicitly.
The offline PonyExl3 conversion workflow remains separate and Python-based.

`scripts/build-macos-app.sh` builds Rust with `mlx,chat`, builds Swift, bundles
`mlxl3`, `libmlx.dylib`, `libjaccl.dylib` and `mlx.metallib`, and signs local
artifacts ad-hoc. `scripts/build-macos-dmg.sh` packages that app. Neither command
installs it into `/Applications`. Developer ID signing/notarization is not
provided by these scripts.

## Acceleration and state reuse

Managed MTP heads are pinned to supported Qwen targets; target weights are not
downloaded by the head command. For example:

```sh
./target/release/mlxl3-rs mtp-head --target /path/to/Qwen3.8-27B-EXL3
./target/release/mlxl3-rs mtp-head --target /path/to/Qwen3.6-35B-A3B-EXL3
```

The bridge loads/unloads heads explicitly and Tune compares eligible modes
against ordinary decoding. DFlash2 has a separate matching-draft loader and Tune
path for the supported 27B/35B configurations. MTP and DFlash2 are mutually
exclusive; proposals are target-verified. These paths are greedy, not a general
implementation of stochastic speculative sampling.

Prefix reuse, prefill checkpoints, QMM and capacity-managed Qwen KV storage
already exist. They are architecture-specific; hybrid states cannot be restored
from KV alone. LFM baseline attention concatenates KV and short-convolution
state may retain a larger backing allocation.

`MLXL3_LFM_MOE_PREFILL=1` opts into microblocks of at most eight tokens only for
qualified LFM2.5-24B-A2B V2 geometry and mapped K4/MCG expert projections.
The default, `0`, and invalid explicit values keep the canonical serial path.
Attention, convolution, norms, residuals, gates and the final head keep canonical
M1 shapes; only independent MoE FFN rows are grouped. This path remains experimental
because the complete performance campaign was interrupted by host sleep and a
concurrent graphics workload. No qualified speedup is claimed for this release.
The [LFM report](../docs/measurements/audit-api-lfm-20261016/lfm-final.md) records
the numerical matrix, T600 checks and real bridge/API checks with this path active.
These are bounded validations, not an all-position proof or BF16 quality measurement.
The capacity-cache prototype was removed after a screen showed no useful benefit;
LFM keeps concatenated KV.

Adaptive MTP and several fused/alternative paths are experimental and off by
default. In particular, `MLXL3_MTP_FUSED_NORM=1` and
`MLXL3_QWEN_FUSED_NORM=1` opt into eligible add/RMSNorm variants; the historical
M5 results do not establish a general gain. Do not infer default behavior or
throughput from a branch-specific experiment. Flash Next SSD work and its
long-context qualification remain separate from this stable-source baseline.

## Local HTTP API

Build with `mlx,chat`, then serve a registered model or supported model path:

```sh
./target/release/mlxl3-rs serve my-model --host 127.0.0.1 --port 8000 --api-key local-secret
```

The API provides text `/v1/chat/completions`, including streaming, `/v1/models`
and `/v1/capabilities`. Capabilities separate HTTP-adapter support from public
bridge metadata. Stop sequences, tool calls, constrained JSON Schema, images,
MTP/DFlash through HTTP and unsupported sampling options are explicitly rejected;
MCP support in the app does not add these HTTP features.

`/health` returns HTTP200 while the server is alive and reports `ready`, `busy`
and `bridge_healthy`. `/ready` returns503 when the bridge is busy, dead or
invalidated. A single generation is admitted before reading its body; another
receives429 with `Retry-After: 1`. EOS maps to `finish_reason: stop`; output and
context limits map to `length`. Malformed terminal reasons or token counts are
errors rather than successful completions.

A non-loopback bind requires an API key. CLI clients without Origin work; an
Origin must be HTTP(S) loopback and is reflected only after validation. Without
a key, a present Host must also name loopback; with a key, authenticated network
Host values work. IPv6 listen addresses accept `::1` and `[::1]`.

Body/event/output sizes and waits are bounded. Client closure is observed before
the first delta, cancellation is drained before bridge reuse, and an uncertain
bridge is invalidated. A separate socket write-phase deadline bounds stalled
Hyper writes; it is not a total connection-lifetime or connection-count limit.
Signals are registered before model startup, and all startup/server exits reap
the child and join stderr or report an unconfirmed cleanup.

The [API report](../docs/measurements/audit-api-lfm-20261016/api.md) records
33 passing CPU transport fixtures, three detected/restored protection mutations
and the bounded source checks. They do not establish all concurrent network
interleavings or real-model integration; the exact shipped build and GitHub CI
remain separate release gates.

## Verification and numerical scope

The ordinary CI workflows run on pushes and pull requests. `rust.yml` runs CPU
contracts, codec/transport regressions, Kani and macOS compilation of the shipped
`mlx,chat` variant; `check.yml` runs Desktop/protocol checks. Hosted compilation
is not a numerical GPU test. Relevant local commands include:

```sh
cargo fmt --check
cargo clippy --locked --all-targets --features chat -- -D warnings
cargo test --locked --features chat
cargo build --locked --features chat
python3 -m pytest -q native/test_streaming_parity.py native/test_kernel_sources.py
```

With the pinned MLX environment, also compile/lint `--features mlx,chat`.
Physical GPU checks must be selected explicitly and run separately from builds,
proof tools and other model loads. Examples of existing tools are:

```sh
PYTHONPATH=src .venv/bin/python native/check_parity.py --binary target/release/mlxl3-rs --mlx
PYTHONPATH=src .venv/bin/python native/check_model_parity.py /path/to/LFM2-EXL3 --binary target/release/mlxl3-rs
```

Those commands can load models/use the GPU; they are examples, not checks run by
this documentation update. Existing reports contain bounded differential
validation for LFM, Qwen and Gemma and identify incomplete Ling qualification.
Do not turn a small imposed-token parity test into a universal quality claim,
add overlapping test counts, or treat Kani CPU properties as proof of Metal,
MLX, allocation or concurrency. Quantized-vs-BF16 quality and native-vs-reference
runtime parity are separate comparisons.

Read [opti.md](../opti.md) before optimization work. It preserves rejected
experiments, exact protocols, source identities and actual integration status.
For release validation use the matching `docs/release-*.md` and linked manifests,
not the historical counts in the archived README. This documentation update does
not publish a release or replace the installed application.
