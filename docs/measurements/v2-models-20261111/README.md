# MLXL3 v2 model snapshot — 11 November 2026

These are six separate, local measurements of the MLX-only engine. They are a
small snapshot, not a claim of a speedup or a ranking independent of workload.

## Machine and runtime

- Apple M5 (`Mac17,2`), 24 GiB unified memory, macOS 27.2, AC power.
- Engine 2.0.5, MLX 0.32.2; source commit `a3cffdcaa7bafb513eb127c15319dc3c0a1f5046`.
- Measured executable SHA-256: `7f5ea49e6768ffddf0ad10e39dfb2cae48410789ce92842d4d1a42e7660338a0`.
- One model process at a time. The Desktop was closed and no compiler or other
  model process was running. Ordinary background applications remained open.
  The OS page cache was not cleared or controlled; the Flash Next checkpoint
  uses SSD-backed weights. The system reported 4 GiB swap allocated, with
  about 2.8 GiB in use at the recorded probes. `pmset` reported no recorded
  thermal or performance warning; this is not a GPU occupancy measurement.
- All local Safetensors files, including the Flash sidecars, matched the SHA-256
  values in their listed immutable Hugging Face revisions. The individual file
  hashes, sizes, and revision checks are in [model-identities.json](model-identities.json)
  and [remote-identity.json](remote-identity.json). Tokenizer and configuration
  file hashes identify the local files; only Safetensors were checked against
  the remote LFS metadata.

## Workload and interpretation

Each model received the same [prompt](prompt.txt) (SHA-256
`7447c24f4d66adca67c19034f192b182d5f8a4d680489adcaec41e5de7a16e13`).
The bridge used context 4,096, temperature 0, top-k 0, repetition penalty 1,
128 output tokens, no MTP, DFlash, MCP tools or prompt-cache reuse. One 128-token
warmup preceded three 128-token measured generations in the same process.
The runner checks finite positive metrics, exact output length and stable greedy
output hashes. Each model was run separately, in the order in this table:

| Model | Prompt tokens | Prefill tok/s | Decode tok/s, median (range) | TTFT s | Total s | Peak process GiB |
|---|---:|---:|---:|---:|---:|---:|
| [LFM3 original 8 bpw](lfm3-original.json) | 61 | 886.35 | 49.47 (46.94–51.20) | 0.069 | 2.64 | 2.58 |
| [LFM3 Heretic NX 8 bpw](lfm3-heretic.json) | 61 | 868.87 | 46.80 (46.31–48.65) | 0.071 | 2.78 | 2.58 |
| [LFM3-VL 2B 8 bpw](lfm3-vl.json) | 61 | 885.66 | 46.62 (46.07–48.32) | 0.069 | 2.79 | 2.76 |
| [Qwen3.8 27B 2 bpw](qwen-27b.json) | 104 | 73.74 | 8.33 (8.13–9.13) | 1.411 | 16.67 | 11.19 |
| [Qwen3.6 35B-A3B 2.49 bpw](qwen-35b-a3b.json) | 62 | 223.11 | 42.75 (41.08–44.15) | 0.278 | 3.25 | 13.02 |
| [Qwen3.8 Flash Next 2.05 bpw](flash-next.json) | 104 | 9.97 | 10.05 (9.87–10.24) | 10.433 | 23.17 | 9.67 |

Prefill, TTFT and total are medians. Decode is counted over 127 tokens after the
first token. TTFT and total start with the generation request and exclude model
loading. Peak is the maximum reported process lifetime footprint among the
three runs; it is neither the full system footprint nor a model-file size.
The VL row has no image: it measures the text path. Different tokenizers produce
61–104 prompt tokens, and the models have different architectures, so the
numbers are not a controlled comparison of equivalent model work.

## Quality checks and limits

Before timing, the LFM3 original, Heretic and VL checkpoints were compared
against the earlier 2.0.4 runtime at 3 and 499 forced tokens. In these six cases,
all finite logits were identical (KL 0); the 35 short-prefix state arrays per
model were byte-identical. The Qwen checkpoints were compared on one forced
3-token prefix each; their finite logits also matched (KL 0). See
[quality.json](quality.json) and [quality-qwen.json](quality-qwen.json). These
are bounded examples, not a proof of model equivalence on arbitrary prompts.

The production Desktop → bridge → MLX engine → LFM3-VL path correctly reported
red and blue for two simple image fixtures. This checks image delivery and color
discrimination only; it is not a broad vision, OCR or video evaluation.

## Reproduce

Use the same engine build, model files and hardware, with MLX 0.32.2. After
confirming no other model or GPU benchmark is running, run once per checkpoint:

```sh
python benchmarks/v2_snapshot.py /path/to/mlxl3 /path/to/checkpoint \
  docs/measurements/v2-models-20261111/prompt.txt /tmp/lfm3-result.json
```

The result is written after each completed generation, including failures. The
runner uses the repository's bounded `compare_native.generate` and JSON process
helpers. The checked-in runner preserves the measured generation and validation
logic; it omits local executable paths from routine runtime metadata and drops
process identifiers and battery IDs. Error messages and stderr can still contain
local paths. The files here are the original measured values with identifiers
removed. Background workload, power, thermal state and SSD cache can
change results; record them and compare repetitions rather than one run.
