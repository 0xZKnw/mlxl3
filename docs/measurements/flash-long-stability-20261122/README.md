# Qwen3.8 Flash Next long-prompt stability on Apple M5

On the 2.0.23 engine, grouped prefill with a serial suffix, a 5 GiB expert cache and early computation of resident experts completed the 654-token prompt and 128-token response in **80.3 s on average**. The starting configuration took 113.5 s on the same binary. This is a 29.2% reduction in response time. Sustained decode at 9 tok/s remains unverified.

| Configuration | Runs | Prefill, tok/s | Decode, tok/s | Decode range, tok/s | Mean first token, s | Mean response, s | Mean peak process footprint, GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Serial, 3 GiB | 2 | 6.94 | 6.59 | 6.30–6.91 | 94.2 | 113.5 | 7.06 |
| Grouped + serial suffix, 5 GiB, early resident experts | 2 | 10.16 | 7.95 | 7.83–8.07 | 64.4 | 80.3 | 9.13 |
| Same with cold waves disabled | 2 | 10.10 | 7.93 | 7.84–8.03 | 64.8 | 80.8 | 9.15 |

Each run started a fresh bridge on the same M5 with 24 GiB, MLX 0.32.2, the 2.05 bpw checkpoint, context 4096, temperature zero and prompt-cache reuse disabled. The comparison used the same input and verified all 128 generated tokens, text and chat history against the previous reference. Throughputs pool tokens divided by phase duration. Response time includes prefill and generation and excludes model loading. The final six measured runs were on battery in order A H J J H A after short warmups. The separate reversed series on 2.0.21 tested the added resident-expert option against the previous 5 GiB mode.

## What the variation tells us

The historical 9.2 and 4.6 tok/s CLI results each measured 31 decode intervals. Both occurred on battery, with identical output and no per-phase device counters. The serial control at the same cache size also fell from 7.1 to 4.4 tok/s over the campaign. A retrospective check of system thermal messages identified no CPU, GPU or SSD throttle event for these requests.

In the 2.0.21 screen the hybrid modes incurred roughly 113–114 MiB of process-attributed disk input per emitted token over events 1–32, versus 77–79 MiB over events 96–128. Equal outputs and similar total bytes still took different amounts of CPU and wall time. Early resident computation lets the GPU begin work before cold reads finish. On the four H runs across both 2.0.21 series, output throughput pooled to 7.84 tok/s versus 7.16 over four G runs with the same 5 GiB cache. Operating conditions varied, and the historical factor of two cannot be assigned to a single hardware cause.

Keep the GPU-hot option explicit. Disabling the second cold wave gave mixed paired results on 2.0.23. At the same 89 slots per layer, LRU also predicted more reads on both archived route traces and was excluded before speed testing.

## Run the measured configuration

```sh
MLXL3_SSD_CACHE_GIB=5 MLXL3_FLASH_PREFILL_EXPERT4=1 \
MLXL3_FLASH_PREFILL_SERIAL_TAIL=1 MLXL3_SSD_GPU_HOT=1 \
mlxl3-rs run "$MODEL" --prompt "$(cat docs/measurements/flash-long-decode-20261121/heldout.txt)" --max-tokens 128
```

The table uses the chat bridge. The command above enables the same policy for the CLI. Power, thermal states, scoped disk/GPU observations and run-level values are recorded in [results.json](results.json). The previous CLI and bridge measurements are in the [initial long-prompt report](../flash-long-decode-20261121/summary.json).

## Verification

The shipping binary produced bit-identical finite logits (12 × 248,320) for A, H and J. Two separate physical GPU tests compared full model state and continuation logits after a 257-token prefill. The MLX/chat library suite passed 558 tests with 329 ignored, the CPU suite 481 with 9 ignored, the benchmark/process tests 98, and the portable Kani manifest tests 8. Kani verified the CPU suffix selector over all usize endpoints and Boolean modes, including all three cover properties. Format, strict Clippy and the 341-file source inventory passed. A separate overlap component test stopped before execution because its external fixture was unavailable. CrossHair was unavailable for the local Python measurement helpers.
