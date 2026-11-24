# Qwen3.8 Flash Next with the optional 5 GiB expert cache

On battery with nominal thermal status, the reference and the lower-memory setting both exceeded **9 tok/s over each complete 128-token response**. The reference kept the faster pooled prefill. The lower-memory alternative saved 0.24 GiB and was about 3.1% slower at prefill in that series. Earlier runs on this Mac were substantially slower. Keep the 5 GiB cache optional.

## Controlled battery comparison

| Configuration | Prefill (tok/s) | Decode (tok/s) | Decode by run (tok/s) | Mean / maximum process peak (GiB) |
|---|---:|---:|---|---:|
| H · 512 MiB of allocator buffers, four SSD readers | 12.95 | 9.78 | 9.99, 9.57 | 9.136 / 9.136 |
| R · 256 MiB of allocator buffers, two SSD readers | 12.55 | 9.75 | 9.85, 9.65 | 8.894 / 8.904 |

From 14:55 to 15:00 CET the Mac reported battery power at every process boundary, with no sleep entry in the system log. The largest difference between elapsed wall-clock and monotonic time per process was under 1 ms. One fresh process per trial, two short warmups, then H, R, R, H. All four long responses matched the historical output. H finished in 63.50 s on average, R in 65.16 s. The paired comparisons split: H was faster at both phases in the first pair, R in the second. Some early windows fell below 9 tok/s (8.65 for R, 8.96 for H).

The thermal state reported by macOS throughout the measured decode windows was nominal (0) in this series and fair (1) during the earlier four-reader screen and the first two-reader series. Those conditions were observed, not controlled. The historical mean process peak for H was 9.13285 GiB. H's new mean was higher by 2.7 MiB, while both individual R peaks stayed below that historical mean. R qualifies as an optional lower-memory setting. The measurements do not establish a repeatable decode speedup over H or a 9 tok/s floor under all conditions.

## Measurement

Apple M5 with 24 GiB, Qwen3.8 Flash Next EXL3 2.05 bpw, MLX 0.32.2, engine 2.0.23. Each timed run loaded a fresh chat bridge with context 4096, processed the [same 654-token input](../flash-long-decode-20261121/heldout.txt), then generated 128 tokens greedily. The bridge timed 127 decode intervals. The rates pool tokens divided by the duration of the corresponding phase. The memory metric is the lifetime peak footprint of the entire process. The expert cache budget covers only the routed expert weights.

All configurations use 89 expert slots per layer, grouped prefill with a serial suffix and early GPU processing of resident experts. H leaves the allocator at the engine's 512 MiB default. X limits unused allocator buffers to 256 MiB. Y adds the GDN epilogue fusion to X. The historical environment variable `MLXL3_FLASH_SWIGLU_DOWN` was also passed in Y. This engine has no reader for that variable.

The screen used three short warmups (8 generated tokens) followed by H, X, Y, Y, X, H. Each full response was checked against the frozen token, text and chat-history hashes. Before timing, X and Y each reproduced the exact independent reference for all 12 × 248,320 finite logits. A separate physical test per configuration compared a grouped 257-token prefill with serial evaluation, including state and continuation logits (one passed per configuration, 886 filtered per invocation). No model or compiler ran alongside the timed bridge. All power readings, per-run peaks and decode windows are in [results.json](results.json).

To reproduce H, set these variables before starting the bridge:

```sh
export MLXL3_SSD_CACHE_GIB=5 MLXL3_SSD_GPU_HOT=1
export MLXL3_FLASH_PREFILL_EXPERT4=1 MLXL3_FLASH_PREFILL_SERIAL_TAIL=1
mlxl3-rs bridge "$MODEL" --context-length 4096
```

Use a fresh process for each configuration. For X, additionally export `MLXL3_ALLOCATOR_CACHE_MIB=256`. For Y, also export `MLXL3_FLASH_GDN_EPILOGUE=1`. For the lower-memory R configuration, start from H and additionally export both `MLXL3_ALLOCATOR_CACHE_MIB=256` and `MLXL3_SSD_READERS=2`. All runs sent one user message through the bridge with the exact document text, temperature zero, top_k zero, repetition penalty 1, prompt reuse off and speculative decoding off. The expert budget is opt-in. The [previous same-binary measurements](../flash-long-stability-20261122/README.md) provide the starting serial configuration and the earlier 5 GiB reference.

## Results with four SSD readers

| Configuration | Prefill (tok/s) | Decode (tok/s) | Decode by run (tok/s) | Mean / maximum process peak (GiB) |
|---|---:|---:|---|---:|
| H · reference | 8.83 | 6.77 | 6.63, 6.91 | 9.135 / 9.135 |
| X · 256 MiB | 8.58 | 6.65 | 6.62, 6.67 | 8.891 / 8.901 |
| Y · 256 MiB, fused GDN | 8.43 | 6.71 | 6.83, 6.58 | 8.884 / 8.885 |

The lower buffer limit saved about 0.24 GiB of peak process memory. Both configurations lost prefill throughput against H. The computer switched from battery to AC power during the first timed H run and stayed on AC for the remainder of this series. The earlier H result on the same binary was 10.16 tok/s prefill and 7.95 tok/s decode. The contemporary H control shows how much the rates varied between sessions.

## Expert retention and cache allocation

Two offline experiments used the archived OFF-MAIN and OFF-HOLD expert route traces. Both kept 4,272 slots across 48 layers. An independent simulator reproduced the archived baseline for 53 slots and the more recent 89-slot control. Selection used OFF-MAIN only. Advancement required at least 3% fewer decode misses on both traces.

| Policy | OFF-MAIN misses | OFF-HOLD misses |
|---|---:|---:|
| Uniform, frequency counts halved every 16 batches | 12,119 | 13,422 |
| Uniform, counts halved every 32 batches | 12,104 | 13,508 |
| Six groups of eight layers, 121/89/105/73/89/57 slots per layer | 11,851 | 13,400 |

The block allocation selected from 1,751 budget-preserving profiles reduced held-out misses by 0.16%. Neither profile qualified for a device run. The replay uses archived routes from different prompts and supplies no tokens-per-second measurement.

Across the six long runs with four readers, the process accounted for 114–117 MiB of disk input per emitted text event over events 1–32 and 77–82 MiB over events 96–128, rounded to whole MiB. These counters measure bytes attributed to the process. They do not identify the sole cause of the rate variation or measure physical NAND traffic.

## Two SSD readers

R combines the 256 MiB buffer limit in X with `MLXL3_SSD_READERS=2`. A second series compared this combination against H with four readers after a short warmup per mode. The measured order was R, H, H, R. This comparison cannot isolate the effect of reader count by itself because H also uses the larger buffer limit. For the intended configuration decision, H is the reference the alternative has to beat.

| Configuration | Prefill (tok/s) | Decode (tok/s) | Decode by run (tok/s) | Mean / maximum process peak (GiB) |
|---|---:|---:|---|---:|
| H · reference | 8.23 | 6.07 | 6.74, 5.53 | 9.135 / 9.137 |
| R · 256 MiB, two readers | 8.93 | 6.85 | 6.74, 6.95 | 8.891 / 8.901 |

All four measured runs were on AC power. The consecutive H controls differed by 1.21 tok/s, while the first R and H ran at almost identical speeds. All four responses matched the reference. R passed the same 12-vector logits check and the 257-token state and continuation test as X and Y. The initial journal write for R failed during numerical qualification. The full protocol was recorded before the warmups and speed measurements.

The reverse-order series returned the exact expected outputs, but the Mac slept during the series. Its timings are excluded from speed comparisons. A guarded AC retry found the Mac on battery during warmup and stopped before any long generation.
