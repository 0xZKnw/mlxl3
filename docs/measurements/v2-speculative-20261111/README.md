# Qwen3.6 35B-A3B: MTP and DFlash2 snapshot — 11 November 2026

This is a small local measurement of existing engine behavior, not an engine
optimization or a guarantee of speed on other prompts or Macs.

## Setup

- Apple M5 / 24 GiB, macOS 27.2, AC power, MLX 0.32.2.
- Engine 2.0.5, commit `a3cffdcaa7ba`, executable SHA-256
  `7f5ea49e6768ffddf0ad10e39dfb2cae48410789ce92842d4d1a42e7660338a0`.
- Qwen3.6-35B-A3B EXL3 2.49 bpw, target revision
  `91a4efc46c1fece6426ece167f127d29d075f392`; both local model weight hashes
  matched the [previous identity record](../v2-models-20261111/model-identities.json).
- MTP1, stock 4-bit head SHA-256
  `77fbc6594cdd830cae89e0b693f18278dd0a7a7c5749fd33d4bc5817dbb91bad`.
  DFlash2 Auto, packed Q4 draft, local manifest SHA-256
  `22aa0f68a76fa8b84245b81eb1606f25aaa002259663f84f361fd7c66f244418`.
- Context 4096, temperature 0, top-k 0, repetition penalty 1, tools and
  prompt-cache reuse off. Before and after the campaign, process-list probes
  showed no other known inference process, compiler or prover; these probes did
  not continuously monitor the machine. Desktop remained open without an engine
  child at those probes, and other background applications were not controlled.
  No thermal or performance warning was recorded; GPU temperature, clocks and
  occupancy were not measured.
- Swap in use was 2.48 GiB before the ordinary run and about 6.97 GiB at the
  four speculative pass probes. File caches and system memory were not reset.
  This change limits comparisons across the ordinary and speculative runs.

## Results

The first prompt is the French short prompt built into `compare_native.py`;
its rendered input was 27 tokens. The second is the 62-token
[v2 snapshot prompt](../v2-models-20261111/prompt.txt). All runs produced 128
output tokens, of which 127 were decode transitions.

Ordinary decoding ran first with one warmup and three measured generations in
one process. MTP1 and DFlash2 then used four fresh processes in ABBA order,
one warmup and two measured generations per prompt and process. MTP and DFlash
had four measured responses each per prompt. Rates and timings are medians of
individual measured responses; the range shows observed decode variability.

| Prompt | Mode | Decode median (range) | TTFT median | Total median |
|---|---|---:|---:|---:|
| Snapshot 62 tokens | Ordinary | 50.60 (50.27–50.61) tok/s | 0.243 s | 2.752 s |
| Snapshot 62 tokens | MTP1 | 75.12 (74.45–75.75) tok/s | 0.243 s | 1.935 s |
| Snapshot 62 tokens | DFlash2 Auto | 70.56 (70.22–70.69) tok/s | 0.251 s | 2.051 s |
| French 27 tokens | MTP1 | 70.71 (70.60–71.04) tok/s | 0.128 s | 1.925 s |
| French 27 tokens | DFlash2 Auto | 61.11 (60.74–61.21) tok/s | 0.138 s | 2.216 s |

On these examples, all speculative passes produced identical token, text and
history hashes. The ordinary snapshot output also matched them. MTP accepted
62/65 proposals on the snapshot and 57/69 on French; DFlash accepted 98/123
and 91/136 respectively, constant across repetitions. These are bounded output
checks, not proof of parity on arbitrary inputs. No fresh baseline was run for
the French prompt. The unpaired ordinary run and large change in swap use mean
that these values must not be treated as a controlled v2 speedup.

Peak process lifetime footprint across measured runs was 12.86 GiB in ordinary
mode, and 13.01 GiB in both speculative modes; these are process lifetime peaks,
not total system memory or GPU-only allocations. Results remain sensitive to
prompt, output length, model, memory pressure, background work and the chosen
draft. The [ordinary record](ordinary.json) and [speculative record](speculative.json)
retain individual stats and conditions with local paths, process IDs and battery
identifiers redacted.

## Reproduce

After checking the machine is idle and no other inference or GPU benchmark is
running, use the repository scripts with the same model, head, draft and engine:

```sh
python benchmarks/v2_snapshot.py /path/to/mlxl3 /path/to/target \
  docs/measurements/v2-models-20261111/prompt.txt /tmp/ordinary.json
python benchmarks/compare_native.py /path/to/target \
  --baseline /path/to/mlxl3 --candidate /path/to/mlxl3 \
  --output /tmp/speculative --prompt-file docs/measurements/v2-models-20261111/prompt.txt \
  --tokens 128 --warmup-tokens 128 --repeats 2 --order ABBA --settle-seconds 3 \
  --mtp-depth 1 --mtp-head /path/to/mtp-head \
  --dflash-mode 1 --dflash-draft /path/to/dflash-draft --compare-mtp-dflash \
  --load-timeout 180 --request-timeout 180 --campaign-timeout 900
```
