# Qwen3.8 27B: MTP and DFlash2 snapshot — 11 November 2026

These are small local measurements of existing behavior, not a universal
ranking, optimization claim or guarantee for other prompts and hardware.

## Setup

- Apple M5 / 24 GiB, macOS 27.2, AC power, MLX 0.32.2. Desktop remained open
  without a loaded engine at preflight. Other background apps were not controlled.
- Engine 2.0.5, commit `a3cffdcaa7ba`, executable SHA-256
  `7f5ea49e6768ffddf0ad10e39dfb2cae48410789ce92842d4d1a42e7660338a0`.
- Qwen3.8-27B EXL3 2 bpw, target revision
  `e5e1f4b37c77641e465e24adf5f44d01bdd48de2`. The two target weight hashes
  match the [v2 identity record](../../v2-models-20261111/model-identities.json).
  MTP 4-bit head and six packed Q4 DFlash draft file hashes are in [hashes.txt](hashes.txt).
- Context 4096, temperature 0, top-k 0, repetition penalty 1; tools and prompt
  reuse off. The process-list supervisor checked every second for another known
  local inference, compiler or prover, and observed no conflict. Sampling cannot
  exclude shorter events or other GPU activity. No thermal/performance warning
  was recorded; GPU clocks, temperature and occupancy were not measured.
- System swap in use was 6.43 GiB before the ordinary process; the speculative
  pass probes ranged from 3.44 to 6.37 GiB. Caches and system memory were not
  reset. These changes limit comparison across ordinary and speculative runs.

## Results

The first prompt is the short French prompt in `compare_native.py` (69 rendered
input tokens); the second is the [v2 snapshot prompt](../../v2-models-20261111/prompt.txt)
(104 input tokens). All generated 128 tokens, including 127 decode transitions.

The ordinary run used one warmup and three measurements in one process. The
speculative runs used four fresh processes in MTP/DFlash/DFlash/MTP order,
with one warmup and two measurements per prompt and process. Medians and
ranges below are over individual measurements. The ordinary mode was not
interleaved, and no ordinary run was made for the French prompt.

| Prompt | Mode | Decode median (range) | TTFT median | Total median |
|---|---|---:|---:|---:|
| Snapshot 104 tokens | Ordinary | 8.88 (8.61–9.04) tok/s | 1.310 s | 15.612 s |
| Snapshot 104 tokens | MTP1 | 12.73 (12.57–13.23) tok/s | 1.329 s | 11.312 s |
| Snapshot 104 tokens | DFlash2 Auto | 6.55 (6.36–6.60) tok/s | 1.378 s | 20.766 s |
| French 69 tokens | MTP1 | 12.76 (12.49–12.97) tok/s | 1.103 s | 11.052 s |
| French 69 tokens | DFlash2 Auto | 6.00 (5.85–6.23) tok/s | 1.099 s | 22.253 s |

The speculative token, text and history hashes matched across modes on both
prompts; the ordinary snapshot output matched too. MTP accepted 56/71 and
58/69 proposals for snapshot and French; DFlash2 accepted 88/144 and 77/140.
These are bounded output checks, not a proof for arbitrary prompts. Observed
MTP/DFlash rates varied by up to about 6.5% across individual repeats and swap
use changed substantially. The ordinary measurement is unpaired. Do not infer
a stable percentage gain or a general ordering from these small samples.

The largest process lifetime footprint among speculative measured responses
was 11.19 GiB; this is not GPU-only or total system memory. [Individual ordinary
runs](ordinary.json), [speculative passes](speculative.json) and the
[supervisor status](supervisor.json) are retained with local paths, PIDs and
battery identifiers removed. All child processes exited normally.

## Reproduce

Use the same commands in the [35B report](../README.md#reproduce), substituting
the 27B target, compatible MTP head and packed DFlash draft. Check the machine
is idle and run only one inference campaign at a time. Record any contention,
failures, power, thermal and memory conditions; a successful process exit alone
does not establish output parity or a performance gain.
