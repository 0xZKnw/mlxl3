# Native Flash AR lookahead: no overall gain qualified

Keep `MLXL3_SSD_LOOKAHEAD_DEPTH=0`. This archives the completed 3 November screen; it does not rerun the model or change an engine default. All measured outputs passed the recorded text/hash/token/context/cache-capacity checks. Performance remains **inconclusive overall**: the last MAIN control fell 24.03%, while both HOLD candidates were slower than both HOLD controls.

| Decode tok/s | A: depth0 | B: depth1 | B: depth1 | A: depth0 |
| --- | ---: | ---: | ---: | ---: |
| MAIN | 4.7406 | 4.5465 | 4.3683 | 3.6014 |
| HOLD | 4.7665 | 4.4116 | 4.4578 | 4.6038 |

The nominal pooled MAIN change of +6.87% cannot be attributed to lookahead because its control collapsed. HOLD's nominal change is -5.35%, with two samples per condition. These short observations do not establish a sustained or universal effect. The cause of the late MAIN drift is unknown; GPU, thermals, SSD, and compression are not established causes.

The protocol used the native release CLI on an Apple M5 with 24 GiB, macOS 27.2 build26B123, Qwen3.8 Flash Next checkpoint `65c895314393431c09050b2e04e250836b3a6eb4` at an average 2.05 bpw. Source freeze: `89fdabbff6ef66ea131fab5ff95d28341959a061`; CLI SHA-256: `f20ba847f97a643100b3f1ffc8357b95b1eecc99c45c6728248534dddb76778d`. The reports retain the original self-reported `599628120c95-dirty` build stamp; it must not be mistaken for a clean source identity. Runtime version 2.0.2, bridge protocol 1; the archived harness SHA-256 is recorded in [copy-verification.json](copy-verification.json).

Each ABBA process produced one warmup and one measured response per MAIN/HOLD prompt: eight warmups and eight measured responses, 32 tokens each, context 4096, greedy temperature 0/top-k 1, repetition penalty 1, prompt-cache reuse disabled. NT2 and dynamic readers were explicitly OFF. Configuration was 3 GiB with exactly 3,174,942,528 B of expert-bank capacity, 53 slots in each of 48 layers, four demand readers, overlap ON, auto cache policy, and F32 prediction. Depth 1 additionally uses four background workers, an owned-prefetch cap of four, and predictive LRU. Equal expert capacity does not mean equal total CPU or memory resources. AC power at 100% was recorded before/after every arm; thermals were not measured.

[summary.json](summary.json) preserves every sample, timings, separate Metal/process/compression memory metrics, and the original decision. The four adjacent `screen-*/report.json` files are byte-exact originals. Their four empty stderr files are preserved as empty; they are not numerical evidence. [copy-verification.json](copy-verification.json) records source/copy hashes and the nonempty output, count, capacity, ABBA, and joined-cleanup consistency checks. Prior independent logits/state qualification is separate and is not rerun or newly proved by this archive.

All screen children were gone by 21:33:39 UTC. The separate Cargo-cache cleanup started no earlier than 21:48:54 UTC, so it cannot explain this earlier control collapse. No gain is added to the separate NT2 result. No installed application or engine behavior changed, and no repetition is proposed merely to obtain a more favorable sample.
