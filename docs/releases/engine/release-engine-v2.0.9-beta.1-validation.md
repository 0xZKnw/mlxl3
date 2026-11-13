# MLXL3 Engine 2.0.9 beta 1 — validation

Engine source `e7923197cbafc4f63688a8d5cb4a2ba455c4e6e9`. The archive was built from a clean checkout. The follow-up packaging fixture and measurement records are separate commits.

## Archive and compatibility

**MLXL3-Engine-v2.0.9-beta.1-arm64.tar.gz** · 68,086,449 bytes · SHA-256 `7d8c6eb1f5c116d16783f443f499aad41b24f2b683fe34859006ac3075b56c6c`. The signed arm64 binary reports version 2.0.9, MLX, chat enabled and source revision `e7923197cbaf`. The macOS deployment target is 26.2. The Desktop updater remains on its existing channel.

| Component | Bytes | SHA-256 |
|---|---:|---|
| libjaccl.dylib | 1,546,672 | `b6bdf1d1d472298217a0fb27893e15c35e78d1436f2bd023dc711ef84640b2b3` |
| libmlx.dylib | 21,815,104 | `40cb0b60beb0a1dde0e13647a644ece3625d6dd1a048ac846005db2c67c9fff2` |
| mlx.metallib | 182,351,120 | `dc59d1cceb1a5c7e578232e6e41e28e2c73c9463ac6dbc3886c3ee17ffc270ed` |
| mlxl3 | 12,682,768 | `1bef4f88d92f8264cb5b1785c6ebedc1e21cb91d5df6698a7d6492f91bfb65e7` |

Both archives were independently extracted, checked against their file manifests, and executed from their relocated directories. All three Mach-O objects in each release passed arm64 and strict ad-hoc signature checks.

## Decode throughput of the packaged engines

Apple M5, 24 GiB, macOS 27.2, MLX 0.32.2. Qwen3.6-35B-A3B EXL3 2.49 bpw, MCG K2/K3 experts, K4 dense, K6 target head, same affine 4-bit MTP head. Greedy, context 4096, prompt cache off, model-resident decode. The French and code prompts tokenize to 27 and 43 tokens. Each engine ran twice per prompt in BAAB order, with one warmup followed by two measured generations of 128 tokens on each pass. Each generation contributes 127 post-first-token decode steps. On battery, no overlapping local GPU benchmarks or compiles.

| Mode | French, 2.0.8 → 2.0.9 tok/s | Change | Code, 2.0.8 → 2.0.9 tok/s | Change |
|---|---:|---:|---:|---:|
| AR | 47.98 → 51.10 | +6.51% | 45.77 → 50.03 | +9.30% |
| MTP1 | 78.17 → 80.34 | +2.78% | 76.40 → 78.56 | +2.84% |
| MTP2 | 64.51 → 67.89 | +5.25% | 65.72 → 68.39 | +4.06% |
| MTP3 | 60.20 → 63.30 | +5.15% | 61.56 → 61.53 | -0.05% |

| Mode, French / code | Prefill change | TTFT change | End-to-end latency change |
|---|---:|---:|---:|
| AR | +0.15% / +1.80% | -0.11% / -2.07% | -5.72% / -8.07% |
| MTP1 | +0.78% / -0.09% | -0.75% / +0.10% | -2.56% / -2.48% |
| MTP2 | -0.55% / +3.54% | +0.62% / -3.49% | -4.74% / -3.87% |
| MTP3 | +3.54% / -3.35% | -3.61% / +3.40% | -4.91% / +0.26% |

## Kernel isolation and repeatability

The isolated kernel compares a common local engine with only the expert layout changed. Other local experimental paths in that baseline were excluded from the release. Do not add the following deltas to the archive comparison. Absolute results varied between passes.

| Experts, comparison | French, baseline → candidate tok/s | Change | Code, baseline → candidate tok/s | Change |
|---|---:|---:|---:|---:|
| AR ABBA | 47.49 → 47.69 | +0.42% | 47.44 → 47.74 | +0.63% |
| MTP1 ABBA | 67.52 → 69.33 | +2.68% | 64.93 → 66.71 | +2.75% |
| MTP1 BAAB | 72.04 → 78.21 | +8.57% | 73.07 → 75.31 | +3.08% |
| MTP2 ABBA | 65.35 → 68.90 | +5.45% | 67.62 → 70.32 | +3.99% |
| MTP2 BAAB | 65.84 → 68.50 | +4.04% | 66.40 → 69.28 | +4.33% |
| MTP3 ABBA | 66.64 → 67.32 | +1.01% | 65.67 → 67.39 | +2.61% |
| MTP3 BAAB | 65.39 → 66.97 | +2.41% | 63.96 → 65.22 | +1.97% |
| MTP3, local baseline → release BAAB | 62.29 → 62.59 | +0.47% | 60.78 → 61.92 | +1.88% |

The paired workload has 58/69 and 57/70 accepted proposals in MTP1 for French/code, 72/108 and 73/107 in MTP2, and 81/138 and 80/139 in MTP3. Each completed campaign validated token IDs, text and history. K4 regressed across the MTP depths and K6 regressed in MTP1 and MTP3. The first K6 build reused an earlier shader binary. It was excluded before timing and then rebuilt with a private target. All reported results passed binary marker checks.

## Numerical and functional verification

The release source passed six explicit Apple GPU tests covering replay rebinding/error cases, mixed I16/U16 expert banks, rejected bank formats, physical expert layouts at T1/2/4, complete 35B snapshots, and recursive MTP sessions. All 12 snapshots matched 248,320 finite F16 logits and 80 cache/state arrays per position at prefills 23/128/256 with three further forced tokens. The harness observed 180 replay applications over 30 enabled layers. Sessions exercised accepted and rejected proposals at depths 1–3 and adaptive transitions. The packaged CLI matched four further full logit vectors. Real bridge requests passed fixed depths 1–3, budgets 1/2/3/8/17/64, cancellation and recovery. All checks used identical weights and quantization.

The accompanying public benchmark receipt contains median decode/prefill, TTFT, total latency, per-pass warmup and measured samples, process footprint and MLX peak allocation. The separate [35B profiling issue](https://github.com/0xZKnw/mlxl3/issues/172) includes sampling coverage, incomplete traces, the qualified shader rankings and CPU waits. The profiling tools did not provide DRAM throughput or ALU utilization.

## Local checks and verification bounds

- On the clean engine source, Cargo fmt and strict all-target Clippy passed for both CPU/chat and MLX/chat. The full CPU/chat suite passed 526 tests with 10 ignored. The full MLX/chat suite passed 565 with 254 ignored. Six additional physical selections were run explicitly, one at a time. The CLI and bridge were also tested directly against the extracted release binary.
- Kani 0.68.0 with CBMC 6.11.0 checked five harnesses on the source used to build the archive. The expert-layout guard covers symbolic row count, all four other shape dimensions, quantizer width and codebook flag (68 obligations). GDN replay dimensions and default/override guards account for 27 and 42 obligations. Bank layout arithmetic accounts for 102, and diagnostic route admission for 151 with one unreachable obligation. Route admission used arrays of up to eight IDs with unwind bound 12. No failures, 14/14 cover goals reached. The GPU, allocator, OS and general model quality are outside these proofs. CBMC could not parse the actual C++ replay wrapper's libc++ headers in the prior qualification. No C++ proof is claimed.
- Python packaging, inventory and Kani registry tests passed 95 cases plus two subtests, with 66 platform/feature skips. The first hosted packaging-fixture run exposed a missing TensorFold notice in its disposable project. The fixture was fixed, all 25 targeted cases passed locally and the hosted native-coverage job passed on its follow-up commit.
- The Desktop updater installed the signed prerelease from its archive into a disposable store, located and executed it, and passed fallback checks. The temporary test driver supplied the beta tag carried in the artifact. Both bundled dylibs were observed loading from the relocated candidate. The draft archive downloaded from GitHub matched the local archive byte for byte.
- The three isolated prototype library suites each passed 465 tests (247 ignored for the experts, 246 for each reader). The K4/K6 integer extraction checks model the SIMD shuffle and cover 23 and 32 obligations. The initial stale K6 binary was excluded and the verified rebuild was used for results.

## Reproducibility

The two archives were compared with `benchmarks/compare_native.py` using `--tokens 128 --warmup-tokens 128 --order BAAB --repeats 2 --settle-seconds 2`, the repository code-prompt file, and `--mtp-depth 1|2|3` with the same sidecar for MTP. The isolated trials used the same workload and explicit per-campaign order. The [per-pass benchmark data](https://github.com/0xZKnw/mlxl3/blob/main/docs/measurements/qwen36-profiles/qwen36-engine-2.0.9-bench.json) provide all 21 campaigns. The [profile measurements](https://github.com/0xZKnw/mlxl3/blob/main/docs/measurements/qwen36-profiles/qwen36-35b-m5-profile.json) include all qualified shader windows and the separately recorded main thread states.
