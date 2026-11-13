# Issue164 native M5 speed pilot — 2026-11-08

The optimized composition reaches **10.35 decode tokens/s** on this short MAIN64 pilot: **+23.86%** over the same-binary ablation and **+6.05%** over the single AR control. This is not a broad workload qualification or a 20 tokens/s result.

## Identity and protocol

Source `8b3bf72ec3bfe805eb8f1fec2899e79d27446662`; frozen CLI SHA-256 `32b3f074b633d54cc6a052e4b9cc637fa5cb6372181f7ea754da2c2befee9316`. The binary embeds the precommit dirty revision; per-source hashes in [identity](final-identity.json) bind it to the measured source. Apple M5, 24 GiB, AC power, Qwen3.8-Flash-Next EXL3 2.05 bpw/head4/MUL1, context4096, greedy T0/K1/penalty1, 3 GiB expert cache (48×53 slots), four readers, no prompt cache or SSD lookahead. One 64-token warmup then one measured response per process, serial A1/B1/AR/B2/A2, no simultaneous compiler/prover/inference. Exact settings/checkpoint metadata hashes: [plan](speed-rotation-no-cap/plan.json). Conditions and process/Metal memory are retained per arm.

A is the same final CLI with the new kernel/arena/segment flags disabled and common executor enabled; **it is not the original R3 binary**. B enables the qualified composition, Hot+HC workspace+deferred, finish/mix disabled. AR uses B settings with MTP disabled and the draft unloaded. MTP remains request-controlled.

## Measured responses

| Arm | Decode tok/s | Prefill s | TTFT s | Total s | Metal allocated GiB | Process footprint GiB |
|---|---:|---:|---:|---:|---:|---:|
| A1 | 8.4812 | 10.3189 | 10.3201 | 17.7484 | 6.3632 | 6.9548 |
| B1 | 10.3445 | 8.8111 | 8.8119 | 14.9021 | 6.5890 | 7.0944 |
| AR | 9.7589 | 8.8905 | 8.8910 | 15.3466 | 6.3844 | 6.8841 |
| B2 | 10.3547 | 8.9017 | 8.9025 | 14.9868 | 6.5890 | 7.1029 |
| A2 | 8.2307 | 10.4259 | 10.4268 | 18.0811 | 6.3632 | 6.9639 |

All measured responses generated64tokens/63decode with identical nonempty text/hash `d7c2f53d11d70633`; both MTP arms accepted28of34proposals. All children/readers joined. Controls drift −2.95%; optimized repeats differ +0.10%. The small sample and one prompt limit generalization. B uses about231MiB more Metal allocation than A. AR and MTP have different draft residency and actual memory: no equal-process-budget claim.

The first configured24576MiB request-ceiling preflight **failed before MTP generation**, with a conservative full-context block admission rejection. Its [failed report](speed-rotation/preflight-budget/report.json) remains preserved. The revised protocol explicitly removed that ceiling before any64-token measurements; hardware/cache stayed equal. The new defaults retain the previous profile when a request ceiling is configured.

## Default engine integration and validation

The subsequent default-selection change promotes this composition on **exact Apple M5** without changing shader arithmetic. Strict0/1 overrides remain. `MLXL3_NATIVE_FLASH_COMMON_EXECUTOR=0` disables the new automatic profile; configured request ceilings/nonzero lookahead keep the previous profile. Explicit finish/mix avoids automatically enabling Hot scratch. SSD cohorts and MTP block verification default on only when actual layers admit the combination; incompatible cache/route settings retain scalar verification. Other hardware and unrelated experiments keep existing defaults. No installed app or release artifact is replaced.

The measured runtime previously passed365librarytests/133ignored, independent exact-Half component oracles, and full-model B2/B3/B4 every-prefix logits/hidden/state/continuation/drop/late-error gates. See [implementation status](../../issues/163-164/issue164-native-progress.md) and `opti.md`. The CPU portability repair passes strict Clippy with/without Metal,353CPUlibrarytests/10ignored, and six targeted Kani harnesses. Final default-source release library: **366 passed / 133 ignored**; strict CPU/Metal Clippy, formatting and CLI build pass. Default selector tests and three targeted Kani harnesses pass; the profile and strict parser are checked for symbolic booleans/UTF-8 strings up to2bytes. These CPU bounded checks do not prove Metal, allocator or concurrency. Final default bridge receipts are archived alongside this report.

The complete issue164 is not implemented: N08/N10/N11/N13/N15 and remaining N06/N07 stages are pending. No claim closes issue164. Broad holdouts/contexts, full ignored physical suite and new Metal debug validation were not run. User explicitly requested merge without waiting for CI; the previous head had two macOS process-cleanup PermissionError failures in the profiling-controller tests and a CPU Clippy failure repaired here. Exact final-head CI must be read separately; local success does not imply hosted success.
