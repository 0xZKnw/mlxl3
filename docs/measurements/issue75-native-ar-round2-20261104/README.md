# Native Flash Next AR latency integration

The qualified **Apple M5** now enables fixed K2 projection dimensions, local routed-expert scratch reuse and HC-to-router command chaining by default. The three options retain explicit `0` rollback: `MLXL3_NATIVE_FLASH_FIXED_DIMS`, `MLXL3_NATIVE_ROUTED_SCRATCH`, `MLXL3_NATIVE_AR_HC_GATE_CHAIN`. Other devices retain disabled defaults; the fixed-shape path keeps its existing M5-only guard. HC chaining also requires depth 0 and no prefetcher.

The change preserves the canonical shader arithmetic, quantization, routing, expert-cache budget and recovery behavior. Actual component counters confirm 10→6 allocations per routed expert, four distinct output buffers, and 48 fewer separate routing command buffers per row. The wait API call count is unchanged.

## Measured behavior

32-token AR, MAIN/HOLD, one warmup and one measured response per prompt/process, reference/ensemble/ensemble/reference. Two measured samples per variant/prompt, context 4096, 3 GiB expert cache (48×53 slots), four readers, lookahead 0, prefix reuse disabled, temperature 0. Reference is the qualified main21cf binary `942d…`; candidate is `789a…` with all three options explicitly enabled. No compilation, prover or other inference overlapped the campaign.

| Prompt | Decode, reference → candidate | First token, reference → candidate | Complete response |
| --- | --- | --- | --- |
| MAIN | 5.877 → 5.862 tok/s (−0.26%) | 15.240 → 14.119 s (−7.36%) | −5.37% |
| HOLD | 5.950 → 5.936 tok/s (−0.23%) | 14.091 → 13.525 s (−4.02%) | −2.86% |

Decode throughput is effectively unchanged in this short screen. Both candidate latency samples are below both reference samples for each prompt. All eight warmups and eight measured responses match in full text, hash and counts. Allocated Metal memory is 6,608,617,472 bytes in all eight measured responses; process footprint and compression remain separate raw fields. Preserve the observed variation rather than treating the medians as a sustained-throughput result. No individual gains are inferred or added.

[Protocol](performance-protocol.json), [samples and summary](performance-summary.json), [artifact manifest](performance-manifest.json). AC power at 100%; background application activity was observed, temperature was not measured. Sparse attention beyond 512 rows and sustained/long-context speed remain unmeasured.

## Verification and final default

[Twenty-one qualification checks](quality/report21-summary.json) cover independent stock components, finite values and exact word counts, actual selected PSOs, scratch alias/poison/drop/error/recovery, independent 84-state/logit hashes, context 4096 and the real indexer, route ownership, prefetch fallback, restored prefixes, production cancellation and memory-budget recovery. The counter-checker mutation is included in that count. All process groups and readers exited.

The final default artifact is separately frozen as CLI `f4b592bc…` / libtest `af2c8eb3…`, minos26.2/sdk27.2. [Three default checks](default-quality/report.json) remove all six new/existing option variables from the actual child environments: all three initial states are true before mutation; the independent oracle executes 192 joint gates, fixed counts [3840,1920] and 5760 scratch uses; each context4096 branch executes 7200 scratch uses and 240 joint gates; the real CLI matches the independent two-token golden and 48×53 ledger without MLX. Inner shaders, selectors and arithmetic match the timed candidate; default-only policy is requalified without another timing campaign.

Final local suites: CPU 336 passed/9 ignored, direct Metal 345/47, MLX compatibility 381/265; strict Clippy in all three configurations passes. Earlier composed focused checks passed 117 Rust tests/8 ignored/174 filtered and 61 Python tests/44 Linux-only skips/26 subtests. Commands and logs: [default CPU receipt](../issue75-native-ar-defaults-20261104/cpu-result.json), [composed receipt](cpu-composed.json), [default freeze](../issue75-native-ar-defaults-20261104/freeze.json). Representative commands:

```sh
cargo fmt --all -- --check
cargo clippy --locked --features chat --all-targets -- -D warnings
cargo test --locked --no-default-features --features chat
cargo test --locked --features chat
cargo clippy --locked --features mlx,chat --all-targets -- -D warnings
cargo test --locked --features mlx,chat
```

Targeted Kani checks cover actual projection-key selection over arbitrary geometry/format/split inputs, scratch selection and wide prefix arithmetic over arbitrary u32/u64 inputs, and the shared default policy over hardware/presence booleans and valid UTF-8 strings up to four bytes (unwind10). The final shared-policy check passes 237 obligations and seven covers; seven standard-library guards are unreachable. The old-default mutation fails and restored code passes. Removed duplicate parsers retain their historical receipts. GPU-body verification remains blocked by the recorded Kani context/unsupported-path diagnostics; bounded CPU checks do not prove Metal numerics, concurrency or complete generation. See the adjacent fixed-dims, HC-gate, scratch and defaults reports for individual bounds and retained failed attempts.

The final branch also incorporates main d8438059: its additional Rust changes only widen existing cfg(kani) harness bounds, alongside CI, trace-analysis and historical reports. Runtime/policy/shader files match their qualified frozen sources. GitHub checks must be reported for the exact pushed head; local results do not establish a green CI run. No MTP/DFlash inference, installed app replacement or release is part of this change.
