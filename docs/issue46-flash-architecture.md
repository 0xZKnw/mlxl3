# Issue 46 — critical-path experiments

## A0: validated Hot/I/O diagnostic

Source45deec6d, M5 10-core GPU/24 GiB, AC. Flash Next2.05bpw65c89531, Q4drafta22a1744, fixedB2, expert cache3GiB/allocator64MiB. Probe checks MLX status/event without waiting. No release path, kernel or cache policy changed.

| Prompt | Batches with selected Hot routes, trace1 / 2 | Hot ready at I/O return | Inclusive I/O ms / emitted token | Peak footprint |
|---|---:|---:|---:|---:|
| MAIN | 1545 / 1547 | 87.90% / 88.49% | 25.50 / 26.35 | 8.57GB |
| HOLD | 1583 / 1589 | 87.30% / 86.78% | 26.91 / 25.82 | 8.54GB |

Hot was pending at I/O start except one sample. This supports trying Gate/Up-first reads overlapping GPU compute with Down I/O. It does not measure the exclusive exposed tail or guarantee a speedup. Untraced64-token controls:11.68–11.86tok/s; traces10.76–11.78. HOLD trace1 has noticeable overhead/variation. No gain claimed, no comparison to historical128-token throughput.

Both prompts pass12phases total against sequential tokens, all248320 finite logits and target states, plus independent golden text prefixes. The physical event test passes; all3processes were joined, zero new system swap-outs. Admission:10.19GB estimated available using native free-minus-speculative-plus-file-backed formula,9.5GB budget, compressor0.964GB. Reclaimability is an estimate. The diagnostic supervisor allowed256MiB global swap and10GiB RSS; native phase snapshots cap12GB footprint. No installed app changed.

Format/strict MLX+chat Clippy pass;154CPU tests pass/5ignored. Kani actual full-profile path crashes at intrinsics.rs:243 before obligations; GPU/FFI concurrency remains unproved. Initial command/configuration errors are retained. First compression used an older system Python lacking hashlib.file_digest; raw and complete archive were preserved, verified with project Python3.12, then remaining compression completed. No numerical result affected. Hosted CI inspected on published head b55d7b0f: seven checks in progress, none completed at that snapshot. No green CI claim; later documentation-only heads are not covered by it.

[Summary](measurements/issue46/A0/hot-boundary-summary.json), [identity](measurements/issue46/A0/identity.json), [supervisor](measurements/issue46/A0/run-result.json). Lossless verified GZIP traces, full summaries and component CSVs remain under /Users/justin/Documents/mix-stq1_0/build/issue46/A0/MAIN and HOLD.

## Remaining priority order

A0 is retained as a test-only diagnostic. A1 was withdrawn after its negative throughput screen. B1 is closed below as numerically validated but unqualified for performance, with the prototype withdrawn. The remaining issue46 architecture work has not been implemented. On26October the user requested closure of QSA only and cancelled the proposed transition to issue52. No issue52 work was started. No retired K1/F01/F05/F06 variant or cache enlargement was re-enabled.

## Reproduction and verification scope

- `cargo fmt --all -- --check`
- `cargo test --locked --lib --no-default-features`:154passed,5ignored; focused `flash_profile::` selection:2passed.
- `cargo clippy --locked --all-targets --features mlx,chat -- -D warnings`
- `cargo test --locked --release --lib --features mlx,chat --no-run --message-format=json`
- `cargo kani --lib --tests --features mlx,chat --harness actual_detailed_profile_path --jobs 1 --output-format terse`: compiler ICE before obligations, no proof obtained.
- Exact physical test argv, environment and bounds are in [run.json](measurements/issue46/A0/run.json). Each profile used the existing verifier64 protocol, two traces and sequential oracle;12complete model phases total. The readiness test and two full-profile invocations were selected with `--exact --ignored --nocapture --test-threads=1`, then analyzed by the unchanged `scripts/profile_flash.py::analyze`. Extra readiness checks required nonempty actual probes, valid optional booleans and monotonic readiness on6264Hot/Cold batches.

This verifies the diagnostic paths exercised, not all hardware schedules, long contexts, cancellation/fault paths or a new production optimization. A1 is now archived below.

## A1: staged Cold reads rejected after the throughput screen

The prototype read Gate/Up first, submitted its activation, then read Down. It retained the additional GPU reader, attempted both drains on failure, and published slots only after complete reads. Physical quality passed36MoE cases, independent-oracle mutation detection, real disposable-file Down failure/recovery,36prefix checks and43production bridge checks. The first recovery test had an incorrect second-pipeline expectation: the existing failure policy invalidates the entire cache. It was corrected to check Cold fallback and later staged recovery; the failed first run is retained.

| Fixed DFlash2 B2,128warm+128measured | Control | Candidate | Difference |
|---|---:|---:|---:|
| MAIN | 12.065689 | 11.675183 | -3.236% |
| HOLD | 11.679307 | 11.268663 | -3.516% |

Acceptance, verified rows, logical expert bytes and all eight full128-token golden answers match. Verification cost increased77.39→79.86ms/emitted token on MAIN and79.92→82.86 on HOLD. These are inclusive host times. All processes/readers/samplers joined; zero new swap-outs, process ledger swap-ins0B. Peak footprint stayed below7.84GB. Compression/decompression varied between processes; this single H-C screen does not establish a universal causal regression. It provides no reason to promote the prototype. Reverse-order and OFF/MTP speed checks were not run after this negative screen.

The prototype is withdrawn from executable sources; [withdrawal](measurements/issue46/A1/withdrawal.json) proves exact restoration to A0. Its [patch](measurements/issue46/A1/prototype.patch.txt), [results](measurements/issue46/A1/screen-analysis.json) and quality evidence are retained. No installed app changed, no gain claimed. Kani checked the actual staged partition for symbolic slots0..511, unwind16:0/188failed,2expected unreachable checks,2/2covers. This does not prove GPU/FFI concurrency. The initial plan, failed test, corrected test and raw observations remain in /Users/justin/Documents/mix-stq1_0/build/issue46/A1.

Next priority: B, exact QSA q/out projection batching; no staged-I/O combination.

## B1: QSA B2 closed without performance promotion

The prototype grouped the independent query and output EXL3 projections for exactly two positions using the existing reference MB2 kernels. KV append, index pooling, normalization, RoPE and attention remained causal singleton operations. Prefix snapshots were retained after the first position. The12QSA layers could then use the existing multi-row HC/PLE preparation. B1 and B3–B8, dense projections, quantization, shaders, SSD scheduling and cache sizes were unchanged. No runtime flag was added.

Prototype source: `2fe20263b5ff69380f35f50201672109152ca34f`; CLI SHA-256: `2b9129e6f0f0a75ea3e38cee60e5f565f1a602291370a0e34cc435a007031d26`. Stock control source0842599c2446, SHA-256619d3eb0e4a1affedb4ff296020d88014dec4682fc8aee7cba46ea2f09f6210c. Both copies used identical MLX/JACCL/metallib hashes, recorded in [identity](measurements/issue46/B1/identity.json). The only native difference between that stock source and the experiment baseline8668793b was the test-only A0 probe. No A1 code was active.

### Correctness before speed

- 240 nonempty, finite, byte-exact projection comparisons:60real QSA projections, F16/F32, contiguous/strided inputs and two distinct rows, against two independent singleton calls.
- 84 QSA output/state/prefix/continuation cases:12layers at initial offsets0/3/7/2050/2051/2052/4094. Long states were synthetic native states; this does **not** qualify the independent upstream long-context contract in issue46-F.
- Historical independent upstream QSA stage oracle passed. All36DFlash prefix/capture/logit/state checks passed and required a real batch-call counter on every QSA layer. All43production bridge parity/error/cancellation/recovery checks passed.
- MTP target/head state and greedy-sequence parity passed for budgets1/2/3/17.
- Format and strict all-target MLX/chat Clippy passed; CPU library154passed/5ignored. Frozen release MLX/chat test binary: initial sandbox run178passed/3loopback-HTTP permission failures/157ignored; unrestricted rerun181passed/0failed/157ignored. Ignored hardware tests are separate from the explicitly selected physical cases above.
- `cargo kani --lib --tests --features mlx,chat --harness actual_qsa_batch_entry --jobs 1 --output-format terse` attempted the actual implementation and crashed in the Kani compiler at `intrinsics.rs:243` before obligations. No formal proof of the QSA/GPU path was obtained. No stubs or artificially restricted replacement were used to claim success.

Exact physical commands, environment and assertions: [quality plan](measurements/issue46/B1/quality/run.json), [outcomes](measurements/issue46/B1/quality/run-result.json), [bridge checks](measurements/issue46/B1/quality/checks.json). All children/readers/samplers were joined. Build/test commands used `--locked`, MLX0.32.2, the existing macOS SDK/MLX root and one compilation job. Normal test argv: `B1/native-tests --test-threads=1`; CPU `cargo test --locked --lib --no-default-features`; format `cargo fmt --all -- --check`; lint `cargo clippy --locked --all-targets --features mlx,chat -- -D warnings`; CLI build `cargo build --locked --release --features mlx,chat --bin mlxl3-rs`.

### Actual DFlash2 B2 throughput

M5 10GPU cores/24GiB, AC, Flash Next2.05bpw65c89531, Q4/ring drafta22a1744, fixed one proposal/two target positions,3GiB/53expert slots,64MiB allocator cache, four readers, GPUHot/Wide/multi-token/unique-cohort enabled; lookahead/LES/shared-EXL3/shared-batch/prefetch disabled. Each child warmed128tokens then measured128(127decode), one target at a time, no compiler/prover during inference. Each pair used a fresh control and candidate process. These are end-to-end bridge measurements, not isolated kernel timings.

| Prompt/order | Control tok/s | Candidate tok/s | Difference |
|---|---:|---:|---:|
| MAIN H-C | 11.766785 | 12.079032 | +2.654% |
| HOLD H-C | 11.364061 | 11.556398 | +1.693% |
| MAIN C-H | 10.774796 | 11.752067 | +9.070%, unqualified amplitude |
| HOLD C-H | 11.254705 | 11.466843 | +1.885% |
| MAIN H-C recheck | 11.806061 | 11.864953 | +0.499% |

The+9.07% MAIN pair has an8.43% slower stock control than the first pair; it cannot substantiate that gain. HOLD is positive in both orders. MAIN remains positive but its effect size varies substantially. Counts, acceptance, logical expert bytes and full128-token golden answers match within all pairs. Recheck verification cost78.970→78.593ms/emitted token; logical expert bytes188.430MB/decode token and138verified rows are identical. No SSD-byte saving is claimed.

### OFF/MTP1 control prevents promotion

| Prompt/mode, single pair | Control tok/s | Candidate tok/s | Difference |
|---|---:|---:|---:|
| MAIN OFF, H-C | 10.798500 | 9.745652 | -9.750% |
| HOLD OFF, C-H | 10.486043 | 9.750983 | -7.010% |
| MAIN MTP1, H-C | 10.929094 | 10.398500 | -4.855% |
| HOLD MTP1, C-H | 10.084632 | 9.678674 | -4.026% |

The candidate MAIN warmups were slightly faster for OFF and MTP, then the measured results reversed. OFF never uses the new B2 path, but the scalar QSA helper was refactored; MTP verification also calls `prepare_rows` and does use the new B2 path. The source/code-generation and machine-condition contributions were not isolated. These single pairs establish an observation, **not** a causal or reproducible QSA regression. Neither a reverse-order OFF/MTP campaign nor a new variant was run after the user requested closure.

All20warm/measured answers in this final campaign match the independent128-token golden references, with identical mode/acceptance counters. Peak process footprint8,221,186,600B; process logical swap-in ledger0B. Global swap-outs rose65,536B during the stock DFlash control, with attribution unknown; all other children in that campaign had0new global swap-outs. Earlier completed B1 physical and throughput campaigns had0new global swap-outs. Compression/decompression varied across processes; it was observed, not established as the cause. The first speed attempt stopped during pre20on a power-state change before any inference, then was rerun with a fresh AC preflight. An initial ad-hoc result extractor used an incorrect mode filter and failed with KeyError before writing any analysis; the corrected extractor explicitly checked all20answers.

[First screen](measurements/issue46/B1/screen-v2/analysis.json), [reverse order](measurements/issue46/B1/confirm/analysis.json), [final controls](measurements/issue46/B1/controls/analysis.json), [final plan](measurements/issue46/B1/controls/run.json), [raw supervisor](measurements/issue46/B1/controls/run-result.json). Reports, system observations and failed attempts are retained alongside them; complete local binaries/build artifacts remain under `/Users/justin/Documents/mix-stq1_0/build/issue46/B1`.

### Closure and integration

**Numerical validation passed; performance promotion did not.** The prototype was withdrawn rather than enabling it by default or adding another experimental option. The three changed existing source files are byte-identical to8668793b and the added QSA test file was removed from active compilation. Its complete [patch](measurements/issue46/B1/prototype.patch.txt), proof harness, commands, source/binary hashes and evidence files are archived. [Withdrawal manifest](measurements/issue46/B1/withdrawal.json) records exact restoration; private K1 files were preserved. The shared CLI build artifact is rebuilt from the restored source; the installed Desktop/engine remain stock beta2 source0842599c2446. No app replacement, release, merge, issue46 closure or issue52 implementation is included. 15tok/s was not reached. The underlying cause of the OFF/MTP timing reversal remains unverified.

Restored CLI build completed successfully with MLX/chat from sourceae20d395; loader/`--help` passed without loading a model. [Final artifact](measurements/issue46/B1/final-restored.json) records its hash and exact equality of the entire native tree to8668793b. The subsequent closure commit only updates documentation/evidence. No new model throughput is attributed to that restored build. CI workflows already cover PRs and main pushes with pinned MLX0.32.2 and strict native checks; physical model checks remain local.
