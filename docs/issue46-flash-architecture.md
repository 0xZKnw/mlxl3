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

A1 staged Cold pipeline first; then B exact QSA q/out MB2; I release checkpoint inspection metadata in resident bridge; F long-QSA independent contracts before G persistent KV/selection. C/D/E/H/J follow measured cost: state layout, controller, resident MTP, prepared host segments and prefill. Separate decode/MTP/long-context/TTFT benefits. No retired K1/F01/F05/F06 variant or cache enlargement re-enabled. The issue remains in progress; only A0 is complete.

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
