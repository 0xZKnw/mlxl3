# Issue 46 — critical-path experiments

## A0: validated Hot/I/O diagnostic

Source45deec6d, M5 10-core GPU/24 GiB, AC. Flash Next2.05bpw65c89531, Q4drafta22a1744, fixedB2, expert cache3GiB/allocator64MiB. Probe checks MLX status/event without waiting. No release path, kernel or cache policy changed.

| Prompt | Batches with selected Hot routes, trace1 / 2 | Hot ready at I/O return | Inclusive I/O ms / emitted token | Peak footprint |
|---|---:|---:|---:|---:|
| MAIN | 1545 / 1547 | 87.90% / 88.49% | 25.50 / 26.35 | 8.57GB |
| HOLD | 1583 / 1589 | 87.30% / 86.78% | 26.91 / 25.82 | 8.54GB |

Hot was pending at I/O start except one sample. This supports trying Gate/Up-first reads overlapping GPU compute with Down I/O. It does not measure the exclusive exposed tail or guarantee a speedup. Untraced64-token controls:11.68–11.86tok/s; traces10.76–11.78. HOLD trace1 has noticeable overhead/variation. No gain claimed, no comparison to historical128-token throughput.

Both prompts pass12phases total against sequential tokens, all248320 finite logits and target states, plus independent golden text prefixes. The physical event test passes; all3processes were joined, zero new system swap-outs. Admission:10.19GB estimated available using native free-minus-speculative-plus-file-backed formula,9.5GB budget, compressor0.964GB. Reclaimability is an estimate. The diagnostic supervisor allowed256MiB global swap and10GiB RSS; native phase snapshots cap12GB footprint. No installed app changed.

Format/strict MLX+chat Clippy pass;154CPU tests pass/5ignored. Kani actual full-profile path crashes at intrinsics.rs:243 before obligations; GPU/FFI concurrency remains unproved. Initial command/configuration errors are retained. First compression used an older system Python lacking hashlib.file_digest; raw and complete archive were preserved, verified with project Python3.12, then remaining compression completed. No numerical result affected. Hosted CI not yet run for this local experiment.

[Summary](measurements/issue46/A0/hot-boundary-summary.json), [identity](measurements/issue46/A0/identity.json), [supervisor](measurements/issue46/A0/run-result.json). Lossless verified GZIP traces, full summaries and component CSVs remain under /Users/justin/Documents/mix-stq1_0/build/issue46/A0/MAIN and HOLD.

## Remaining priority order

A1 staged Cold pipeline first; then B exact QSA q/out MB2; I release checkpoint inspection metadata in resident bridge; F long-QSA independent contracts before G persistent KV/selection. C/D/E/H/J follow measured cost: state layout, controller, resident MTP, prepared host segments and prefill. Separate decode/MTP/long-context/TTFT benefits. No retired K1/F01/F05/F06 variant or cache enlargement re-enabled. The issue remains in progress; only A0 is complete.
