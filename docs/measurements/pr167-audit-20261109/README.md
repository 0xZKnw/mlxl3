# PR167 audit corrections — 2026-11-09

These are native test prototypes on top of cf7ba172, not an installed application update. Production inference defaults remain exact; Q01 stays withdrawn and MB3 selection is test-only/off by default.

## Corrected behavior

- Recursive MTP hidden transport uses one reusable 40 KiB F32 GPU buffer and a four-byte finite-value flag. Exact IEEE widening handles signed zero and Half subnormals; encoding shares the existing draft-head command. It removes the CPU copy/conversion/allocation, not the per-proposal host token dependency. The old CPU transport remains an explicit comparison switch.
- Final cancellation routes successful operations through target+draft rollback before publishing tokens or counters, including D0. Prefix fixtures force rejection at each rank and EOS, compare full target GDN/QSA/hidden and draft state, and check resumption. The fixture executes both Q04-off and Q04-20%: eight final-cancellation cases per policy, sixteen total.
- Q04 choice, minimal starvation transitions and full validation are separate. Timed scalar runs do not read all512 probabilities or recompute the sorted oracle. Quality mode retains that oracle. Scalar picker buffers use bounded leases; block CPU choice keeps only three useful candidates on the stack and preserves sequential logical residency. The block picker still runs on the host; full GPU policy/state execution remains open.
- Actual three-row QMV and expert-consumer batches can use exact MB3 PSOs, retaining the canonical reduction and Half/split boundaries. B1/B2/B4 keep previous paths. All13 actual PSOs fit the M5 limit: 12,288 or24,576 static threadgroup bytes, device limit32,768. Register occupancy was not measured.
- Reports now include active context versus capacity, draft/repair/target work, conditional rank counters, logical/physical residency mismatches, cache clone counts and SSD service records/bytes even when Q04 is disabled. SSD requests are not NAND bytes; elapsed host phase timings are not exclusive GPU times. Shape observation is explicit and disabled during timing.
- Profile-controller cleanup handles macOS group-signal permission races while still reaping the direct child and confirming descendants. Genuine live-child denial remains a failure. Two new regression checks detect the original code. The recursive CPU baseline also satisfies the newer Clippy lint used by hosted CI.

## Same-binary speed pilot

Apple M5/24 GiB, Qwen3.8-Flash-Next EXL3~2.05bpw/head4/MUL1, same French MAIN prompt47tokens, capacity4096, active final offset110. Greedy,53-slot LFU target caches/3 GiB, four readers, resident routing on, HC workspace off. Fresh process per arm,64-token warmup then64 outputs/63 timed decode steps. One model at a time; compiler/prover/other-engine conflicts abort the arm. See [matrix](speed-r2/matrix.json), [summary](summary.json), [identities](identity.json), and [raw status](speed-r2/status.json).

| Configuration | Decode tok/s | Draft acceptance | Accepted / proposed | Measured responses |
|---|---:|---:|---:|---:|
| AR / off | 10.591 | — | — | 1 |
| AR / Q04 20% / fast | 11.087 | — | — | 1 |
| AR / Q04 20% / full oracle | 11.416 | — | — | 1 |
| MTP D1 scalar / off | 10.068 | 77.1% | 27/35 | 1 |
| MTP D1 scalar / Q04 20% | 10.898 | 77.1% | 27/35 | 1 |
| MTP D1 block / off | 9.638 | 77.1% | 54/70 | 2 |
| MTP D1 block / Q04 20% | 9.786 | 77.1% | 27/35 | 1 |
| MTP D2 / CPU hidden / MB2 / off | 9.181 | 69.2% | 72/104 | 2 |
| MTP D2 / GPU hidden / MB2 / off | 9.436 | 69.2% | 36/52 | 1 |
| MTP D2 / GPU hidden / MB3 / off | 8.851 | 69.2% | 72/104 | 2 |
| MTP D2 / GPU hidden / MB3 / Q04 20% | 8.856 | 69.2% | 36/52 | 1 |
| MTP D3 / GPU hidden / MB3 / Q04 20% | 8.600 | 51.4% | 38/74 | 1 |
| MTP D4 / GPU hidden / MB3 / Q04 20% | 9.004 | 48.2% | 41/85 | 1 |

Acceptance is accepted draft proposals / proposals, excluding bonus target tokens. Two-response rows show arithmetic mean throughput and summed acceptance counts; other rows have one measured response. Original interrupted runs and the pre-pause reference are preserved, excluded from resumed means. All32 resumed trajectories match the earlier same-policy AR references. These short single-prompt results do not qualify long-context or general performance.

MB3 is numerically correct but does not make D2 competitive on this workload. GPU transport alone is a small single-arm signal. Q04 benefits the scalar pilot; the block/deep composition and removal of the full oracle do not establish a speedup. Correct bounded candidates remain available for follow-up; their gains are not added. See summary.json for control drift. No new10% or D3/D4-off measurements were run.

## Validation and limits

Native release library372passed/139ignored; five physical gates explicitly selected and passed separately: all63,488finiteHalf bit patterns plus four nonfinite/extent negatives; independent12-case F16/F32 picker and transitions; B2/B3/B4 projection oracle; B1..B4 expert-consumer and failed-fill oracle; full-model recursive transport/prefix/rejection/EOS/cancellation/resume oracle.

Four Kani helpers passed. Publication is checked over symbolic success/cancellation/token values; geometry over allu32rows; recovery over allu8counters/two booleans. Selected-ID transitions use canonical original/candidate IDs0..19, symbolic ten u8 counters/hit/swap masks, unwind13 and an arbitrary untouched index10..511. No proof of full MTP transactions, Metal, allocation or concurrency. Earlier Kani CLI/type errors and compiler ICE are retained. Returning Option from the actual pure admission helper, with error context at callers, allowed verification without a stub or narrowed counter domain. CrossHair was unavailable.

Full portable Python:2742passed, one source-inventory mismatch,146skipped and123subtests. Only the affected inventory was refreshed; final focused inventory/controller/replay/manifest58passed+60subtests, then controller28passed+58subtests after a Ruff-only fixture cleanup. CPU/Metal strict Clippy, Rust formatting, Python Ruff and production release CLI build passed. Hosted CI must be read at the subsequently pushed head.

The earlier384-token quality profiles remain historical evidence; no new broad task-quality campaign was run under the user's quota instruction. Q02 affine deferred/centered/H128 variants, a less synchronous GPU-token draft chain, grouped/deferred KV repair, adaptive depth and a GPU logical-block picker remain separate research, not completed fixes. The prior norm→prepare negative pilot does not reject these Q02 variants. No merge, release, default activation or installed-app replacement.

Raw logs retain original whitespace. Frozen executables and model/reference payloads remain local, identified by hashes. Scripts record original experiment paths and refuse to overwrite completed campaign directories.
