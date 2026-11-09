# Q04 — expert substitution and MTP pilot, 2026-11-09

The engine default is unchanged. These are local native-model test prototypes, not a production CLI/app benchmark or a published feature.

## Scope and source

Adapted the same-command GPU picker and early resident-expert work from [Sushi 0bb32df5](https://github.com/beamivalice/sushi/blob/0bb32df5d1359f17a46eb2b904bbe1d9a8fd99ae/src/trunk.zig#L3492-L3660). The three highest-scoring original experts remain protected; at most three misses are replaced, each original mixture weight at most 0.10, with original mixture weights preserved and three-deferral recovery. Probabilities are reused; the gate is not recomputed. GPU scalar choices are checked against the host reference. Block choices use that host reference in row order. Sushi copyright/license is retained in `docs/licenses/sushi-MIT.txt`.

The first diagnostic prototype recomputed the gate and regressed by about 10–11%. The GPU-pick-only version still regressed at 10% and 20%. Both negative rotations are preserved. The later fix fed the replaced IDs into existing early resident computation; the previous test had disabled that path.

## Model and protocol

Apple M5, existing Qwen3.8-Flash-Next EXL3 ~2.05bpw checkpoint, context4096, greedy T0/K1/penalty1, cache3GiB/53slots per target layer, four readers, LFU, resident route on, HC workspace off, no enforced process ceiling. One French LFU/MoE prompt rendered with thinking=false. Each fresh process: 64-token warmup then 64 measured outputs, 63 timed decode steps. Exact rendered prompt/checkpoint/IDs, memory, power/thermal/VM snapshots and process-conflict checks are in raw reports/status files. GPU jobs were serial and separated from compilers/provers. Two attempts were stopped when external builds appeared; those reports remain failures, not speed results.

## AR and scalar D1

The six-arm AR Hot rotation measured means reference10.422244, Q04-10%11.128257, Q04-20%11.436433 tok/s (+6.77%/+9.73%). Reference drift was −5.64%; this is a promising short pilot, not a stable general gain. Every arm observed5280Hot submissions/run and matched earlier same-policy token trajectories.

The separate scalar D1 MTP rotation (`block_verify=0`) measured means reference9.959527, Q04-10%10.338952, Q04-20%10.790201 tok/s (+3.81%/+8.34%), control drift−0.57%. It is a different configuration from the block matrix below. Do not add gains or use it to infer a deeper-mode improvement.

## MTP blocks up to three, D1–D4

| Requested depth | Q04 off | Q04 10% | Q04 20% | 10% vs same-depth off | 20% vs same-depth off |
|---|---:|---:|---:|---:|---:|
| D1 | 9.861 | 10.201 | 10.275 | +3.44% | +4.20% |
| D2 | 8.929 | 9.038 | 9.557 | +1.23% | +7.04% |
| D3 | 8.749 | 8.324 | 8.697 | -4.86% | -0.59% |
| D4 | 9.173 | 8.953 | 9.191 | -2.40% | +0.19% |

Units: native decode tok/s. One measured pass per cell. D1 actually verifies at most two rows; D4 uses at most three-row chunks (five potential input rows =3+2), not a five-row kernel. Actual draft-length and verification-width histograms are checked and saved. All24 warm/measured64-token sequences match same-policy AR references; none silently falls back to D1.

On this matrix, D1/20% is fastest. D2/20% benefits relative to its own off control but is slower than D1. D3/D4 bring no useful benefit here; a single pass cannot establish small differences. Deeper recursive proposals are less often accepted, and their generation/repair/rejected target work is included.

## Numerical quality and decision state

The original seven384-token/six-domain teacher-forced profiles are preserved in `quality/` and `quality-summary.json`. At max3, 10% tolerance gave aggregatePPL+0.4608% (worst domain+3.1913%); 20% gave+0.0852% (worst domain+5.5298%). The initial strict per-domain screen failed; the user chose to continue these candidates. PPL is not an answer-error percentage. Neither aggregate result proves task quality.30%/40% were not selected.

Same-command GPU-picker integration was separately checked with12independent F16/F32 fixtures and384score records at tau0/.1/.2 exactly equal to previous targets. After the final Hot and deeper-MTP changes the user explicitly requested direct speed runs without another quality/full-test campaign. The inline64-token comparisons do not replace a broad numerical/task-quality validation.

For block MTP, logical cache metadata, starvation counters and positions are now prefix state. Verification snapshots keep each accepted prefix; rejected future reads may remain physically cached but do not influence routing decisions. Logical metadata uses copies/COW, not fake ownership of GPU weights. The draft is excluded from target Q04. Recursive draft hidden states are widened exactly Half→F32; replay from verified target hidden states repairs its prefix. Pending output buffers are bounded by requested depth. This new experimental implementation is test-only and the production Flash API still accepts onlyD1.

## Checks and limits

Before the final user-directed direct-speed changes: native release library369passed/137ignored; strict Metal and CPU Clippy/fmt passed on earlier qualified revisions, replay/native inventory/manifest30passed plus2subtests,44Linux proof-runner checks skipped. Native input-count Kani and the actual recovery helper passed; recovery scope was allu8counters andtwo booleans,6obligations/3covers. Those proofs do not cover the new full MTP transaction, Metal, allocation, concurrency or whole generation. CBMC and Python CrossHair were unavailable; actual Metal source compiled with Xcode and the selected physical oracle passed.

Final Hot/depth code: formatting, a real release test-binary build (`cargo test --locked --release --lib --features chat --no-run`) and the ordinary production-library check (`cargo check --locked --lib --features chat`) passed. The last check reported the existing future-incompatibility warning for `block` 0.1.6. No new full suite,384-token quality campaign, or source-level proof of the new prefix adapter was run after those edits, per user. All completed benchmark children were joined; failed/aborted runs are retained. Hosted CI had not run at this local evidence freeze. Publication is a separate user-requested PR; no merge, release or installed-app replacement.

## Evidence

The [throughput and draft-acceptance table](throughput-acceptance.md) covers all18 final configurations; [machine-readable rows](throughput-acceptance.json) retain accepted/proposed counts. Acceptance excludes bonus target tokens. AR and scalar-D1 throughput are means of two measured responses; each block-depth cell has one measured response.

Source/binary SHA-256 receipts are the `*-identity.json` files. Raw output tokens/text, acceptance and shape histograms, measured timings and memory are under each campaign directory. The original full reference logits and46.6MBroute-probability payload remain local under `build/issue165-20261108`, with earlier receipts; they are not duplicated here. Temporary executable binaries likewise remain local, identified by hash.

The prototype scripts capture the original experiment paths; run from the isolated issue165 worktree with the existing project Python3.12. They refuse to overwrite completed experiment directories.

Archived test logs retain their original whitespace, including interrupted lines and trailing blank lines. Source/documentation whitespace checks exclude these raw `.log` receipts. The final production-library check log is archived as `q04-pr-final-production-check.log`.
