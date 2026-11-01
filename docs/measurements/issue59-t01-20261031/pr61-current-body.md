# Engine: T17/T04/T18a experiments and staged optimization checks

Head: `a36ba5077bf386466660d6b478058fbe6880245c`. https://github.com/0xZKnw/mlxl3/pull/61

Prepare three independent engine experiments from the revised priorities in #59 and the profiling limits in #60. All three remain **OFF by default**; this draft makes the implementation and M5 validation procedure available for review. No measured throughput gain is claimed.

## Changes

- **T17 — GDN epilogue:** compile only `norm.astype(F32) * sigmoid(z.astype(F32))`, then cast to F16 before the existing output projection. Preserve the native norm per row, H128, recurrence and tapes. Batch the epilogue for real B2/B4/B8 verification; retain the reference path and strict dtype/shape guards. This uses sigmoid, with no extra `z` factor.
- **T04 — Cold SwiGLU/down boundary:** preserve the existing SwiGLU arithmetic in GU's F32/F16 dtype, explicitly round the activation to F16, multiply by `down.suh` in F16, then use native H128 and the existing EXL3 splits/reductions. Restore the original output dtype. Restrict the experiment to unmasked, ungrouped Cold slots; Hot submissions and fallbacks remain intact.
- **T18a — reader work assignment:** optionally replace static reader stripes with a bounded atomic job queue, retaining 1–4 readers, validation, destinations, error draining and full-batch publication. Teach profile validation the actual policy while keeping strict compatibility with old static traces. This does not implement T18b progressive Cold publication.
- Add five physical numerical oracles, portable concurrent queue tests, proof harnesses, CI coverage and a recorded M5 procedure. No Desktop files are changed.

## Faster optimization procedure (documentation only)

A separate commit, `a36ba507`, updates the actual root `opti.md` and one necessary link in the M5 guide. It adds three stages: a focused correctness smoke with early rejection; qualification of affected production paths before timing; and complete relevant checks plus fair MAIN/HOLD confirmation before activation/merge.

This removes automatic full-suite/full-matrix repetition from each edit, allows documented reuse of unaffected evidence, and avoids carrying historical fixed repetitions, pauses or power requirements into unrelated functional checks. It retains numerical fidelity, KV/recurrent state and rollback, affected MTP/DFlash paths, memory budgets, GPU isolation, applicable verifier attempts and final CI. AGENTS.md, executable code, system protections and all historical results are unchanged.

The existing tools support an initial pilot of **2 processes / 4 generations** (`--prompt MAIN --tokens 16 --repeats 1`) after numerical/integration qualification; the recorded complete per-candidate matrix uses **24 processes / 96 generations**. A failed idea can stop after its first relevant oracle. These are stage/work counts, **not a measured or guaranteed time saving**, and the pilot cannot establish a model gain or replace final qualification. No universal noise-hiding threshold is introduced.

The documentation follow-up reran **81 existing CPU benchmark/supervisor tests** (a subset of the earlier relevant suite), and checked actual CLI help, shell syntax, selectors, links, schedule counts, byte-preservation of historical entries and unchanged executable source hashes. No GPU/Metal benchmark or Mac action was run. See the [current procedure](https://github.com/0xZKnw/mlxl3/blob/a36ba5077bf386466660d6b478058fbe6880245c/opti.md#procédure-courante--exploration-courte-qualification-proportionnée--30-octobre-2026).

## Independent options

| Candidate | Enable | Reference |
| --- | --- | --- |
| T17 | `MLXL3_FLASH_GDN_EPILOGUE=1` | `0` or absent |
| T04 | `MLXL3_FLASH_SWIGLU_DOWN=1` | `0` or absent |
| T18a | `MLXL3_SSD_DYNAMIC_READERS=1` | `0` or absent |

Keep the other candidates at `0` for each A/B. T18a reads its option at the first batch on each thread; compare it in separate processes. These are numerical-contract experiments: explicit casts in a compiled graph are not a substitute for bitwise qualification of the Metal result.

## Cloud checks

- Rust CPU with `--no-default-features --features chat`: **245 passed, 6 ignored**; Linux `runtime_gpu` selected **0 tests** and proves nothing about GPU behavior.
- CPU `cargo fmt --check`, strict Clippy and build passed.
- Queue tests: **4,084 cases per run** under ASan+UBSan and TSan; a temporary missing-job mutation was detected. The actual GCD/preadv/lease integration was not executed on Linux.
- Relevant Python suite: **312 passed**; Ruff, format and compilation passed. Profile regressions reject corrupt/incomplete dynamic traces and retain legacy static validation.
- Rust MLX/frontend check and strict Clippy for all targets passed through an **isolated temporary manifest** for `aarch64-apple-darwin`: build.rs disabled, original source paths, and tokenizers fancy-regex replacing native cross-build dependencies only in that temporary manifest. This is **not a shipped build, FFI link or Metal compilation**.
- Actual `mlx,chat` build was blocked by the Linux host; native disk-reader compilation lacks libdispatch. Initial cross-check failures are retained. Kani, CBMC and CrossHair were attempted but unavailable; harnesses are retained and no formal proof is claimed.

The archived cloud report describes the pre-publication phase; this draft is the subsequently authorized publication. Full commands, results, failed attempts, source hashes and issue snapshots: [cloud report](https://github.com/0xZKnw/mlxl3/blob/a36ba5077bf386466660d6b478058fbe6880245c/docs/measurements/cloud59-20261030/README.md), [verification receipt](https://github.com/0xZKnw/mlxl3/blob/a36ba5077bf386466660d6b478058fbe6880245c/docs/measurements/cloud59-20261030/verification.json).

## Required M5 validation before promotion

Use the same pinned Flash EXL3 checkpoint/draft, 3 GiB target cache and 4 readers. Build and test the real Apple variant before timing:

```sh
mkdir -p build/cloud59-m5
cargo fmt --check
cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
cargo test --locked --features mlx,chat
cargo build --locked --release --features mlx,chat
cargo test --locked --release --lib --features mlx,chat --no-run --message-format=json > build/cloud59-m5/build-tests.json
```

Then use the exact-test runner and environment setup in the [M5 procedure](https://github.com/0xZKnw/mlxl3/blob/a36ba5077bf386466660d6b478058fbe6880245c/docs/measurements/cloud59-20261030/README.md#ce-soir--qualité-avant-vitesse): it rejects zero selected tests, saves logs, bounds each test and stops on failure. Run the five new tests and existing SSD, full logits/states, rollback, MTP and DFlash oracles alone, with the required independent fixtures. **None of these native Apple/Metal/model checks has been run in this cloud session.**

Only after numerical qualification, run independent release A/Bs per candidate, mode and MAIN/HOLD prompt using the recorded ACCA matrix, warmup128 + measured3×128 per process. AR, MTP and DFlash2 are separate campaigns. For example, with the documented environment and local checkpoint paths:

```sh
python scripts/benchmark-flash-bridge.py target/release/mlxl3-rs "$MLXL3_FLASH_MODEL" build/cloud59-m5/AR-T17-MAIN-C --cache-gib 3 --prompt MAIN --tokens 128 --repeats 3
# MTP: add --mtp; DFlash2: add --dflash-draft "$MLXL3_FLASH_DRAFT_MODEL" --dflash-mode 1
```

Compare full outputs and state oracles before throughput; record source/binary hashes, power, thermal and swap conditions, reference drift and independent process counts. Aggregate decoded tokens / elapsed decode time rather than adding unrelated percentages. Actual dispatch elimination, native numerical equality, spills, I/O latency effects and M5 throughput are **unmeasured**. A failed parity check blocks timing/promotion.

## Scope and source warning

The clean published baseline was `f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6`. The unpublished local changes cited by the issues (`7978fce…` and the October 29 measurement checkout) were not available in the cloud. **Compare their diff before applying this branch to Justin's current checkout; absence of duplication against those unseen changes is not established.**

T01 is deferred pending post-hit attribution/private-backend feasibility; T07/T09/T10/T12 need real projection/route/spill evidence. T18b needs a separate lease/cohort/sum-order protocol; T19 needs real chunked prefill. This does not reactivate the rejected ARC/static-cache/grouped variants, change Metal thresholds, remove Hot submission, or promote T05's mixed MAIN/HOLD result.

Draft for review and M5 testing only. No merge, deployment, installed-app replacement or performance promotion is requested.

