# COLD-READY-01: partition and readiness audit

1 November 2026. This report covers the new CPU partition helper and an
independent review of the existing trace calculation. It does not qualify the
asynchronous reader, the Metal consumers, model outputs, or performance.

## Contract and implementation

`native/src/cold_ready.rs::partition` takes ordered Cold occurrences `(row, slot)`
and the slots observed complete. It returns two stable lists: every occurrence
whose slot is ready, and every remaining occurrence. Duplicate experts and even
identical input tuples remain separate occurrences. Empty and duplicate readiness
lists are supported. Membership uses only equality; there is no offset arithmetic,
unsafe code, global state, or GPU call. The caller retains responsibility for
publication ordering, valid slots, and buffer lifetimes.

The function allocates two metadata vectors, each with capacity `cold.len()`. It
does not allocate expert weights or increase the 3 GiB bank budget. These CPU
allocations and the extra MLX submission still have costs that require measurement.

## Checks

- `cargo test --locked --release --no-default-features --lib cold_ready::tests
  -- --test-threads=1`: 3 passed, 0 failed, 0 ignored, 182 filtered. Log:
  `cpu-partition-tests.log`.
- The tests exercise empty/all/none/unrelated slots, maximum integer values,
  identical tuples, B2 duplicates with different activation rows, duplicate ready
  slots, and noncontiguous ready occurrences. One test visits all 1,024 readiness
  masks for 20 occurrences of 10 slots; an independent bitmask oracle checks
  membership, original order, and complete occurrence preservation.
- A disposable copy with ready membership negated failed all 3 tests, as expected.
  The production source was never mutated. Receipt: `partition-mutant.json`.
- `rustfmt --edition 2024 --check native/src/cold_ready.rs` and `git diff --check`
  passed. Python Ruff format/check and `py_compile` passed for `readiness.py`.
- Workflow YAML parsed before the final resource-safety relocation below. The
  normal Rust tests retain the helper's deterministic checks; the existing Python
  job runs readiness format, lint, compile and self-check. Async reader ASan/UBSan,
  TSan and proof-source syntax checks were added beside the existing readers.
  CI has not run remotely for these uncommitted changes.

## Kani scope

The harness calls the actual production helper with arbitrary six-element arrays
of `(usize, u32)` occurrences and ready slots, and independent symbolic lengths
from zero through six. Integer values are unrestricted. An independent nested
membership check asserts stable output contents, each output length and total
length. Unwinding bound is eight, with unwinding assertions enabled. Six covers
request empty, none-ready, all-ready, mixed, duplicate-expert and noncontiguous
cases. No stubs are used. This bounded domain is smaller than a full B2 route;
tests additionally exercise 20 occurrences.

Historical interrupted command, with Kani 0.68.0's solver directory on PATH and
one build job (retained as evidence; do not relaunch):

```sh
cargo kani --lib --no-default-features \
  --harness cold_partition_preserves_symbolic_occurrences_and_stable_order \
  --jobs 1
```

Result: **interrupted for memory pressure; not verified**. The solver was stopped
with SIGTERM and the whole owned Kani process chain joined. Last observation at
5 min 25 s of CBMC runtime: RSS 5,370,896 KiB (about 5.12 GiB), CPU 63.7%. The
planned five-minute attempt exceeded that wall limit by 27 seconds before the
parent's pressure alert and immediate termination. The solver reached 17,813
verification conditions (5,927 after simplification), but returned no assertion
or cover verdict. Its final summary says `CBMC failed with status 15` and zero
verified harnesses: this is a termination result, not a counterexample. No proof
retry or domain reduction was made. Log: `kani-partition.log`; stop receipt:
`kani-partition-interrupted.json`.

The parent observed severe system memory pressure and a large global Swapout
increase during this CPU phase. This agent captured no peak RSS or per-process
swap counter, so exact swap attribution to CBMC is **not measured**. The final
5.12 GiB CBMC RSS is sufficient evidence of material memory use, not its peak or
total demand. No further build or prover is launched by this agent.

### Resource-safety relocation after interruption

The same six-by-six harness, assertions, covers and unwind bound were moved
unchanged to `native/tests/cold_ready_proof.rs`. Its path import references the
actual `../src/cold_ready.rs` helper; no alternate implementation or stub was
introduced. It is not registered as a Cargo integration target. The explicit
CI harness selector was removed, and the ordinary `cargo kani --lib` traversal
cannot discover this standalone file. The production helper and deterministic
tests are unchanged. The original source accompanying the interrupted run is
retained in `cold-ready-before-manual-proof.rs`; the original log and interruption
receipt are retained as well.

This relocation prevents the known expensive attempt from silently rerunning
inside routine CI or another full-library Kani command. It preserves the property
and evidence without lowering the input domain or treating termination as success.
No execution, build, formatting tool, test or prover was run after this final
edit, at the parent's explicit request during the memory incident; the relocated
entry point therefore remains unexecuted.

A future manual attempt must use an **externally constrained worker** with an
enforced RAM cap (for example 4 GiB), no swap, and a process-tree wall-clock limit
of at most 300 seconds, including termination and joining all children. Check
those limits before invoking the following inner command from the repository
root; do not run it in an unrestricted local shell:

```sh
kani native/tests/cold_ready_proof.rs \
  --harness cold_partition_preserves_symbolic_occurrences_and_stable_order
```

Use the existing Kani 0.68.0 toolchain and solver directory. A resource limit,
timeout or unsupported standalone compilation must remain an explicit unverified
result; this report does not assert that the suggested cap will suffice.
The full Kani package, MLX build, integration tests and physical GPU tests belong
to the parent campaign, not this focused CPU check.

## Readiness calculation audit

The availability window for a native epoch `e` is independently expressed as
`max(0, e + last - max(e + ready, gpu_frontier))`. It is monotone in `e`, so the
lowest and highest admissible epochs yield the bounds used by `window`.
The self-check discretely enumerates possible epochs and time steps instead of
reusing the endpoint formula. All 6,592 finite cases pass; 1,660 have positive
windows and 4,932 have zero windows. Six invalid cases are rejected. These are
finite exhaustive checks, not a proof of arbitrary Python inputs. The previous
CrossHair attempt reports the module absent; no symbolic verification is claimed.

Independent recomputation checks the three source hashes and the 1,696-batch
bijection and exactly reproduces all eight recorded selection/size bounds. The
trace has 5,985 complete jobs and 7,469,351,820 bytes. Receipt:
`independent-readiness-audit.json`. No defect requiring a script edit was found.

For two first-ready jobs, availability is 9.0482244–9.052704575 ms per output in
1,071 batches. For the first two jobs in submission order, it is only
5.6734159–5.675432575 ms in 812 batches. Neither measures recoverable speedup:
publication wakeup, partition, MLX encoding, GPU execution and contention have
not been subtracted. The bounded C++ epoch and historical clock tolerance are
retained; this is not a direct measurement of NAND latency or GPU idle time.

## Numerical and integration review

Read-job order is not Cold-occurrence order: Cold may include cache hits, and a
single read job can supply multiple activation occurrences. Partition by ready
slot and retain every occurrence. Two compact Cold waves can change the temporary
layout, but concatenating Hot/W1/W2 and applying one inverse occurrence mapping
restores the existing canonical Top10 ordering **before** the existing cast,
weight multiply, reshape and sum. Summing waves separately would change floating
point association and is outside this design.

For the audited B1/B2 slot-major path, each expert row retains the same internal
dimensions, dtype, split factors and kernel family; only the number of dispatched
independent rows changes. This source inspection is not a bitwise GPU parity
result. The first prototype excludes grouped shared-EXL3 and weighted-epilogue
paths because their shape/interface choices need separate qualification. GPUHot
outputs may contain the full masked Top10 span, so remap offsets must use the
actual Hot layout rather than just its count of hits. Hot and W1 must remain live
and be drained on every later I/O failure before bank invalidation or reuse.
