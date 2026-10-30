# Exact-head follow-up and compiler logging repair

The previous head `1f918af9f242b116b73da34de851a5939780b6e5` completed
[Native CI 41062436174](https://github.com/0xZKnw/mlxl3/actions/runs/41062436174)
with failure. This remains a partial repair in draft PR63.

| Group | Exact-head result |
| --- | --- |
| Allocator | 4/4 succeeded |
| Core | 73 successful selectors, including the two API instances; sidecar collection timed out at 600s |
| Speculative | 197/198 succeeded; the remaining selector hit a compiler ICE before CBMC |
| Flash | 29 succeeded; two grouping proofs timed out at 600s; failed-fill verification returned backend ERROR; parser ran out of the shard's remaining 344s; eleven later selectors had no verdict |
| Native Ubuntu | Passed |
| Native macOS | Cancelled at the existing 20-minute limit during EXL3 lane checks, after the Rust compilation/test/lint steps passed |
| Conversion | Passed, run41062436199 |
| Desktop regression | Failed at the existing `StudioModesCombined.swift:2250` helper precondition, exit133, run41062436194 |

## Compiler crash

The complete local reproduction also produced 197 successes and one compiler
crash in `tuning_never_selects_missing_or_slower_candidate`. Adding `--verbose`
had enabled INFO statistics. Kani 0.68 calls
[print_stats only at INFO level](https://github.com/model-checking/kani/blob/kani-0.68.0/kani-compiler/src/codegen_cprover_gotoc/compiler_interface.rs#L737),
and its [statistics visitor unwraps an optional MIR body](https://github.com/model-checking/kani/blob/kani-0.68.0/kani-compiler/src/kani_middle/analysis.rs#L26).
The failure is in that reporting path. The same remote ICE was confirmed once
the completed run's logs became available.

Removing `--verbose` preserves regular per-check output, all model-checking
options, assertions, domains and deadlines. The previously failing real proof
then passed 395 checks, zero failed, six unreachable assertions and 2/2 covers
in 130.177 s. A CLI regression emulates this specific Kani 0.68 failure and requires
all 198 speculative selectors to succeed. The complete orchestration suite has
34 passing tests. The old failed local report is retained; it is not relabelled
as a green full-shard run.

## Fill proof and job distribution

The fill harness now initializes the same three symbolic 24-entry arrays before
placing the cache inside `Rc<RefCell<_>>`. The complete boolean domain, the real
`ExpertFill` drop/commit, all field assertions, both covers and unwind 27 remain.
The unchanged input state is now easier for the prover to track across borrows.
The complete original harness reported 1247 checks, zero failed, 12 unreachable
assertions and 2/2 covers in 132.293 s in the isolated trial, then 97.515 s in the final checkout. No production cache implementation changed.
The two constant-case pilots are preserved as intermediate evidence, not added
to the required inventory. A pending-clearing mutation is checked against the
existing production unit regression and then restored.

Grouped-route proofs now have a separate logical shard. There are five logical
groups and at most four simultaneous workers: one Flash job plus a matrix with
`max-parallel: 3`. This prevents the two grouping attempts that consumed twenty minutes together from consuming
the remaining Flash budget. The aggregate still requires every group. The
compiled required inventory remains 319 library harnesses plus both API instances.
Every discovered harness is assigned once. Limits stay 600 s per selector,
2700 s per group and 6 GiB per inherited process.

## Rejected work and remaining limits

All original byte/ID/width domains were retained in the promoted proof changes.
No borrowed-map, reserved-vector or stack-buffer production prototype qualified.
The sidecar proof still failed with bounded internal-loop encoding; a measured
10 GiB experiment still timed out at 600.002 s (sampled peak RSS 7922200 KiB).
CI memory limits were therefore not raised. New fixed-buffer parser experiments
failed and are not promoted. Their 65536-suffix construction test passed, but
that does not turn their failed Kani run into a proof.

The remaining sidecar, grouping and parser proofs need further work. No such
obligation is excluded or marked successful. The next exact-head CI must supply
the final verdict, including the previously unvisited Flash selectors. CPU
proofs do not establish Metal numerics or an all-generation Mac hardware matrix.

PR61/62 branches remain untouched. When integrating them, resolve the changes
to `rust.yml` by keeping their additional native checks and deployment target;
the dynamic inventory will automatically select their new standard proofs.
PR62's standalone 15-minute guard workflow remains separate and incomplete;
it must not silently substitute for the complete inventory or hide its GDN and
attention obligations. The known Desktop assertion is addressed in PR62.

Completed-run logs became readable after earlier connector Internal errors.
The Flash artifact was fetched as a file reference, but Library's official
materialization helper returned `download failed`; no local ZIP was installed
and no alternate network route was attempted. Per-selector job logs, local
reports, command receipts and negative source patches are retained in the
evidence archives and opti.md. No cancellation, merge or deployment was made.
