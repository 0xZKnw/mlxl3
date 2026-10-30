# Kani CI investigation — 2026-10-30

Status: **implementation prepared; full-suite and exact-head CI validation in progress**.

Base: `f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6` (remote main checked).
Local branch: `fix/kani-ci-stability`. PR61/62 branches were not modified.

## Evidence

- Latest fully successful Native Rust checks run in the complete successful-run
  listing: [38064605338](https://github.com/0xZKnw/mlxl3/actions/runs/38064605338),
  head `260cbf42cd4b5de0833ace8ee3f0f1ec7299a368`, 2026-10-10. Kani job
  `114249593414` succeeded in 17m20s: 52/52 harnesses verified, Kani0.68.0 /
  CBMC6.11.0 (same versions as the recent failed runs). See
  `last-green-global.log.gz`. This establishes a historical success,
  not success of the present set of harnesses.
- Current main run [40538121772](https://github.com/0xZKnw/mlxl3/actions/runs/40538121772),
  Kani job `122308819173`: API proof succeeded, library proof step cancelled;
  Flash Kani `122308819535`, Linux and macOS succeeded.
- PR61 run [41030107141](https://github.com/0xZKnw/mlxl3/actions/runs/41030107141),
  job `123977679734`: Kani0.68.0, CBMC6.11.0. The allocator harness starts at
  09:04:30.1655125Z, explicit runner shutdown at09:06:20.0791152Z (109.914s).
  The configured 60-minute job budget was not reached at the shutdown timestamp.
  Flash job `123977679492` succeeded separately. The library step had completed
  45 harnesses successfully and started its46th at the interruption; the API
  step had also completed its two compiled instances successfully. See `pr61-global.log.gz`.
- PR62 run [41038402005](https://github.com/0xZKnw/mlxl3/actions/runs/41038402005),
  global job `124005223598`: allocator starts10:13:23.1159264Z, runner shutdown
  10:15:18.0927015Z (114.977s), followed by cancelled operation. See
  `pr62-global.log.gz`. Flash job `124005224167` is cancelled, not passed.
- Earlier identical shutdowns while proving this harness are already recorded
  in opti.md (60,63,64D,70M,72/73 and PR47). The allocator was introduced in
  `a620ac54` on2026-10-20, so it cannot explain every earlier failure since
  the last green run on10October. `engine-v1.4.8/ci-kani-diagnosis.json` records
  an earlier incomplete DFlash length17 proof with SIGTERM of unknown origin.

There is **no established counterexample, OOM, proof timeout or sender of the
shutdown signal** in these logs. A resource problem is a hypothesis to reproduce,
not a fact. Workflow metadata completion time can lag the last runner log and
must not be treated as the prover's elapsed runtime.

## Workflow and proof design observations

`.github/workflows/rust.yml` runs the entire library serially (`--jobs 1`)
inside one60-minute job; an expensive early harness prevents later verdicts.
The separate Flash action repeats proofs also selected by the global library
action. Flash has a30-minute budget and has succeeded in about20minutes.
The global job invokes the Kani install action twice, including installation of
stable Rust and `cargo install --version0.68.0 --locked kani-verifier` each time;
installation succeeds in the inspected runs. No evidence supports blaming a
Kani/CBMC version mismatch. Workflow concurrency cancels superseded runs on the
same branch; that is a distinct explanation for some cancelled runs, not a
demonstrated explanation for the runner shutdowns above.

The allocator harness keeps four unconstrained `u32` costs and an unconstrained
`usize` budget. The real implementation rejects budget>2 for its two-layer,
two-level fixture, then builds vectors and dynamic-programming loops sized by
the symbolic budget. A candidate reformulation is separate fixed-budget0/1/2
proofs plus rejection for all invalid symbolic budgets, preserving every cost
value, all meaningful branches, the unwinding assertions and original oracle.
This candidate has **not** been implemented or measured.

A subsequent orchestration fix should enumerate the compiled Kani harnesses,
assign every one exactly once to a small number of explicit shards, reject empty
or missing selections, run with finite per-harness time/resource budgets and
save each verdict even after a failed proof. Timeout, signal, missing output or
counterexample must fail the aggregate. Install/cache changes should follow
measurement rather than hide the expensive harness. PR62's additional standalone
compatibility workflow would still need its own explicit integration and verdict;
this branch must not silently alter that draft PR.

## Initial blocker (resolved by explicit user authorization)

AGENTS.md requires: “Pour chaque changement de code exécutable, charger
`$formal-proof-skill` et la référence du langage concerné.” No such skill or
Rust/Python reference is present under `/workspace/.agents`, repository
`.agents/skills`, the available executor skill catalog, or the cloud skill
catalog inspected. The local `.agents` directory is empty and read-only.
No executable file, workflow, harness or test has been changed. A request for
the skill's location is pending. The exact remaining action is to obtain/read
that skill (or receive an explicit instruction overriding this repository
requirement), then implement, reproduce and prove the repair.

The managed image also has no cargo, rustup or Kani on PATH. No local proof was
started, and no local tooling was installed. The180s/6GiB baseline experiment
preregistered in opti.md remains **not launched**. Network policy is enforced;
Git fetch works. Direct `gh run list` was denied with HTTP403 on api.github.com;
the authorized GitHub connector successfully supplied runs, jobs and logs.
No cancellation, permission change, retry of the previously denied force-cancel,
push, PR, merge or deployment was attempted.

Those statements describe the initial investigation before resumption. Justin
explicitly authorized bypassing only the missing skill prerequisite for this
task (“vas y”). AGENTS.md itself remains unchanged. No remote cancellation or
permission change has been attempted.

## Reproduction and implemented repair

The original harness was reproduced under a6GiB address-space limit and180s
deadline using Kani0.68.0/CBMC6.11.0. It failed after58.11s with maxRSS5551224KiB.
The diagnostic repeat explicitly reports **“Solver ran out of memory during
propositional reduction”**, with ERROR obligations rather than a counterexample
(`allocator-diagnostic-v2.log.gz`, uncompressed line746). This establishes local resource
exhaustion. The exact host-level cause of historical CI shutdown signals remains
unknown; the reproduction does not identify who stopped those runners.

The single two-layer/two-level proof is replaced by an exhaustive partition:

| Budget domain | Property and implementation | Unwind | Targeted result |
| --- | --- | --- | --- |
| 0 | Real allocator, exact sum, bounds, independent minimum-cost oracle | 5 | Pass, 1/1 cover |
| 1 | Same, all four unconstrained u32 costs | 5 | Pass, 1/1 cover |
| 2 | Same, all four unconstrained u32 costs | 5 | Pass, 1/1 cover |
| Every usize >2 | Real entry guard rejects without entering allocation body | 3 | Pass, 2/2 covers (3 and usize::MAX) |

The three valid cases call the actual algorithm without a stub. For the invalid
case only, the unchanged DP body is extracted into a private helper and replaced
with an unconditional panic. This is a conservative **unreachability** proof:
any call to the body fails the proof, so a passing proof establishes rejection
before that call for every invalid budget. It assumes no result of the body.
All original inputs and useful obligations are retained; unwinding assertions
remain enabled. No assumption on the symbolic costs or budget was added.
The wrapper and helper remain in the existing test/Kani-only module.

The first three formulations that still encoded the unreachable body exhausted
the6GiB bound. Their failures are retained. The final targeted run passed4/4
proofs with5/5 covers in about23s of verification. The14/11/11 unreachable
obligations in the valid cases and2 in the invalid case are retained in logs;
these are not missing verdicts. Temporarily removing the real budget guard and
reversing the real minimum-cost comparator each produce FAILURE/counterexamples.
Both mutations were restored; hashes and commands are in `mutation-results.json`.

`scripts/kani_ci.py` discovers compiled harnesses and checks inventory schema,
version, counts and uniqueness before selecting any proof. Four fixed shards
cover every library harness exactly once; unknown future modules go to core.
The API proof retains both compiled test instances. The original140 library
harnesses become143 solely because the allocator obligation is split into four.
Only the documented panicking stub is accepted by the verdict parser; every
other proof must be unstubbed. Kani's stubbing feature is enabled explicitly.

Each proof has600s and6GiB of address space; each CI shard has a45-minute total
budget including discovery. Timeout, signal, nonzero exit, missing/incomplete
output, zero proofs or unsatisfied cover fail the aggregate. Remaining proofs
are attempted after individual failures. The atomically updated report starts
incomplete, records pending/running statuses, and only becomes successful when
every planned proof has a checked verdict. Process groups are terminated and
the direct child reaped on timeout/interruption. Logs and inventory are uploaded
even after ordinary failures. The existing `kani` aggregate and
`kani-flash-contracts` check names remain; the aggregate requires all shards.
No workflow permissions were changed. The Kani action is pinned to its inspected
commit, Kani/CBMC versions are unchanged, and each runner installs Kani once.

## Validation at implementation review

- 32 deterministic orchestration regressions pass, including real child
  processes, silent timeout, signals, descendant cleanup, stale/empty inventory,
  partial output, API instance counts, failed aggregate and continued execution.
- Rust CPU suite:238 passed,5 ignored; strict Clippy and format checks pass.
- Actionlint1.7.12 and Ruff0.16.5 pass.
- CrossHair0.0.101 (Z3 5.3.3) attempted the shard postcondition for10s and reports
  “Not confirmed”. No formal Python proof or concurrency/GPU guarantee is claimed.
- Full compiled-library/API run and remote CI are still pending at this stage.
  No success is inferred from the targeted proof results.
