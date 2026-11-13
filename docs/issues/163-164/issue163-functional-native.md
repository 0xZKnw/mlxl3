# Issue 162: terminal SSD batch failure recovery

## Preregistration, 2026-11-09

Baseline: `8080c96c`, isolated cloud checkout. Parent entry:
`PR163-FUNCTIONAL-8080-20261109` in opti.md. Read AGENTS.md, the existing native,
Metal retirement and physical I/O cancellation reports, and production callers.
No CI edits, Mac, model, GPU, benchmark, installed app or remote write.

Concrete remaining defect: explicit request cancellation abandons a batch,
but payload errors, failed partial enqueue, receive failure and deadline errors
do not. After an EOF a caller's Drop can wait for another worker until the
30-second batch deadline. A retained failed batch can also publish a later
successful completion. These are terminal batch errors, not retryable polls.

Plan: use the existing owner-preserving abandon path on terminal enqueue and
completion errors, keeping the original error as the first result. Do not
abandon an empty/nonblocking successful poll or a caller's invalid ready count.
The worker still owns its File, destination and active Ticket. Shared/split
allocations remain held until every physical reader returns. No GPU command
retirement change and no forcibly interrupted active OS read.

Before production edits, add regressions through real ReadService/ReadBatch:
held worker plus early EOF must return/drop without waiting, late publication
must remain impossible, failed-batch admission stays blocked until drain, then
a fresh batch succeeds. Include retained timeout/poll, partial enqueue failure,
queued result/receiver disconnection, successful poll/invalid argument controls.
Use bounded gates and always release/join before assertions. Reproduce against
baseline, then run affected tests, SSD suite, strict portable Clippy, fmt/diff.
Add/attempt actual-source bounded Kani state transition checks. Formal-proof
skill/Rust reference unavailable in executor catalog and local paths; no proof
claim follows from their absence. Preserve diagnostics and verification limits.

## Reproduction and first verification

Before production edits, four added tests failed on baseline in 0.26s. The
observed failures distinguish held-worker cleanup, partial enqueue cleanup,
retained timeout state, and queued-success publication after failure. Every
held worker was released and the consumer joined before failure assertions.
Log: `docs/measurements/pr163-functional-native/before.log`.

First patched SSD run: 32 passed, 1 failed. The older wave/deadline fixture
explicitly expected to continue consuming a batch after timeout. Updated it
to require terminal detachment and moved its independent EOF diagnostic check
to a fresh batch; this is the intended contract change, not an ignored test.

## Final verification

- Portable SSD suite: **33 passed, 0 failed, 0 ignored**, 0.20s. Five added
  regressions exercise the actual batch/service and worker entry points.
- No-default-feature strict all-target Clippy initially stopped on a concurrent
  unrelated `structured.rs` test's `err_expect` lint. No native diagnostic was
  emitted; the coordinator owns aggregate rerun after that correction.
- Kani 0.68.0 / CBMC 6.11.0: two targeted actual-source harnesses pass,
  **99 obligations, 4/4 covers**. New harness uses a symbolic bool and unrestricted
  u32 payload/error, with no assumptions or loops; it checks exact result
  preservation and failure-only stop callback before return, then checks the
  irreversible latch. The existing harness checks initial/requested booleans
  and reset. This proves the bounded synchronous helpers, not channels, worker
  concurrency, filesystem ownership or Metal. Initial compilation emitted existing
  future-compatibility/dead-code warnings; not a strict Kani compilation claim.
  Repeat the two quick harnesses with regular output solely to inspect individual
  obligations/unreachable status, not to change their domains or outcome.
- Scoped rustfmt and full whitespace diff check pass. Sources frozen for the
  coordinator's final aggregate review; no source mutation remains.

Commands use the existing `/tmp/pr163-native-target`, debug info zero and
incremental compilation disabled:

```
cargo test --locked --no-default-features --lib ssd_reads -- --nocapture
cargo clippy --locked --no-default-features --all-targets -- -D warnings
timeout 120s cargo kani --lib --no-default-features --jobs 1 \
  --harness terminal_batch_error_stops_before_return_and_preserves_the_error \
  --harness batch_cancellation_is_irreversible_across_flag_reset
rustfmt --edition 2024 --check native/src/ssd_reads.rs
git diff --check
```

## Integration and remaining limits

`ReadService::start_fallible_payloads` and `ReadWindow::refill` share the enqueue
error boundary. `ReadBatch::ready_min` is used by demand/block/window reads;
`ReadBatch::poll` is used by prefetch. Each terminal error invokes the existing
abandon path before returning the original error. Successful empty polls and
invalid caller minimums leave live work intact. Repeated calls after a failure
report that the batch is cancelled/abandoned instead of publishing stale data.
The existing wave fixture now treats timeout as terminal and tests independent
EOF diagnostics in a new batch. Old EOF retry tests explicitly wait for physical
drain rather than conflating logical batch termination with worker completion.

No worker completion, Shared allocation, split allocation, Ticket, GPU producer,
or GPU retirement code was changed. `Read::run` retains its file/payload/ticket
through the syscall; `SplitRecord` retains the entire backing until both parts
finish. Admission remains blocked while active workers own those resources.
The 250ms fixture allowance is scheduler tolerance, not a production latency
bound. Already executing OS reads are never forcibly interrupted.

N01 serial verification still stops at accepted/rejected/bonus EOS; N07 observes
full-cycle cost after commit; N08 retains the run/export rollback boundary;
N02 retains all-owner command retirement plus monotonic producer poisoning.
No additional demonstrated defect was changed in those areas. Linux portable
coverage does not qualify Apple Metal/Shared compilation, physical GPU failure
recovery, live-model continuation or measured user Stop latency. Those existing
qualification limits remain. No performance claim, CI modification, or blanket
closure of issue 162 is made here.

Individual Kani obligations were inspected in `kani-detail.log`: all 99 are
SUCCESS, all four covers SATISFIED, no failed or unreachable obligation was
reported. The second run kept identical source and domains. Both processes
finished, leaving no active verifier. A second strict portable all-target Clippy
attempt still encountered the same unrelated `structured.rs:513` `err_expect`
lint: the concurrent correction was not yet present at compilation. Both attempts
failed, with no native diagnostic. The coordinator's post-freeze aggregate must
establish the final lint result. Source hashes and both negative runs are retained
in `docs/measurements/pr163-functional-native/`.

Final lint result after verifying the structured correction was present:
`cargo clippy --locked --no-default-features --all-targets -- -D warnings`
**passed**, exit 0, 15.30s including build-directory lock wait. The prior two
failed logs remain. Evidence: `clippy-after-structured-correction.log`.
All worker-owned checks are complete; coordinator still performs combined
chat-feature/other-component aggregate checks on the final assembled source.

## Integrated evidence delivery

Raw logs referenced above are retained inside
[functional evidence archive](../../measurements/pr163-functional/pr163-functional-evidence.tar.gz),
under the corresponding `pr163-functional-*` directory. Temporary structured
logs were copied into `pr163-functional-structured` before archiving. Coordinator
integration on base `8080c96c`: 420 Rust library tests passed, 10 ignored;
format and the 314-file source inventory passed. Hosted CI and physical Apple
qualification are separate and not claimed here.
The coordinator widened only fixture scheduling margins after review: ACK one
second (batch deadline two seconds), drain five seconds. Gates remain held until
after observation, preserving the before-release oracle. All 33 SSD tests passed
again; production deadlines and the proven helpers were not changed.
