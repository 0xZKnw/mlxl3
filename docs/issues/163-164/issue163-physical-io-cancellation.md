# N03: worker I/O cancellation follow-up

## Preregistration (2026-11-09, before code changes/tests)

Baseline: PR163 b28d33b (local d5d tree), existing logical cancellation and
drain ownership. Scope: `native/src/ssd_reads.rs` and its submodules. No Mac,
push, merge, dependency changes, or performance measurements.

Investigate whether macOS offers safe, bounded interruption of an already
running file read. If not, preserve that limitation explicitly. Candidate
improvement: a permanent per-batch cancellation token checked by whole,
shared-buffer, and split workers, with bounded individual read submissions.
Cancellation must prevent queued/remainder reads, survive request flag reset,
and retain every file and allocation until the active syscall returns.

Planned checks if implemented: deterministic production-worker cancellation
before submission and between bounded reads, split ownership and retry,
short-read/EINTR/EOF/offset boundaries, isolated mutation detection, focused
CPU Rust tests, strict Clippy, format, and Kani attempt. No CPU test establishes
Apple filesystem/device preemption or a wall-clock cancellation bound.

## Platform evidence

Apple's public `aio_cancel(2)` API permits `AIO_NOTCANCELED` and requires the
caller to inspect completion status. More decisively, XNU's actual
`aio_entry_try_workq_remove` only cancels work still in the queue. `_aio_close`
waits for active requests, and `do_aio_read` invokes the ordinary file-read
path. Therefore replacing Rust `pread` workers with POSIX AIO would not provide
bounded interruption of a read that is already executing. The raw-device
restriction in the manual is separate; the application reads regular files.

Sources inspected on 2026-11-09:
- https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/aio_cancel.2.html
- https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_aio.c
  (`aio_entry_try_workq_remove`, `_aio_close`, `do_aio_read`).

No forced worker termination, early file close, descriptor reuse, or early
payload reclamation is an acceptable substitute for completion ownership.

## Implementation and verification in progress

The whole/Shared/split paths now use the same bounded `read_at` loop. Each
submission requests at most 256 KiB, handles short reads and EINTR, checks the
complete offset extent, and checks cancellation before and after every syscall.
A per-batch atomic stop latch is shared by its workers and consumer. Abandon
sets that latch before detaching publication; resetting the request flag cannot
restart old work. Files, allocations, and active tickets remain owned until all
parts return. A deadline encountered by Drop also stops the remaining work.

The 256 KiB bound is a requested-byte bound, not a device-transfer, kernel
readahead, or wall-clock bound. A cancellation racing the check/submission
boundary may leave one bounded syscall per worker executing. No active syscall
is killed. No throughput or Mac cancellation latency has been measured.

Callers reviewed: ReadService enqueue (demand/background), ReadWindow refill,
Read::run (Vec and Metal Shared), SplitRecord::run_part (two disjoint ranges),
and ReadBatch abandon/check/drop. The payload and lease ownership protocol is
unchanged. Apple-only Shared paths still require Apple compilation/testing.

Initial focused compilation failed because the new sibling regression test
could not access the private existing lease fixture. Its visibility was widened
to the parent module only; no production behavior changed. Next attempt hit
concurrent draft-GPU test module cfg errors, reported to its owner. Neither
attempt ran any SSD test. Original diagnostics retained in /tmp/pr163-io-tests*.

Formal-proof skill and language reference are absent from the catalog and
selected environment. Kani 0.68.0 has now been installed by the parallel proof
worker; a focused attempt is planned. The new harness covers the actual
submission-size helper, not filesystem or concurrent cancellation semantics.

First focused run after compile repairs: 28 SSD tests passed, none failed or
ignored. The production regression was then expanded to prefetch polling as a
fourth mode, in addition to static, dynamic, and split demand reads. A Clippy
collapsible-if style diagnostic in the test probe was corrected.

Mutation preregistration: compile an isolated copy of the actual read-loop
module, changing the persistent stopped-token check to the resettable request
flag. The sticky-cancellation test must fail after flag reset, then the intact
module must pass. A compile-only no-op test gate supports this isolated wrapper;
neither read-loop test exercises that hook, and production worker behavior is
validated separately by the full package tests. No production checkout mutation.

## Final local results

Environment: cloud Linux, Rust 1.99.1, no Apple hardware/model. Cargo commands
used `CARGO_TARGET_DIR=/tmp/pr163-native-target`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`.

- `cargo test --locked --no-default-features --features chat --lib ssd_reads -- --nocapture`:
  final **28 passed, 0 failed, 0 ignored**, 0.17 s. Production cancellation
  regression covers eight static/dynamic/split/prefetch scenarios, before the
  first syscall and after exactly one chunk, checks no further submissions
  after request reset, retained ownership/admission refusal, and byte-exact
  successful retry. Existing window, EOF, split completion, partial enqueue,
  deadline, and owner-reuse tests also pass. Independent loop tests exercise
  short reads, EINTR, offset overflow, EOF, invalid counts and ordinary errors.
- `cargo clippy --locked --all-targets --no-default-features --features chat -- -D warnings`:
  **passed**. `rustfmt --edition 2024 --check native/src/ssd_reads.rs` and
  `git diff --check`: **passed**.
- Isolated sticky-token mutant: **detected**, exit 101, 1 failed/1 passed.
  Intact module: **2 passed**, exit 0. Production checkout never mutated.
- `cargo kani --lib --no-default-features --harness ssd_reads::cancellable_io::ssd_read_submission_is_bounded_and_makes_progress --output-format terse`:
  Kani 0.68.0/CBMC 6.11.0 **passed**, 0 of 16 failed and 3 of 3 covers, 0.080 s
  verification. Symbolic full `usize` domain, no assumptions: chunk is at most
  256 KiB and at most remaining bytes, zero iff no remainder, and positive
  submissions decrease the remainder. This proves only the real arithmetic
  helper, not the OS, allocator, atomics, syscall completion or worker lifetime.

Logs and exact source SHA-256 values are in
`docs/measurements/pr163-io-cancellation/`; raw originals are retained in JSON
alongside whitespace-normalized readable logs. Earlier compilation failures
remain preserved rather than being relabeled as passing tests.

Existing `.github/workflows/rust.yml` selects portable tests on Ubuntu and
macOS-26 and the Kani manifest shards. The parent/proof worker integrates the
new harness manifest and runs aggregate checks separately. No hosted exact-head
CI success is inferred from these local results. Metal Shared paths, real
device cancellation, performance and full application behavior remain untested.

Status: scoped worker cancellation **implemented and CPU-validated locally**;
all worker-owned checks completed. Active-syscall physical preemption remains
unsupported by the verified macOS AIO path and is not claimed fixed. This is
stronger than dropping results: remaining physical reads are no longer issued,
but an in-progress kernel read still finishes before its owner can be reclaimed.
