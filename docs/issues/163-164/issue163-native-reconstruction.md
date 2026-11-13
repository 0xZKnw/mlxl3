# Issue 162 native boundary reconstruction

## Preregistration (2026-11-09, before changes/tests)

Baseline: local f428945b, tree of published PR163 2ddde23, including main
0fd9fbd1. Previous lost local work is not validation. Scope: N01, N06, N07,
N08. No physical Mac, GPU, model or performance measurements are planned.

- N01: Session::advance and its native bridge caller must receive EOS IDs.
  Extract the actual serial verification loop into a CPU-testable helper.
  Stop on accepted/rejected/bonus EOS; count accepted proposals explicitly
  (rows minus one is false for accepted EOS). Captures represent consumed
  inputs, terminal output stays pending, and width respects remaining output
  and context budgets. Deterministic independent AR sequences cover EOS in
  each position, rejection, bonus, budgets 1/2/3 and a forbidden post-EOS
  forward fault. Attempt symbolic Kani for the production helper.
- N06: decode_image/prepare_messages must inspect decoder dimensions and
  orientation, calculate resized pixels and tokens, and admit against the
  remaining aggregate budget before image decode or patch allocation.
  Patch reservation becomes fallible. Test actual decoder/production entry
  with malformed/compressed images, exhausted aggregate budget, eight small
  images, orientation and extreme aspect ratios; preserve existing patch
  normalization oracle. No RSS claim.
- N07: both native controllers receive the same elapsed cycle including
  draft, verification and cache commit. Component timings remain separate.
  Add deterministic injected-duration controller regression where expensive
  commit reverses the partial-cost ranking; retain reset/error coverage.
- N08: transaction covers Qwen run plus fallible trace export. Any error
  restores the pre-call snapshot including offset, KV and GDN. Exercise the
  production transaction helper with export failure/retry and retain an
  explicitly ignored physical joined/nonjoined model regression.

Validation: inspect actual Cargo features, targeted CPU tests, nontrivial
mutations where practical, fmt and strict Clippy; parent runs aggregate
gates. Attempt cargo kani on actual helpers; unavailable tooling is recorded
as unverified. Metal/MLX compilation and physical numerical state parity
remain unverified on Linux. No push, merge, release or default promotion.

## Results

Implemented locally, not published by this worker:

- `speculative::verify_serial` is the real native verification loop. Its CPU
  oracle enumerates budgets 1–4, terminal positions 1–4 and each rejection
  position. It asserts exact AR outputs, consumed inputs, accepted counts and
  no call after terminal; faults propagate. Native captures/cache commit use
  exactly those consumed rows. EOS enters through the bridge tokenizer.
- `vision_input` admits the oriented/resized decoder dimensions against both
  remaining pixels and image tokens before decoding/RGB/resizing/patches.
  The maximum remains 16,777,216 pixels / 16,384 image tokens across messages;
  this is not a newly claimed language-model context limit. Patch storage
  uses `try_reserve_exact`. Tests count patch-allocation attempts and cover
  pixel/token boundaries, eight images, a compressed 4096² image, EXIF,
  invalid inputs/extreme ratios and the independent normalization oracle.
- Native adaptive and pre-draft observations now use the same elapsed cycle
  after cache commit; component timers remain diagnostic. Accepted EOS is
  counted as accepted, and no unverified confidence receives a reject label.
  A synthetic duration oracle distinguishes full-cycle vs partial rankings
  and rejects invalid controller observations without changing learned sums.
- Qwen uses `transaction::with_rollback` across run and trace export. CPU
  tests cover run failure, export failure and successful retry. A separate
  ignored physical test injects export failure for joined/nonjoined Qwen,
  compares offset/KV/GDN trace bytes and retry/continuation logits. It has
  not run, and CPU transaction tests do not validate GPU storage semantics.

Validation so far: targeted serial 1, vision 5 (before the compressed fixture
addition), controller 1, transaction 1 passed. Mutating away the EOS check
failed on a post-terminal forward assertion; mutating away restoration failed
on changed state. Both were restored and rerun green. First mutation attempt
hit a compiler interruption before tests; its diagnostic tail did not establish
a cause, so that attempt is inconclusive, not mutation validation. First test
compile also found a test-only Debug bound from `unwrap_err`; fixed without
changing PreparedImage. All subsequent focused builds succeeded.

Kani attempted with `cargo kani --lib --no-default-features --harness
serial_verification_never_consumes_eos`: unavailable (`no such command:
kani`). Harnesses retained for serial EOS, rollback and image aggregate
arithmetic (now in ungated image_budget for the repository runner). No proof claimed. The requested formal-proof
skill/language reference is absent in the available catalog and `.agents`;
repository verification requirements were followed directly. No Mac, MLX,
Metal build, physical fault injection, model parity or speed result obtained.

Expanded vision tests: 6 passed, including compressed4096², in 0.60s.
Moving admission after patch allocation was detected by the allocation counter
(1 instead of 0); restored vision suite6 passed in0.75s. Strict portable
all-target Clippy reached only three diagnostics in the concurrently edited
MCP module (needless borrows102/106, collapsible if606); reported to its owner.

## P05 follow-up preregistration (before edit)

Current native Draft::forward clones all five hidden matrices even when
proposals discards the trace. Preserve the public diagnostic forward contract,
add a private trace switch, and use trace-off from proposals. Required target
capture layers remain unchanged. Extend the existing ignored independent
NumPy oracle to compare trace-on/off output bitwise and require no diagnostic
rows in trace-off. Source/format checks on Linux cannot validate Accelerate
or model output, and no latency or memory measurement is planned.

Final four-fix portable strict Clippy passed (7.20s); scoped formatting and
diff check passed. P05 trace switch implemented; physical oracle still unrun.

## N03 follow-up preregistration (before implementation)

The current main still blocks ReadBatch::ready_min and its Drop until the
30-second batch deadline. Existing workers own the file and destination and
already retain the active counter through physical completion. Reuse those
owners, attach the bridge SIGUSR1 Arc flag to demand/prefetch readers, poll at
10ms intervals and permanently abandon a cancelled batch receiver. Resetting
the shared flag must not resurrect that batch or publish its stale leases.
Drop must not wait after logical cancellation; actual SSD syscalls may remain
uninterruptible. Admission must continue refusing work while old workers are
active. Prefetch must invalidate all pending leases and discard queued results
on cancellation. Native DFlash checks between target forwards under rollback.

CPU tests: held worker gate, cancel during wait, drop/ack before gate release,
clear flag before worker finishes, reject retry while active, release/join drain,
then successful new batch with no stale publication. Include already-queued
completion cancelled before consumption. Attempt Kani where the actual state
machine permits, keep hardware claims separate. Physical Metal lease/cache and
bridge Stop latency tests remain unavailable; parent coordinates GPU file edits.

N03 implementation: production Metal owns the same cancellation Arc registered
with SIGUSR1 by direct_bridge. Read services and each submitted batch retain it.
A batch permanently latches observed cancellation, drops its receiver and
marks its service draining; resetting the shared flag cannot revive publication
or admit any new demand/background work while active workers remain. Workers
retain destinations/files and decrement active only on physical completion.
ReadWindow shares the same state. Prefetch cancellation invalidates pending
leases, including Flash.reset before the bridge clears the flag. DFlash checks
between verification forwards and before cache commit, restoring target on
failure. No arbitrary Metal command or SSD syscall is force-interrupted.

SSD suite24 passed after the implementation. The cancellation test uses a
held physical worker and a separate consumer, acknowledges/drop before release,
clears the flag, rejects both admission modes, then drains and retries. Queued
completion test proves permanent abandonment after flag reset. Test allowance
250ms is scheduler tolerance, not a measured bridge guarantee; receive polls
are capped at10ms. Deterministic wait-entry synchronization makes the regression
non-sleep-based. Replacing the short receive slice with the original full
deadline failed the acknowledgement assertion in0.25s; workers were released
and the waiter joined. Restored SSD24 passed in0.12s.
The actual latch Kani harness is retained, but the command is unavailable.
After the other worker resolved draft_storage diagnostics, strict portable
all-target Clippy passed4.68s. Vision6, image-budget oracle1 and serial oracle1
also passed after the ungated helper extraction. Physical integration/state
parity remains unrun.

Final review found that the existing ReadWindow destructor left window_active
set permanently when a cancelled worker was still active. The corrected Drop
releases logical ownership while retaining the draining admission latch until
active reaches zero. A held-worker regression now cancels, drops, clears the
flag, verifies admission refusal while active, releases the worker and starts
a fresh successful window. This extends the batch recovery test to the PR161
window path. Final SSD suite25 passed0.26s; strict portable all-target Clippy
passed3.16s. No remaining process or source mutation from this worker.

## N09 implementation plan, not claimed fixed

Current public MatrixView::read_output accepts any contiguous Shared view.
Public encode_model_rms returns such a view before submission. There are about
70 read_output calls, 70 MatrixView::output constructors and 15 direct view
construction sites across the GPU modules/CLI. Reducing visibility alone
breaks the separate binary crate without establishing producer ownership.

A follow-up needs two distinct capabilities: an encode-side pending view tied
to its producing command/queue, and a completed immutable host-readable result.
The submit/wait boundary must consume the pending producer and create the
completed capability only after successful command completion and device
identity validation. Rows/slices must preserve the capability; same-queue
binding remains available before completion and must not add waits. Uploaded
initialized input storage has a separate host-ready constructor. Migrate public
encoder results and high-level output owners, then CLI diagnostics; do not
accept an arbitrary unrelated completed command as proof for a view. Tests
must cover early reads, failed/foreign producers, alias lifetime and completed
row slicing, plus compile-fail API tests and existing numerical GPU oracles.
This refactor is intentionally not mixed blindly into the current coordinated
pipeline/Qwen changes. N09 remains open until implemented and qualified.
