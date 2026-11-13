# Issue 162 N02: Metal failure retirement and diagnostics

## Source contract

Read AGENTS.md, opti.md and existing retirement/history before edits. This work
extends merged PR161 retirement rather than claiming its existing helper as new.
Preregistration: PR163-N02-COMPLETE-20261109 in opti.md. Formal-proof skill and
its Rust reference were unavailable in the inspected catalog/local skills.
No user Mac, push, commit, benchmark or model execution was used.

Two concrete error-path defects were addressed:

- `FlashMoeRow::forward_routed` used `self.last.drain(..)` and returned at the
  first non-completed command. It now uses all-owner retirement. A completed
  command incurs no new wait. Nonterminal/failed commands are waited before
  releasing command tracking, and all commands are visited after a failure.
- `FlashMoeRow::drain` returned on prefetch/SSD failure before GPU retirement.
  It now evaluates all cleanup results before returning the first error.
  Pending cancelled SSD reads retain their own backing ownership, and a
  non-idle read check still prevents shrink/recycling.

`CommandHealth` is a monotonic, Rc-shared failure latch used by the actual row
and its AR/block outputs. Observed command or Hot producer failure prevents
future row execution, cache reuse/shrink, gate encoding and output publication.
An empty successful second drain cannot clear it. A fresh model load is required;
there is no unsafe attempt to recycle possibly incomplete cache data. Already
submitted commands are retired even when the latch was previously set.
Recoverable CPU cancellation/read failures do not alone poison GPU health.

`gpu/command_status.rs` is the common failure-only diagnostic boundary used by
central Metal command/dispatch, router, trellis quantization and MoE output/drain.
It does not wait. Completed status returns immediately without NSError access.
Failure reads the nullable NSError with explicit Objective-C return types and
copies domain, signed code and localized description while the command owns it.
Missing error/string pointers have explicit fallback text. Operation names and,
for MoE drain, the known checkpoint layer prefix are included. No success logging
or global synchronization was introduced by this diagnostic helper.

Existing ignored physical Hot-producer ownership test now also checks the shared
poison latch. New ignored physical command-status fixture checks successful
command completion and missing-NSError diagnostics. Neither fixture manufactures
a real GPU fault or claims to prove device fault recovery.

## Verification and evidence

Directory: `docs/measurements/pr163-metal-recovery/`.

- `cargo test --locked --no-default-features --lib gpu_wait_policy`: 4 passed,
  0 failed, 0 ignored, both initial and final run. Existing all-owner lifetime
  and multi-failure retirement tests retained; new production latch test injects
  each of four failed owners, checks every wait, verifies empty retry remains
  poisoned and fresh load starts healthy.
- Isolated source mutation changed the monotonic latch to the latest result.
  New regression detects it (exit101, failed assertion on successful empty
  retry). Shared production source was never mutated; final normal tests pass.
- Actual-source Kani harness uses two symbolic booleans without assumptions,
  asserting monotonic failure and covering success/failure-then-success. Attempt
  `cargo kani --lib --no-default-features --harness
  gpu_wait_policy::failed_command_health_cannot_be_cleared_by_success` fails
  before verification: no such command `kani` (exit101). Zero properties proved.
- Scoped rustfmt and full `git diff --check` pass. Parent runs aggregate Rust
  tests/strict Clippy after source freeze. Existing Apple compatibility CI builds
  the actual direct-Metal variant and is required to validate these ObjC calls.

Not executed here: macOS compilation, NSError live fault values, physical GPU
fixtures, real-model continuation after a driver fault, throughput/RSS or Metal
lifetime validation. CPU tests and retained harness are not GPU proofs. Source
hashes are recorded in `source-sha256.txt`; concurrent unrelated files were
preserved. Source was frozen before the coordinator's final aggregate checks.
