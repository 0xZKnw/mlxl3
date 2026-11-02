# Cold-ready P1: independent runtime and test review

1 November 2026. Source review of the local dirty tree, after P1 preregistration.
No GPU execution, benchmark, Cargo build or proof was performed by this reviewer.
The parent owns the coordinated verification and `opti.md` updates.

## Scope

Read `opti.md`, the formal-proof skill and Rust reference, current SsdMoe
ColdConsumers/fill_ready/forward assembly, Array's batch wrapper and C++ batch
publication/lifetime implementation. Test additions are confined to
`native/src/ssd_moe/cold_ready_tests.rs`. The parent adds its module declaration.
The existing external reference format and layer-zero MoE values are reused.

## Runtime findings

- The first wave is partitioned by completed **slot**, preserving every route
  occurrence. The rest follows it in `cold`. `first_rows` is the physical row
  count of W1, so slicing `cold[first_rows..]` selects W2, including B2 duplicates.
- The existing output remap indexes the reordered Cold layout by original
  occurrence. GPU Hot keeps an `ids.len()` physical offset; compact Hot uses
  `hot.len()`. The final cast, weight multiplication and single Top10 sum remain
  in canonical order. No separate partial sums are introduced.
- Async read metadata validates all destinations before workers launch. Each
  reader writes disjoint ranges; publication follows complete record reads,
  with release/acquire ordering. The wrapper retains cloned files and C++ MLX
  owners, and is not Send/Sync. Workers do not invoke MLX.
- The batch local is dropped inside fill_ready on failure before the caller's
  ColdConsumers guard. That guard is declared after ExpertFill, so it drains
  Hot and W1 before ExpertFill invalidates keys. On success, `last` retains the
  final dependent output before the guard's references are cleared.
- W1 is retained before async_eval; a submission error therefore still leaves
  a consumer for cleanup. The loop attempts every consumer even if one eval
  fails, then attempts synchronize. A Metal/context error that defeats both
  eval and synchronization remains unverified; logging is not a proof of a
  usable GPU context after that failure.
- Two issues were reported to the parent during implementation: COLD_READY=1
  plus DYNAMIC_READERS=1 initially silently selected static async scheduling;
  and the initial Hot `try_clone()?` could return before the guard acquired its
  reference. `self.last` still retained Hot in that allocation-failure case,
  but the stronger immediate-drain-on-error claim needed qualification or a
  catch-and-drain fix. The parent is responsible for final resolution and
  affected tests; this record does not declare either fixed.

## Tests added, not yet executed

Exact filters below are under `ssd_moe::cold_ready_tests::`:

1. `ready_cold_waves_preserve_external_rows_duplicates_and_fallbacks`
   checks 20 mixed cases (B1/B2, both Hot layouts, 20/53 slots), 20 all-Hot
   controls, two empty-cache controls, four one/two-job fallbacks, one B2 case
   sharing expert IDs with different activations, and one B3 fallback.
   Each mixed case holds job zero until finish, forcing non-prefix readiness;
   the runtime test hook evaluates W1 and verifies a record is still pending.
   It asserts execution counters, distinct read-byte accounting, shapes,
   finite values, exact external output bytes, and recovery of canonical order.
   The scaled-input duplicate case compares a separately evaluated stock row;
   it is not an external golden for that additional scaled input.
2. `ready_cold_failure_drains_consumers_invalidates_reservations_and_recovers`
   injects a Rust error after W1 submission, without the success test's early
   eval. Slow Hot is enabled. It checks that W1 was submitted, Hot is complete
   after the error, all reserved IDs were invalidated, and retry/reset match
   external golden bytes. This does not prove W1 was pending at injection, does
   not inject a late OS read failure, and does not simulate a GPU context fault.
   CPU reader tests and the parent's integration checks cover separate layers.
3. `ready_cold_invalid_flag_fails_before_loading_weights`
   is a separately selected ignored test with COLD_READY=invalid. An empty
   Checkpoint ensures rejection happens at the actual loader flag boundary
   before accessing weights. No model or GPU is required for this selector.

The physical tests require FLASH_MODEL and FLASH_REFERENCE, COLD_READY=1,
SHARED_EXL3=0 and WEIGHTED_EPILOGUE=0; select them individually in a fresh test
process. They deliberately avoid mutating checkpoint files or process-global
Rust environment variables. No tests are silently skipped inside a selector.

The tests validate final per-token MoE outputs and execution counters. They do
not expose and separately compare every intermediate expert contribution,
complete-model recurrent states, logits, MTP/DFlash acceptance or throughput.
The pure partition helper's proof belongs to the separate CPU agent; it cannot
prove Metal, the I/O race boundary or the full forward path.

## Commands executed here

`/Users/justin/.cargo/bin/rustfmt --edition 2024 native/src/ssd_moe/cold_ready_tests.rs`
completed successfully. Source searches/read commands only otherwise.
Build, clippy, physical GPU tests, Kani/FFI verification and exact-commit remote
CI remain coordinated by the parent, not claimed by this review.
