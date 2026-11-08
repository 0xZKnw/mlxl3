# Issue 160: correctness qualification

Qualified runtime source: `830bb38a9039fb56f2474a3735a5872d51efaaee`, PR #161.

No throughput benchmark was run. No speed improvement or target tokens/second result is established.

## Passed controls

- `cargo clippy --locked --all-targets --features chat -- -D warnings` passed.
- `cargo test --locked --features chat -- --test-threads=1`: **442 passed, 128 ignored**. Ignored physical tests were selected separately, not counted as standard-suite passes.
- Three targeted Kani contracts passed: `invalid_sample_preserves_statistics` (175 checks, 2 covers), `memory_admission_never_exceeds_limit_or_budget` (153 checks, 3 covers; one unreachable standard formatting assertion), and `verify_rows_dispatch_is_bounded_and_opt_in` (77 checks, 2 covers). Proof runs were on `6d1af312772203ab5bac4be9cc32626f63837305`; their contract implementations were unchanged in the final source. These bounded CPU proofs do not establish Metal or I/O concurrency correctness.

## Physical test results

The campaign recorded 12 passing non-mutant invocations and three initial failures subsequently resolved. Eleven distinct test functions were exercised; shared-expert batching was checked with its flag both off and on.

- `physical_flash_verify_shared_rows_match_scalar`: 11,538,432 exact FP16 output words compared. Qualified on `6d1af312772203ab5bac4be9cc32626f63837305`; the tested projection implementation was unchanged afterward.
- `shared_expert_block_matches_scalar_with_offset_rows`: batching off and on, offset rows and F16/F32 inputs passed on `c86d7f8ee933ea238ffe58dc0723b1427a23209d`; the component was unchanged afterward.
- `physical_mtp_kv_only_matches_attention_state_and_next_query`: retained states and next-query parity passed on `c86d7f8ee933ea238ffe58dc0723b1427a23209d`, and again on the final source after the negative control.
- `block_experts_read_once_and_recover_after_failed_fill`: final source passed canonical outputs, read-once accounting, capacities 10/20/40, cancellation and failed-fill recovery.
- `physical_mtp_layer_extend_uses_kv_only_and_preserves_next_output`: final source passed production-layer continuation parity.
- `dropped_invalid_failed_and_cancelled_blocks_preserve_target`: final source passed staging rollback.
- `final_head_only_prefill_matches_per_token_heads`: final source passed logits/state parity and vocabulary-head counts.
- `device_proposal_matches_normalized_host_on_oracle_rows`: final source passed against held-out oracle rows.
- `block_verification_matches_scalar_mtp_and_eos_prefixes`: final source passed acceptance, rejection, pending-token and EOS parity.
- `ssd_cohort_verification_matches_scalar_and_rejects_unsafe_workspace`: final source passed the actual SSD-cohort path with shared QMV enabled, widths 2–4, every retained prefix, logits/state/hidden comparisons, continuation, workspace rejection and rollback.
- `layer_major_verification_matches_scalar_prefixes_and_continuations`: final source independently matched ordinary scalar forwards for widths 1–4, every retained prefix and continuation.

Negative control: temporarily omitting the KV-only K append caused the retained-state oracle to fail as expected. Exact source restoration, a clean source diff, reconstruction and a passing rerun followed. No mutant was committed. This is additional to the 12 passing campaign invocations.

## Failures and corrections

- Initial formatting/test lint issues were corrected; strict checks passed.
- Initial SSD compilation failed because of an inaccessible flag parser and a vector passed where an iterator was required. Narrow corrections passed compilation.
- Four HTTP fixture tests failed in the initial parallel suite. All 26 transport tests and the full standard suite passed serially without a runtime change. This is not evidence of a proven pre-existing regression.
- The first SSD test exceeded the reader's maximum submission size. The correction retains the full cache reservation but submits and drains reads in groups of at most ten. The same test then passed.
- The KV-layer fixture requested one cache slot, below the ten-slot minimum. The fixture was corrected and the numerical test passed.
- An initial model load stopped at automatic memory admission before model computation. Later qualification passed with the unchanged admission policy. No guard was bypassed.

## Remaining limits

- No throughput, SSD bandwidth or memory-peak benchmark was executed. Test durations are not performance measurements.
- Recursive depth-2/depth-3 MTP, adaptive-depth runtime integration and prospective-budget admission are not qualified by this report.
- The original stock MTP-session oracle test and a separate full bridge/CLI generation campaign were not executed here.
- Numerical parity applies to the tested configuration and explicit flags, not every model or flag composition. Experimental paths remain opt-in.
- No merge, release or installation occurred.
