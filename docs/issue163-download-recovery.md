# PR163 download transaction recovery follow-up

## Preregistration (2026-11-09, before implementation)

Baseline: published PR163 tree b28d33b, existing local unrelated edits preserved.
Prior N04/N05 completion-marker fix leaves a crash window after registry replacement,
reports false pending after marker and cleanup failures, and rolls back a valid install
when registry directory fsync fails after replacement. No speed measurements planned.

Design: durable transaction before publication, unique per-attempt identity copied into
published directory and registry entry. Registry truth plus matching receipt replaces
advisory completion markers. Resume reconciles published transactions without network.
Never undo publication on an ambiguous registry error. Sync JSON parents, assembled
files/directory, publication source/destination parents. Legacy markers alone are not
proof of success. Retained stages can be discarded without touching installed models.

Touched entry points: run_download, pending_downloads_in, resume_with_phase,
publish_checkpoint, write_json; registry register/additive managed wrapper.
Preconditions: exclusive per-download lock for mutation, managed destination containment,
verified checkpoint before publication, matching transaction/request and registry path.

Planned checks: deterministic crash boundary/restart tests before/after publication,
before/after registry persistence, both marker and cleanup failure, registry unreadable,
old attempt identity/reinstall, foreign destination and retry idempotency. Run focused
hub tests, contracts, fmt and strict clippy; attempt Kani on real decision predicate.
Use isolated temp fixtures, no external downloads, Mac, GPU, push or merge.
Mutation oracle must fail when committed detection ignores the attempt identity.

Formal-proof skill discovery: not available in catalog, checkout .agents/skills absent,
no matching skill found under /home/agent/.codex/skills. Source verifier availability
and attempts will be recorded; missing skill is not represented as proof.

Status: preregistered, implementation/test results pending.

## Results and final scope

Implemented locally in `native/src/hub.rs`, `hub/install_transaction.rs`,
`hub/install_transaction_tests.rs`, `hub/redirect_download_tests.rs`, and
`native/src/registry.rs`. No push, merge, Mac access, installation, or benchmark.
The optional registry `installation_id` defaults to absent and is omitted for ordinary
registrations, retaining version-1 compatibility. A managed publication has a
32-random-byte temporary-name nonce, durable transaction intent, and matching receipt.
The registry's atomically persisted identity is the commit truth, not `completed.json`.
An unrelated or stale receipt/registry never authorizes overwrite, deletion, or hiding
pending work. Explicit resume can reconcile even a committed retained stage and skips
metadata/network work. Failed registration and post-rename directory sync never roll
back a published model. Successful cleanup also syncs its parent directory.

Validation environment: Linux cloud checkout, Rust toolchain at
`/home/agent/.cargo/bin`, cargo target `/tmp/pr163-native-target`, debug info and
incremental compilation disabled for final checks. No GPU/model loaded.

- First targeted implementation: 5 tests passed. Expanded hub suite: 29 tests
  passed, then 30 after production resume integration, and 30 again after cleanup
  failure injection and bounded subprocess waits. No ignored hub tests.
- Final nonce-strengthening change: all 7 transaction tests passed. Child processes
  exit without destructors at durable intent, publication, and registry replacement;
  restart/retry uses real filesystem and registry code. Production `resume_with_phase`
  is exercised in isolated subprocesses, with only `install` progress and no network.
  Each child is bounded to 20 seconds and killed/reaped on timeout or wait failure.
- Directory sync fault injection covers both published-directory rename and registry
  replacement, including the ambiguous registry error after its JSON is already
  visible. The installation stays present and retry completes it.
- A retained stage with an unwritable completion marker and an injected cleanup
  permission failure remains correctly classified as committed, without using the
  marker. Corrupt registry, foreign request, different registry destination,
  foreign receipt, deleted installation/reinstall, and repeated retry are covered.
- `cargo test --test contracts --no-default-features`: 17 passed, 0 ignored.
- Strict `cargo clippy --lib --no-default-features -- -D warnings` passed on final
  files. The broader `--lib --tests` attempt first encountered another worker's
  `ssd_reads/window.rs` collapsible-if; rerun encountered concurrent references to
  feature-gated `vision_input` in other targets. Both diagnostics retained, delegated
  to parent for aggregate resolution; not claimed as a full lint pass.
- Rustfmt check for owned Rust files and `git diff --check` passed.
- `timeout 180 cargo kani --lib --no-default-features --harness
  completion_requires_all_three_identities`: Kani 0.68.0 / CBMC 6.11.0, success,
  1 harness, 0 failures, 2/2 cover properties satisfied. Symbolic domain is exactly
  three booleans through the production completion predicate, with no assumptions.
  This proves that predicate only, not filesystem persistence, power-loss behavior,
  concurrency, OS permissions, or the entire download implementation.
- Mutation oracle: an isolated source copy ignored the registry attempt identity.
  Its reinstall regression failed as expected (exit 101, stale registration was
  wrongly considered completed). Production source was never mutated and passed
  afterward. First isolated mutation build lacked an `include_str!` fixture and
  failed compilation; fixture copied and rerun recorded separately. Initial shell
  test attempt failed because cargo was not on PATH, then corrected explicitly.

Evidence: `docs/measurements/pr163-download-recovery/` retains the test, mutation,
Clippy, contracts, and Kani logs, including unsuccessful attempts. Focused checks
are not a full delivered-feature/Apple/CI validation. Parent owns aggregate checks.

Remaining inherent limits: persistent OS refusal can still prevent physical staging
removal. Such leftovers do not appear as pending when the registry and receipt match;
explicit retry/discard can clean them later. Pending listing is read-only. Existing
legacy markers without transaction evidence cannot prove a committed installation,
so they no longer hide interrupted work. Physical power-cut behavior and hostile
external mutation of managed files were not tested. Formal-proof skill was unavailable,
while the actual source-level Kani attempt above was completed successfully.

Status: implementation and focused correctness checks validated locally; aggregate
feature/CI and physical durability checks remain separate. No performance claim.

Final lint follow-up: `cargo clippy --lib --tests --no-default-features --features
chat -- -D warnings` passed (exit 0) with the actual chat-enabled target graph.
Evidence: `docs/measurements/pr163-download-recovery/clippy-chat.log`. Earlier
feature-omission diagnostics remain preserved above. All work in this follow-up
is now complete; no running experiments remain.
