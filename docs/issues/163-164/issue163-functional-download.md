# PR163 functional download recovery follow-up

## Preregistration (2026-11-09, before code and tests)

Baseline: `8080c96c`, isolated cloud worktree. Previous transaction follow-up:
`docs/issues/163-164/issue163-download-recovery.md`, especially cleanup failure coverage. Its
injected removal failure leaves the entire stage untouched. A real
`remove_dir_all` can delete `transaction.json` and then fail before deleting
`request.json`. A committed installation then reappears as pending and retry
incorrectly requests remote metadata before encountering the existing destination.

Hypothesis and change: when the staging transaction is absent, reconstruct it only
from an installed receipt that agrees with the request and an already committed
registry identity. Never treat a receipt alone as permission to register an
installation or replace a different registry entry. Keep corrupt/conflicting
staging transactions fail-closed. Normalize the existing managed root before its
containment comparison, so listings through symlink aliases recognize the same
committed destination as the canonical publication path.

Affected production entry points: `InstallTransaction::load`, called by
`completed_download` (read-only listing) and `recover_publication` (explicit
resume); `registered` / `receipt_matches` retain the three-identity contract.
Preconditions: retained request belongs to the managed root, request/receipt
identity matches, explicit resume owns the download lock. No metadata requests or
payload changes should occur for a committed recovered transaction.

Planned checks, before execution:

- Add deterministic tests and run them against unchanged production to reproduce
  partial-cleanup and root-alias failures. Exercise the real pending listing and
  production resume in isolated, bounded subprocesses with local fixtures.
- Reject fallback when registry is absent, corrupt, points elsewhere, or has a
  different attempt; reject mismatching request and corrupt staging transaction.
  Verify no registry mutation and no deletion in these cases.
- Run affected hub tests, CLI contracts, Rustfmt and strict chat-enabled Clippy.
  Existing Kani `completion_requires_all_three_identities` is rechecked because
  its actual production predicate now also gates fallback. Its three Boolean
  inputs are unconstrained; it does not prove filesystem or concurrency behavior.
- No CI changes, remote writes, Mac access, model/GPU tests, or benchmark.
  Rust uses `/home/agent/.cargo/bin`, target `/tmp/pr163-native-target`, debug info
  and incremental compilation disabled. Logs are retained beside this report.

The required formal-proof skill and Rust reference are unavailable: no matching
cloud catalog entry, checkout `.agents/skills`, or installed skill package was
found. A source-level Kani attempt remains planned, not a substitute claim for
the missing skill.

Status: preregistered; implementation and validation pending.

Baseline regression results: both partial-cleanup tests fail on unchanged
production (0 passed, 2 failed). The positive test reproduces false pending;
the negative test currently returns no transaction instead of rejecting the
uncommitted receipt, and will guard the new fallback. Root-alias reproduction
is recorded separately before production edits. No baseline production resume
was run because the known fallback would issue a real metadata request.

Initial implementation results: root-alias regression also failed on unchanged
production (0 passed, 1 failed). With the fix, all 10 transaction tests pass,
including six rejected fallback variants, two alias paths (with and without
intent), and production resume after publication, registry persistence, and
partial cleanup. Resume subprocesses use a nonworking loopback proxy so a
regression cannot fetch external metadata; each child has a 20-second bound and
is killed/reaped on timeout. Registry bytes and timestamps stay unchanged in
the successful fallback; failures preserve the installed model and registry.
Full hub/contracts/lint and predicate Kani checks remain pending.

## Additional retained-request confinement preregistration

Before further code/tests: `resume_with_phase` trusts `request.json`'s ID when
constructing the staging directory, and listing follows stage symlinks. A
symlinked `.downloads` parent also redirects `discard` outside managed storage.
Harden only these retained-download entry points: validate 12-hex IDs, match the
entry name to the retained ID, reject symlinked `.downloads` and stage entries,
validate the canonical managed destination before metadata/network or staging
writes, and use no-follow regular lock files. Root aliases themselves remain
supported by canonicalization. Malformed listing entries are skipped.

Tests planned before execution: production resume/discard subprocess cases for
invalid IDs, stage/ID mismatch, outside destination, symlink stage, symlink
`.downloads`, and symlink lock/request files. External sentinel directory and
files must be byte-for-byte unchanged; every process uses a blocked loopback
proxy and the existing 20-second bound. Preserve normal recovery/root-alias
tests. No claim of race-free hostile filesystem containment: eliminating races
in ancestor replacement would require descriptor-relative filesystem operations.
Parent authorized `hub.rs` ownership for this same functional scope.

Previous checks completed: hub suite 33 passed (0 ignored), contracts 17 passed
(0 ignored). Existing predicate Kani attempt is in progress.

Kani completion result: 0.68.0 / CBMC 6.11.0, one actual production predicate
harness passed, assertion passed and 2/2 covers satisfied. Existing unrelated
compiler warnings are retained in its log. The retained listing regression
failed on the unchanged boundary code (0 passed, 1 failed, invalid ID admitted).
The existing path-admission harness is extended before its rerun: original
arbitrary-byte paths of length 0..8 remain, plus independent arbitrary-byte
download IDs of length 0..13; unwind increases from 12 to 16. It checks exact
12-hex acceptance and that accepted IDs satisfy the real safe-path predicate.
No assumptions constrain ID contents; no filesystem proof is claimed.

Boundary implementation: 13 transaction tests pass, including all 13 subprocess
boundary cases and entire isolated tree snapshots. Broader hub rerun: 35 passed,
1 failed. The pre-existing legacy-marker fixture places staging directly under
the temporary root while its destination is a sibling of the stage; it does not
match production's `root/.downloads/id` layout. Before rerun, correct this fixture
to the production layout and canonical temporary root without weakening any
marker assertions. Strict chat-enabled library/tests Clippy passed. Further
test isolation sets both proxy variable casings and a disposable HOME, and
removes HF_TOKEN from recovery children; this prevents inherited environment
settings from enabling real metadata access in a regressed implementation.

Expanded path-harness attempt ended with CBMC out-of-memory (exit 1), not a
counterexample; the log retains its full diagnostic. Audit: its old nested
split-iterator oracle expands substantially with the globally raised unwind
bound. Revised protocol before another attempt: restore the historical path
harness unchanged, and append the independent 0..13-byte ID admission assertions
to the existing small transaction identity harness (unwind 16). Preserve its
original three symbolic Booleans/assertion/covers, and check real ID acceptance
against an independent hexadecimal oracle plus forbidden separator/dot checks.
No ID domain reduction, stub, or suppressed unwinding assertion. No CI edits.

Review refinement before final proof: keep the identity harness unchanged too,
and give ID admission its own clearly named source harness,
`retained_download_ids_are_bounded_hex_components`. Harness auto-discovery does
not require a CI edit. The interim combined-proof attempt is retained as a
separate attempt and is not treated as the final-source verification.
The corrected legacy fixture and fully isolated children pass the full hub suite:
36 passed, 0 failed, 0 ignored. Strict chat library/tests Clippy also passes on
these final runtime and test sources. Final proof-only edit is pending validation.

## Final counterexample audit preregistration

Before the final correction: receipt fallback confirms an existing registry in
`load`, but another process could remove or replace that registration before
`register`. Its generic retry would then register the model anew, contrary to the
receipt-only recovery contract. Add a nonserialized receipt-only provenance bit;
reconciliation may return an existing exact registration, but cannot create one
without staging intent. Test the deterministic load -> registry removal/replacement
-> register interleaving before and after this guard, preserving registry bytes.
This covers a normal concurrent registry edit, not arbitrary filesystem races.

## Final result

All owned runtime and test sources are complete locally. Implemented:

- Reconcile a partially removed staging transaction from the installed receipt
  only when the exact request, destination, and existing registry attempt agree.
- Preserve receipt-only provenance in memory (never serialized), preventing a
  later registry removal/replacement from being undone by fallback recovery.
- Recognize committed installations through legitimate managed-root aliases.
- Reject invalid/mismatched retained IDs, outside destinations, symlinked
  staging/downloads/request paths, and nonregular/symlink locks before transfer
  or destructive staging operations. No descriptor-relative race guarantee.

Validation on final runtime sources:

- `cargo test --locked --lib --no-default-features --features chat hub::`:
  **37 passed, 0 failed, 0 ignored**. This includes 13 independently isolated
  production boundary scenarios, three crash/resume states, missing/corrupt/
  conflicting fallback evidence, and both deterministic registry interleavings.
- The new registry-interleaving regression failed before the provenance guard
  (0 passed, 1 failed), then passed in the final full hub suite. The original
  partial-cleanup/root-alias/listing failures and intermediate legacy fixture
  failure remain recorded; none is silently replaced by a success log.
- `cargo clippy --locked --lib --tests --no-default-features --features chat
  -- -D warnings`: passed. Owned Rustfmt and full working-tree diff checks pass.
- Final Kani 0.68.0 / CBMC 6.11.0 source rerun, 180-second cap: two harnesses
  passed, 4/4 covers satisfied. ID acceptance: 273 checks, 13 unreachable;
  completion identity: one check. IDs cover arbitrary bytes at lengths 0..13
  with unwind 16, and the original three completion Booleans remain unrestricted.
  The original path harness is byte-for-byte unchanged. The interim combined
  harness also passed (274 checks, 13 unreachable, 4 covers) but was superseded
  by the clearer standalone ID harness. The earlier path expansion failed from
  memory exhaustion, not a counterexample. Existing compiler warnings and
  unreachable unsupported-construct notices are retained in the logs.
- Contracts passed 17/17 before the retained-boundary expansion; the parent owns
  final integrated contracts/full-library checks. Focused results are not a
  claim of complete Apple/CI validation.

Evidence: `docs/measurements/pr163-functional-download/`; final runtime file
hashes in `final-sources.sha256`. Logs retain all failed attempts and are
normalized only to remove trailing whitespace. No benchmarks, throughput claims,
Mac/model/GPU tests, CI edits, publication, remote writes, or `.env` access.
Kani proves only the finite identifier/identity predicates, not filesystem I/O,
registry concurrency, cryptographic hashing, or persistence under power loss.
Physical removal can still fail if the OS refuses it; recovery no longer mistakes
the covered partial cleanup for an uncommitted installation. All owned checks
are finished; no experiments or subprocesses remain running.

## Integrated evidence delivery

Raw logs referenced above are retained inside
[functional evidence archive](../../measurements/pr163-functional/pr163-functional-evidence.tar.gz),
under the corresponding `pr163-functional-*` directory. Temporary structured
logs were copied into `pr163-functional-structured` before archiving. Coordinator
integration on base `8080c96c`: 420 Rust library tests passed, 10 ignored;
format and the 314-file source inventory passed. Hosted CI and physical Apple
qualification are separate and not claimed here.
