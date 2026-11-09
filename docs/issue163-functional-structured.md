# PR163 functional follow-up: bounded structured comparisons

## Preregistered scope and protocol (2026-11-09)

Baseline: `8080c96c`, parent experiment `PR163-FUNCTIONAL-8080-20261109`.
History: [protocol reconstruction, T06](issue163-protocol-reconstruction.md).
Only `native/src/structured.rs` and this report are owned by this work.
No CI, Mac, model, benchmark, publication, or environment-secret work.

Confirmed source gap: `const`/`enum` call recursive `json_equal` without sharing
`ValidationBudget`; its string/key comparisons do not poll cancellation. The
`uniqueItems` path charges Value nodes but clones strings/keys, serializes them,
and compares complete serialized strings outside that accounting.

Plan, before implementation or tests:

1. Add deterministic production regression using `items.enum` with 256 distinct
   same-length candidates and 5,000 last-candidate values. The schema and output
   fit the public input caps. Expect budget rejection, including under `not`.
   Run this test against unchanged production first to establish the defect.
2. Charge recursive equality nodes and bounded 256-byte string/key chunks using
   the shared sticky budget. Stop enum traversal immediately on exhaustion.
   Preserve exact integer/float equality, signed zero, and symmetry.
3. Replace full canonical clones/serialization with incremental typed canonical
   hashing using existing SHA-256. Store fixed-size digest buckets; every digest
   match requires bounded exact equality, so collisions cannot create false
   duplicate detection. Charge canonical nodes and byte chunks, and retain the
   existing 100,000-distinct-integer acceptance oracle.
4. Add exact small-budget boundary tests (including exhausted/cancelled state,
   strings longer than one chunk, object keys and nested nodes), public numeric
   parity cases at 2^53/i64/u64 boundaries, and a forced collision-bucket test.
   No timing-dependent cancellation race is used as a deterministic oracle.
5. Run `cargo test --locked --no-default-features --features chat --lib
   structured::tests -- --nocapture`; then strict portable Clippy, owned-file
   Rustfmt and diff checks. Parent owns aggregate tests and `opti.md` updates.
6. Extend the real budget Kani harness for sticky exhaustion and multi-unit byte
   charging, then attempt targeted Kani with a 180-second timeout, preserved
   unwinding assertions and outcome covers. No full parser/hash/allocator or
   concurrent execution proof is claimed. The formal-proof skill/Rust reference
   is absent from the executor catalog and checkout; parent confirmed no cached
   copy is known. Record that limitation rather than fabricate a loaded skill.

Commands use `PATH=/home/agent/.cargo/bin:$PATH`,
`CARGO_TARGET_DIR=/tmp/pr163-native-target`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`.
Temporary raw logs: `/tmp/pr163-structured-{baseline,tests,clippy,kani}.log`.

Callers: `Schema::validate_cancellable` serves real native structured-generation
validation/retry; `validate` wraps it with a false cancellation flag. Schema
admission is also used by bridge/OpenAI/CLI. Public text caps and schema depth
checks stay unchanged. The new budget remains a conservative work allowance,
not a wall-clock deadline. Parsing and final output serialization remain bounded
by input size but are not interrupted midway; no total CPU-complexity proof.

Status: preregistered; implementation and execution not yet started.

## Baseline reproduction

The preregistered public `enum` regression failed against unchanged production:
0 passed, 1 failed, exit 101, 420 filtered out. The invalid success returned the
entire 5,000-item output, confirming this is an executable acceptance gap rather
than only a theoretical source observation. Raw baseline log retained at the
path above. The failure message was subsequently shortened to avoid dumping the
entire valid-sized output in later failures. Implementation now in progress.

## First corrected tests and proof-attempt preparation

All 13 structured tests pass (0 failed/ignored, 413 filtered out, 0.66 s),
including the public baseline regression, forced digest collision confirmation,
chunk-budget boundaries, sticky cancellation, and 30 symmetric numeric/nested
case orientations through each of const/enum/uniqueItems. Existing 100,000
integer acceptance remains green. No performance comparison was performed.

Parent requested the proof timeout be tightened from 180 to 120 seconds.
Kani 0.68.0 / CBMC 6.11.0 is installed now, unlike the historical report.
Reuse the existing target (363 MiB free on `/tmp` at inspection). Attempt the
updated budget harness and `byte_comparison_spends_each_chunk` together:
`timeout 120 cargo kani --lib --no-default-features --harness
validation_budget_never_wraps_or_accepts_cancellation --harness
byte_comparison_spends_each_chunk`. The new harness has symbolic u8 allowance,
cancellation and last-byte equality over two fixed byte chunks, no assumptions,
and equality/inequality/exhaustion covers. Hashing itself is tested, not proved.

First Kani attempt: the sticky-budget harness passed (63 checks, 1 unreachable,
2/2 covers). The chunk harness did **not** verify: the global unwind limit 4
was insufficient for CBMC's real 256-byte `memcmp` loop. One unwinding assertion
failed, 250 checks and 2 covers remained undetermined, 1 cover was satisfied.
This is a proof-bound failure, not a production counterexample. Before retry:
increase only this harness's unwinding bound to 260, preserve the identical
symbolic domain and all assertions/covers, and rerun it under 120 seconds.
The parent-reported strict Clippy `err_expect` diagnostic in the test was fixed
with a `let Err(...) else` assertion; production unchanged.

## Final implementation and verification

**Status: locally validated, sources frozen; no publication or CI claim.**

- `json_equal` spends for each visited comparison node; strings and object keys
  spend/poll once per at-most-256-byte chunk. `enum` stops on exhaustion rather
  than scanning remaining candidates. Exhaustion/cancellation is sticky, and
  the public post-match check prevents `not` from accepting interrupted work.
- `uniqueItems` streams type/length-framed canonical values into SHA-256 instead
  of cloning and serializing complete values. The balanced index compares only
  fixed 32-byte digests. Each digest bucket is checked with exact, budgeted
  equality before insertion. A forced collision test proves unequal values are
  retained and equal integer/float nested values rejected within that bucket.
- Numeric conversion semantics are preserved: integers are never rounded through
  f64; only integral floats within the signed/unsigned conversion range normalize.
  Signed zero, equivalent integral floats, nested identity and distinct integers
  beyond 2^53 all retain the tested behavior.
- Object traversal relies on the existing sorted `serde_json::Map`; `cargo tree
  --locked --no-default-features --features chat -e features -i serde_json`
  confirms `preserve_order` is not enabled. The reversed-key nested regression
  guards that assumption for current features.

Final commands and results:

| Control | Result |
| --- | --- |
| `cargo test --locked --no-default-features --features chat --lib structured::tests -- --nocapture` | 13 passed, 0 failed, 0 ignored, 416 filtered out, 0.81 s |
| `cargo clippy --locked --no-default-features --features chat --all-targets -- -D warnings` | Passed, 7.72 s |
| `rustfmt --edition 2024 --check native/src/structured.rs` | Passed |
| `git diff --check -- native/src/structured.rs docs/issue163-functional-structured.md` | Passed before final report append |
| Sticky-budget Kani harness | 63 checks, 0 failures, 1 unreachable; 2/2 covers |
| Chunk-comparison Kani retry, unwind 260 | 250 checks, 0 failures, 3 unreachable; 3/3 covers, 2.09 s verification |

The Kani unreachable checks concern standard-library formatting/alignment/
allocation-size guards, not the explicit functional assertions or outcome
covers. The first insufficient-unwind failure remains recorded above. The
unchanged symbolic domains are arbitrary `usize` budget + cancellation + prior
exhaustion for `spend`, and arbitrary `u8` budget + cancellation + equality of
the final byte for two fixed chunks. These are bounded model-checking results,
not proofs of arbitrary JSON trees, SHA-256, allocator behavior or races.

Final source SHA-256:
`d70476885acc23b3c39322d81d81fdccfc4ea8d2ada91f69b03811a70ecf93c0`.
The filtered-test count changed during concurrent work; neither that count nor
other workers' code is attributed to this component. Parent owns aggregate
verification, `opti.md`, and any publication.

## Remaining limits

The shared allowance remains 1,000,000 work units; each comparison/hash node
and each byte chunk spends one. This is deliberately not a count of CPU
instructions or an end-to-end timeout. Parsing, final output serialization,
existing string-length scans and schema-property lookup operations are unchanged
and not made individually cancellable here. Input caps still apply. No broader
claim about complete validator CPU accounting, latency or RSS is made. No
physical inference, Mac, CI, speed benchmark or throughput gain was measured.

Raw logs remain temporary in `/tmp/pr163-structured-baseline.log`,
`/tmp/pr163-structured-tests.log`, `/tmp/pr163-structured-tests-final.log`,
`/tmp/pr163-structured-clippy.log`, `/tmp/pr163-structured-kani.log`, and
`/tmp/pr163-structured-kani-retry.log`; parent was given these paths for evidence
collection. No proof attempt or subprocess remains active for this component.

## Integrated evidence delivery

Raw logs referenced above are retained inside
[functional evidence archive](measurements/pr163-functional-evidence.tar.gz),
under the corresponding `pr163-functional-*` directory. Temporary structured
logs were copied into `pr163-functional-structured` before archiving. Coordinator
integration on base `8080c96c`: 420 Rust library tests passed, 10 ignored;
format and the 314-file source inventory passed. Hosted CI and physical Apple
qualification are separate and not claimed here.
