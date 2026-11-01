# Kani repair follow-up, 2026-10-30

Draft PR: https://github.com/0xZKnw/mlxl3/pull/63 . This is a partial repair;
the complete Kani gate is still unverified. The first pushed head `dc889ba9`
failed core/Flash/speculative; native Ubuntu/macOS and allocator passed.

## Qualified changes

| Original obligation | Exhaustive replacement | Local result |
| --- | --- | --- |
| Allocator, every budget and four arbitrary u32 costs | budgets 0/1/2 plus every invalid usize budget | 4/4, already green in CI |
| Copy, prompt length 0..20 and accepted length 0..7 | 21 × 8 separately named cases | 168/168 |
| Latest prompt lookup, history length 0..20 | 21 constant-length cases, width still every usize | 21/21 |
| Expert prefix, rows 1..8 | eight constant-row cases | 8/8 |
| Prediction ordering, length 0..32 and every usize row count | valid geometries (1,16)/(2,32), all other geometries | 3/3 |
| Draft row validation, every usize proposal count | same domain, concrete error and match on result | 1/1; 4/4 covers |

All IDs, history values, anchors and original flags retain their symbolic
domains. Copy uses the real 1024-entry history. The production copy, lookup,
prefix and prediction-order functions are unchanged. Assertions and unwinding
checks remain enabled. Covers are emitted only in the finite partitions where
their original conditions apply, preserving the original union of goals. The
former copy harnesses emitted structurally unreachable covers for empty versus
nonempty constant prompts; the strict orchestration correctly rejected these.

`Kind::rows` now returns a private concrete error with the same display message.
Its seven callers still propagate it into anyhow using `?`. This avoids pulling
error formatting/backtrace machinery into a small arithmetic proof. CPU tests
exercise the error message and conversion. A changed row offset, a corrupted
copy anchor and first-instead-of-last lookup are all rejected by proof mutations.
These are deliberate mutations, not bugs discovered in unmodified production.

The compiled library inventory has **319 obligations**: the original 140 minus
25 replaced harnesses plus 204 exhaustive replacement cases. The checked-in
manifest requires every obligation, rejects a missing partition case, and still
assigns future harnesses automatically. The two compiled API instances remain
required in addition to the library inventory. There are still four CI workers,
not one runner per case. Limits remain 600 seconds per harness, 45 minutes per
shard, and 6 GiB address space per inherited process. This is not an aggregate
RSS cap; an external solver can have its own separate process allowance.

## Validation and evidence

- Rust CPU suite: 238 passed, 5 ignored; strict Clippy and formatting passed.
- Orchestration: 33 tests passed, including real subprocess failures, deadlines,
  interruption/reaping, incomplete verdicts and every missing grid case.
- Regular/verbose Kani output was checked against the actual verdict parser on
  the row proof: 67 checks, zero failed, four covers, 0.129 seconds.
- `followup-library-inventory.json` is the compiled inventory. The evidence
  archive contains the per-case reports and logs, including failed baselines.
  `spec-grid/report.json` intentionally still records failure of the old lookup
  harness; it establishes 168 copy successes, not a green speculative shard.
  `lookup-prefix-grid/report.json` and `ssd-grid-v2/report.json` supply the
  replacement verdicts. A complete exact-head CI rerun is required.

## Outstanding work

The unchanged end-to-end sidecar collection and grouped-route proofs remain
expensive. Constant-shape sidecar prototypes failed with CBMC status 6; a
borrowed-key prototype remains isolated. An eight-origin grouped-route pilot
timed out after 600 seconds. Reserved-vector and dereference-cache trials on
80 routes are isolated and still being assessed. No failing obligation has
been excluded, and no prototype production change has been promoted.

Local allocator/copy diagnostics explicitly report solver memory exhaustion;
this does not establish the sender or cause of the historical hosted runner
shutdown signals. Core/Flash CI timeouts are not counterexamples. The known
Desktop regression failure at `StudioModesCombined.swift:2250` is outside this
change. PR61/62 branches remain untouched. GitHub artifact materialization into
this environment failed with `download failed`; no local artifact was installed
and no bypass was attempted. Local reproductions and job logs remain available.
