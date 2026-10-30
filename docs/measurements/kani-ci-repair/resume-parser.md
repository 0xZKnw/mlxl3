# Layer-profile parser qualification after 8f0cea95

PR63 remains an incomplete repair. The completed CI at `8f0cea95` passed
314/319 library harnesses and both API instances. Five unique harnesses timed
out; this checkpoint locally qualifies one of them, the layer-profile parser.
The next exact-head CI must confirm it. No cache or grouped-route prototype
from the unsuccessful experiments is promoted.

## Preserved contracts

The production parser now scans comma boundaries with an explicit byte cursor.
It still requires exactly48 fields, delegates each field to Rust's actual
`usize` parser, and runs the same capacity/total-budget validator. ASCII comma
boundaries cannot split a UTF-8 character. Missing, empty and additional fields
still fail. Numeric syntax, signs, Unicode rejection and overflow behavior come
from the unchanged standard conversion; no new decimal parser is shipped.

The original proof domain remains every two-byte suffix after47 copies of
`53,`, restricted to the same valid-UTF-8 inputs. A fixed143-byte fixture and
its guaranteed8-byte alignment avoid dynamic String construction. An exhaustive
native check compares all65,536 fixture byte sequences and their UTF-8 validity
with the original construction. The parser oracle still requires acceptance
exactly for suffix `53`, and every returned slot must equal53. Both original
cover goals remain satisfied.

Only the scalar conversion helper is substituted in this caller. Its exact
result on lengths0,1,2 is proved against the real standard parser by three
mandatory independent harnesses over all byte values, including `+5` and
invalid UTF-8. The replacement asserts length<=2 at every actual call; it does
not assume it. The other319 obligations remain required, giving322 library
harnesses plus both API instances. These finite proofs do not establish the
parser's behavior for every possible-length input string.

## Bounds and orchestration

Kani0.68.0/CBMC6.11.0 and the600-second/6GiB limits are unchanged. Global unwind
is144. The fixture's scan examines at most two field bytes before comma/end,
so its explicit loop has bound4. The aligned143-byte UTF-8 fixture needs at
most nine16-byte ASCII block inspections, a scalar tail of at most16 bytes,
and three outer iterations; conservative bounds10/17/4 are retained, with
all unwinding assertions enabled. A wrong bound causes verification failure.

The runner resolves these four IDs from CBMC's compiled loop inventory and
their pretty function names. Missing, duplicate or foreign mappings fail;
compiler-mangled symbols are never hardcoded. Discovery is diagnostic only:
verification still runs separately with modern output and the strict verdict
validator. Discovery and verification share the existing per-harness budget.
Only this caller's exact conversion stub is admitted; the three companion
proofs cannot use it. The manifest makes all three mandatory.

An observed experimental Kani warning about missing call arguments and inserted
nondeterministic values now causes the validator to reject an otherwise green
summary. Ordinary Rust warnings remain permitted. Its regression fails before
the fix and passes after it.

## Qualification and remaining work

On the final main-checkout source, the composed proof passes in87.497s; the
three real conversion proofs pass in2.821/3.270/3.478s. All functional assertions
and cover goals pass; unreachable checks are recorded in the full logs. A
mutation that rejects the last field in the real parser fails its acceptance
assertion in59.515s; restoration passes in86.109s. The earlier real-conversion
mutation is preserved in the preceding experiment records.

Native Rust:240 passed,5 ignored. Orchestration:44 passed. Strict Clippy, Ruff
and formatting pass. CrossHair's attempt on the actual warning validator
remains `Not confirmed`, and is not counted as a proof. The receipt fixes
source/evidence hashes and archives raw positive and negative results.

The four other original CI timeouts remain open: sidecar collection, grouped
routes on8 and80 symbolic entries, and two-route cache placement. Local sidecar
and composed placement attempts reached explicit memory failures; grouped
lemmas succeeded on8 entries but timed out on80. Those local backend failures
are not counterexamples and do not change the recorded CI timeout verdicts.
No proof domain, required harness, failure status or resource limit is hidden.

Updating PR63's description previously returned an API error; a read confirmed
that the write did not apply. The diagnostic was not retained, so its reason
is unknown. No retry or alternate write route was attempted. This does not
prevent the separately authorized normal branch push.
