# Resident snapshot qualification after 4ea54c7b

PR63 remains an incomplete repair. The last completed CI at
`4ea54c7bab6dda58946d1e45e8c891c70963f4d2` passed 312/319 library
cases and both API instances; the seven unique remaining names and verdicts
are recorded in the PR description and final CI artifacts.

The only executable change promoted in this checkpoint is inside the existing
`#[cfg(kani)]` resident snapshot harness. Production cache code is unchanged.
The required inventory is still 319 library harnesses plus both API instances.

## Preserved domain and obligations

For each of 20 slots, an arbitrary `u32` ID below512 is shared by two independently
chosen optional key/pending values, exactly as before. These values are first
prepared on the stack and then copied into the real `ExpertCache`. Query ranges
over every ID0..511 and is not passed to `resident_slot_map`.

The map oracle, clock assertion, three cover goals and unwind513 remain.
Instead of a second loop over all512 LFU counters, the proof asserts that the
counter at the independently universally chosen query remains zero.
For every fixed cache input state, `forall q: P(q) && counts[q]==0` is equivalent
to `forall q: P(q)` together with `forall q: counts[q]==0`. Thus no counter or
query is omitted. The real read-only snapshot implementation remains under test.

Cadical did not finish the reformulated proof within600s. With the bundled
Kissat solver and the same600s/6GiB per-process limits, the isolated proof passed
in148.056s. The final main-checkout proof independently passed in187.889s:
735 checks, zero failed, seven unreachable checks and 3/3 covers. This proves the
combined formulation and solver choice; it does not isolate a speed contribution
for each individual change.

Removing the actual production pending guard makes the existing exact native
snapshot regression fail. Restoring the source makes that test pass. The restored
production source is what the final main-checkout Kani proof checked. Formatting
and diff checks also pass. Earlier CPU238passed/5ignored and34 orchestration
tests remain applicable because no production or orchestration code changed.

## Other targeted work

The original80-slot pending-pattern harness passes unchanged in13.052s, with
574 checks, zero failed and3/3 covers. In the previous CI it had not been reached
before the Flash group exhausted its budget. This local result is not relabelled
as a CI result.

The remaining grouping, parser, two-route and sidecar candidates stay isolated
in `/workspace/kani-repair-next`. Buffer/scatter variants pass their independent
native oracle but still exhaust prover memory. A scalar watched-origin oracle
reduces one model from2.31million to459478steps but still fails memory reduction.
Targeted standard-sort bounds also fail memory reduction. Parser fixture parity
passes for all65536suffixes, but the attempted monolithic proofs time out.
The first distinct-misses partition of the two-route proof times out. These are
negative results, not counterexamples or green proof results.

New composition candidates require their real helper proofs before any caller
stub can be admitted. They are not part of this checkpoint. No new assumption,
stub, obligation exclusion, shard, memory increase or deadline increase is
promoted here. A new exact-head CI run is required after this checkpoint is pushed.

`resume-snapshot-evidence.tar.gz` preserves48 completed logs, receipts, source
snapshots and runner files, with a SHA256 manifest. Pending experiments are
excluded from that archive rather than presented as completed. The source and
archive fingerprints are in `resume-snapshot-receipt.json`; opti.md records the
protocols, failures and active experiments. PR61/62 are untouched. No cancellation,
merge, deployment, permission change or physical Apple GPU claim is made.
