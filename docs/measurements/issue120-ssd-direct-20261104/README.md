# Flash AR: read expert records into their final owned destination

P01 was selected from the current native profile, rather than issue120's numbering.1097expert-record copies account for99.7% of measured upload time,12.734ms per instrumented decode row. The native reader now fills a fresh Shared Metal backing directly instead of initializing a temporary Vec and copying it afterward. This does not remove SSD bytes or shader reads. [Priority analysis](../issue120-profile-priorities-20261104/README.md).

The read job exclusively owns its fresh allocation through physical completion. Only a complete successful read returns a payload; the inference thread validates the lease generation before publishing the expert. There is no pool, reuse, naked pointer transfer or new unsafe Send implementation. Old background reads remain on the existing byte-vector path. Active reads still block the next demand cohort; abandoned jobs retain their own destination. Device allocation accounting includes unpublished buffers through the existing allocated-other ledger.

## Observed short-screen result

Apple M5/24GiB, Qwen3.8-Flash-Next EXL3 average2.05bpw/checkpoint65c89531. AR32 on MAIN/HOLD, context4096, expert cache3GiB/48×53slots, four readers, dynamic0/lookahead0/prefixoff, temperature0/topk1/penalty1. Reference/candidate/candidate/reference, one warmup and one measured response per prompt/process. No compilation, prover or other inference overlapped the owned GPU campaign.

| Prompt | Decode reference → candidate | Decode change | First token, median | Complete response, median |
|---|---:|---:|---:|---:|
| MAIN |6.230027→6.343250tok/s|+1.817%|-3.109%|-2.764%|
| HOLD |6.261465→6.483449tok/s|+3.545%|-2.017%|-2.398%|

Both candidate decode samples exceed both references on each prompt. Reference decode drift is-1.028%/-0.594%; candidate drift-1.092%/+1.128%. All16warmup/measured outputs match in full nonempty text/hash and32generated/31decode counts. Metal allocation is6,608,617,472bytes in every measured row; process footprint/compression stay separate. All four processes and output readers joined. HOLD first-token samples overlap; do not claim uniformly faster TTFT. These are two measured samples per prompt/variant, not sustained or sparse-context results. No percentages are added to older experiments or the cancelled P04.

[Protocol](performance-protocol.json), [summary](performance-summary.json), [all samples and manifest](performance/campaign.json). The timed candidate is immutable CLId860f20b on runtime sourceb0dcf29d with the option explicitly1; referencef4b592bc has the same qualified earlier M5 defaults. AC100%, background apps active, temperature unmeasured; no swap abort. Current main28f1a84e is tree-identical to the c330f846 starting tree, so its merge introduced no extra source adaptation.

## Numerical and ownership evidence

[Quality02](quality-02/report.json) passes four separately selected physical checks, each with one real test or production CLI completion, no MLX and owned process groups empty:

- Four independent byte ranges, distinct live backings, exact1,248,012-byte record, EOF, post-enqueue truncation, delayed timeout, stale generation and batch/service drop. Ready-state metadata seeding forces real same-slot recycling.
- Expert cold/hot OFF/ON with exact finite outputs, identical routes and actual direct/copy counters.
- Target OFF/ON:57,606,144F32state words and744,960logits compared exactly,84independent initial-state hashes and continuation. OFF copies1,395,277,416bytes; ON delivers1118direct records with the same bytes and zero legacy copies. Model fixture context64; production CLI and speed screen use4096.
- Actual CLI4096 two-token golden9f850af1cbba074b,48×53ledger and no MLX.

The initial [quality01](quality-01/report.json) fails a fixture precondition: a partly empty cache does not force the same victim slot. The assertion was retained and a fresh cache with nine Ready entries forces the unique empty slot. Only the fixture was corrected; the original failed libtest443cebed and source identity remain archived. Corrected libtest e9b12a25 and CLId860f20b are separately identified in [release receipts](releases/iofix/README.md). No production change was hidden by this retry.

Initial strict direct Clippy, formatting/inventory, SSD8pass/1physicalignore, counter1pass and Python6pass/44Linux-only skips pass. Three actual CPU Kani helpers pass242checks/15covers over arbitrary extents, booleans, depths and valid UTF8 strings up to4bytes/unwind10; seven standard guards are unreachable and inspected. A copied-source shared-guard mutation fails while original/restored code passes. This checks the helper only: actual worker-premature-publication mutation was not executed, and CPU proofs do not establish Metal, OS or concurrency correctness.

## Final M5 default qualification

The selected integration changes only the absent-option policy: `MLXL3_NATIVE_SSD_DIRECT=0|1`, defaultON on exact `Apple M5`, OFF elsewhere; depth0 and no prefetcher are still required.0 rolls back. The explicitly enabled execution path, shaders, arithmetic, cache policy, budgets and read order match the timed candidate. No extra throughput campaign is attributed to this policy-only change.

Final source08a8aa5a passes all11CPU/build checks: strict Clippy in all three configurations, full CPU332passed/8ignored, direct348/51 and MLX384/269, no filtered tests. Final CLI d419a9ff and libtest1787abcd are separately frozen, minos26.2/sdk27.2. [Final release and CPU receipts](releases/final/README.md). The actual updated flag proof passes240checks/7covers over UTF8≤4/unwind10; with the unchanged extent/eligibility results, final CPU-contract evidence is242checks/17covers. The seven standard unreachable guards were inspected. Mutating the qualified default back tofalse fails; original code passes. [CPU evidence](qa-cpu/summary.json), [default proof](qa-defaults/report.json).

[Six final physical checks](final-quality/report.json) pass with `MLXL3_NATIVE_SSD_DIRECT` genuinely absent from every child environment. Four expert/model initial states aretrue before test overrides, and actual direct/copy counters remain strict. Real CLI4096 passes four independent goldens, rejection, context_usage/delta cancellations and recovery; the frozen budget integration passes two goldens and memory-saver recovery against the final CLI. AR-only prefix reuse passes two goldens with one cache hit. Predictive depth1 preserves the independent archive, publishes/drains prefetch and takes the legacy fallback. All six processes joined and process groups are empty; no extra inference or timing was run for publication. Native worker joins are not individually instrumented; normal teardown and the isolated lifetime cases provide runtime evidence, not a concurrency proof.

No MTP/DFlash, installed-app replacement, main merge or release is included. Sparse-context and sustained-speed qualification remain outside this short screen.
