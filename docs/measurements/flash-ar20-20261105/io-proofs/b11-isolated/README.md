# B11 isolated CPU guard follow-up

This parent-authorized follow-up reruns only the B11 harness that did not start in the preserved [first window](../report.json). The executable source remains `6e18b89a`; the source hash and unchanged full source copy are recorded in this report and `../ssd_reads.rs.txt`. The previous timeout and not-started result remain intact.

Kani 0.68.0 / CBMC 6.11.0: **40 checks passed, 0 failed, 4/4 cover properties satisfied**, one selected and completed harness. Total wall time 2.6353 s including compilation; solver 0.1760 s. The 30 s total budget was respected; owned process group 20925 no longer existed and all owned children were stopped/joined before root was notified.

The actual `readahead_eligible` and `readahead_identity_matches` helpers were checked using two symbolic booleans and two complete `[u64; 2]` device/inode identities, with no assumptions or stubs. The four covers exercise accepted admission, unqualified M5 rejection, replaced device and replaced inode. This is a source-level CPU helper property, not a proof of Darwin `fcntl`, independent descriptor effects, byte reads, Metal, concurrency or full-model parity. The physical ignored gate remains unexecuted by this agent.

No SLRU rerun/domain change, replay, test or GPU work was performed in this follow-up. Full stdout/stderr are retained losslessly in `guards.log.gz`; command, hashes and the scoped result are in `report.json`.
