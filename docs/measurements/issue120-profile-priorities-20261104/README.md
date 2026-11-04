# Priorities selected from the current native Flash AR profile

The next experiment is **P01: read demanded expert records into fresh owned shared Metal buffers**, removing the intermediate initialized Vec and subsequent copy. P04 was cancelled by the user and is excluded from this branch. This ranking follows the measured workload, rather than the issue's numbering.

Evidence: [P00 sampling-first profile](../issue120-native8-profile-20261104/run-01/safe-summary.json), MAIN87/context4096,7decode rows on Apple M5/24GiB, EXL3 2.05bpw,3GiB/4readers/depth0, no prefix reuse. Instrumentation is intrusive and CPU/GPU epochs are unaligned. These are host scopes, not recoverable-time estimates or additive exclusive costs.

| Priority | Candidate | Evidence and decision |
|---|---|---|
| First experiment | P01 record destination |1097copies ×1,248,012bytes =1,369,069,164bytes;89.136489ms total,12.733784ms/row,99.7% of measured upload time. Exact removable duplicate identified in source. Main-thread preallocation and I/O ownership may offset benefits; measure end-to-end.|
| Secondary | P03 workspace reuse |11.295877ms/row covers all native Metal allocations, not HC alone. A concrete HC cache could avoid384requests/row after warmup with≈8.11MB capacity, but its time is unmeasured. GDN frames belong to snapshots; QSA already has slabs/COW.|
| Targeted diagnosis before implementation | P13 HC→GDN scheduling |36pure-GPU boundaries/row are structurally joinable.71.435ms central wait and24.055ms MoE wait are inclusive envelopes, not savings. Changes ownership/error publication; isolate current HC/GDN scopes first.|
| Conditional | P02 SSD queue/tail |Last-read end p50≈0.885ms/p95≈1.838ms from each MoE epoch; no enqueue timestamp or worker ID.35/336layers have no miss, only36/336have≥7. No causal queue bottleneck established.|
| Defer pending kernel attribution | P08/P09/P11 |Current profile does not isolate their kernels. Historical GDN/HC composite GPU groups predate recent optimizations. P09 also changes tiling/register pressure; P11 mostly removes one small launch.|
| Different workload | P05/P14/P18/P19/P20 |Sparse QSA starts at2052tokens; this short prompt does not exercise it. Startup, multirow prefill and TensorOps need their own workload and cannot explain this decode screen directly.|
| Excluded now | P04/P21–24 |P04 cancelled after a drifting, inconclusive speed screen. MTP/DFlash remain explicitly deferred.|

The P01 trace decomposition also records read-duration p50≈0.597ms/p95≈0.971ms, including allocation/initialization and the read call. Those are not NAND latency or isolated queue service times. The11.3ms allocation scope must not be added to12.7ms upload as promised savings.

The selected prototype uses one existing bounded read service. A newly allocated exact-size Shared buffer moves exclusively into the read job and returns only after a complete successful read. No consumer sees a view before generation validation/publication. No buffer pool or reuse is introduced; abandoned/timed-out jobs retain their owners until physical completion. `metal-rs0.33` already declares owned Buffer Send+Sync; this does not authorize concurrent writes or a generic unsafe Send for tensor views. [Apple's shared storage contract](https://developer.apple.com/documentation/metal/mtlstoragemode/shared) still requires synchronized CPU/GPU access.

Quality, actual-path/copy counters and failure/timeout ownership precede the requested32-token AR screen. No P01 speed gain, correctness result, app installation or publication is claimed yet. See `opti.md` entry I120-P01-OWNED-DESTINATION-20261104 for the preregistered scope and protocol.
