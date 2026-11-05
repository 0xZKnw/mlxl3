# Current P01 baseline: HC/GDN attribution

This is one intrusive cfg(test) diagnostic, not production speed and not an optimization result. The unchanged P01 production CLI is d419a9ff. The native libtest is 4ef4d51884b4c2982415e91db5dd459acfd9d9fd67062dc25ff37b92d902a7e2 (9,286,384 bytes, minos26.2/sdk27.2), built from the dirty source hashes in freeze.json. Production arithmetic, shaders, state/cache policy and resource lifetimes were unchanged.

The exclusive root GPU grant ran exactly gpu::flash::tests::native8_sampling_profile_preserves_outputs --exact --ignored --nocapture --test-threads=1, bounded180s. Report.json records the full argv and relevant environment: Flash Next checkpoint65c89531/EXL3 2.05bpw, MAIN87/context4096/cache3GiB/4readers/dynamic0/depth0/prefixoff. SSD_DIRECT, the six previous optimization flags, P04 and the stopped expert encoder were genuinely absent. AC/thermal temperature/background-app CPU were not sampled for this diagnostic; other agents performed no builds/proofs/inferences during its exclusive slot.

Warm/control/observed8tokens/7decode rows match exactly, with finite logits/hidden/28,803,072 continuation F32 values and independent first-state/logit archives. One exact physical test passed in47.826s, no MLX loaded, process waited and PGID empty. 14,271 events cover679HC reads (96body+1head per row),2,716workspace allocations,252GDN bodies/frames. Each allocation is owned only by the innermost event; all three exclusive sums exactly equal the seven row counters.

| Scope | Requests/row | Requested bytes/row | Allocation API host ms/row |
|---|---:|---:|---:|
| HC layer workspace |384|8,110,080|1.266954|
| HC head workspace |4|84,480|0.011655|
| GDN frame |72|115,458,048|0.247720|
| GDN nonframe |720|9,075,456|1.447426|
| Global |5,671.57|352,471,062|11.084184|

HCread-before-GDN36CB/row: GPU4.593619ms, hostencode0.853477ms, hostwait9.955065ms. GDN36CB/row: GPU13.747232ms, hostencode2.125227ms, hostwait22.875435ms. Host scopes are inclusive and overlap GPU; they are not recovered-time estimates. The252consecutive completed-command timestamp pairs have median gap0.293812ms/pair and total11.207667ms/row, including CPU/scheduling/other activity. This is not measured queue idle, device utilization or a proven recoverable gap. HC chain CB durations include their existing gate. All2,343GPU records are valid and unique by (address,start,end); only156addresses exist, so address-only deduplication would be wrong. Metal command timestamps are host seconds per the installed MTLCommandBuffer.h286–294; no mapping to the CPU Instant epoch was recorded.

CPU commands used private CARGO_TARGET_DIR=build/hc-gdn/cargo-target, MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx and MACOSX_DEPLOYMENT_TARGET=26.2:

```
cargo fmt --all
python scripts/check_native_coverage.py
cargo clippy --locked --features chat --all-targets -- -D warnings
cargo test --locked --features chat --lib flash_profile:: -- --test-threads=1
cargo test --locked --features chat --lib flash_profile::native_allocations_belong_only_to_the_innermost_active_scope -- --exact --test-threads=1
cargo test --locked --release --features chat --lib --no-run --message-format=json
cargo fmt --all -- --check
```

Python was exactly /Users/justin/Documents/mix-stq1_0/.venv/bin/python; cargo was /Users/justin/.cargo/bin/cargo. StrictClippy and7tests pass, inventory249 matches. Four inherited P01 inventory gaps versus exact baseline038abd98 and seven new fixture flag mentions are recorded separately in qa/inventory-before.json; only explicit affected entries were updated. A temporary real-source mutant also attributed inner allocations to the parent: the focused regression failed101; restoring the original passed and restored its source hash. No redundant compatibility matrix or production CLI build was performed.

The new actual-counter Kani harness uses four symbolicu64 inputs, two nested real spans, unwind33, no assumes and no clock/Metal stubs. The30s-bounded attempt completed3.233s with Kani0.68 compiler ICE intrinsics.rs243 before obligations. Property NOT VERIFIED; no covers or proof passed. Only test attributes/module exposure were adjusted in the disposable current-native copy; actual counter/harness bodies remained unchanged. The exact command/diagnostic are in qa/kani-counter-report.json and its compressed log. The old lifecycle ICE was not rerun. Metal/OS/concurrency remain unproved; worker threads were not individually instrumented as joined.

safe-summary.json is the safe complete aggregate. Full cfg(test) events and own-process stdout are losslessly gzip archived with hashes and private raw paths in manifest.json; no Instruments/global environment was collected or published. No P03/P13 prototype, production speed comparison, merge, installed-app replacement or release is validated here.
