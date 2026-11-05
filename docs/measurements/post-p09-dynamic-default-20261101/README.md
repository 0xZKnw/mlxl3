# Dynamic + Cold U enabled by default — 1 November 2026

Requested and accepted by Justin, including the measured +0.700% short-MTP
prefill time. Dynamic now defaults on in the shared synchronous/asynchronous
C++ parser and Rust loader. Cold U was already on. Explicit
`MLXL3_SSD_DYNAMIC_READERS=0` still selects Static; invalid values still fail.

This is local engine code, not an installed app update or publication.
The prior [performance campaign](../post-p09-dynamic-cold-20261101/README.md)
remains unchanged: aggregate short decode +0.535% AR, +0.960% MTP,
+1.030% DFlash. No speed campaign was repeated for this default-only change.

`identity.json` pins sources, plans, the release CLI (`3e41cd52…`), libtest
(`341ccbdc…`) and Python. All identities were checked before closure.
Every numbered plan records exact commands, options and fixtures. Run plans
through `decode-profile-p09-20261101/run-mtp-head.py`, with both
`MLXL3_SSD_DYNAMIC_READERS` and `MLXL3_SSD_COLD_READY` unset in the parent.
The unchanged authorized MTPHEAD anchor and 32 MiB write / 256 MiB read swap
limits were retained. No new swap writes occurred during the 13 GPU processes.

Checks inspected:

- Queue tests: 4,096 policies and 4,084 concurrent cases under ASan/UBSan and TSan.
- `cargo fmt --all --check`; strict Clippy with `mlx,chat`; release CLI/libtest builds.
- CPU suite: 276 passed, 9 ignored. MLX/chat suite: 311 passed, 203 ignored.
- Ten physical selections: absent/0/1 scheduling and invalid flags, Cold waves,
  preparation/fences, errors/recovery, independent target states/logits, MTP
  state/continuation and DFlash prefix rollback/captures.
- Three real CLI bridge sessions, 17 outputs on MAIN/HOLD in AR/MTP/DFlash,
  with both flags absent: all 12 warmup/measured responses match external gold;
  all children and reader threads joined. See `qualification.json`.

`cargo kani --lib --tests --features mlx,chat --harness
actual_cold_ready_refinement_entry --jobs 1` failed in the known compiler ICE,
before obligations (6.68 s, peak 896 MB). CBMC on the updated actual queue
proof failed parsing system headers. Formal properties remain unverified;
tests and sanitizers are not complete proofs. Existing CI covers the affected
checks, but no remote run exists for this unpublished local diff.

Operational failures retained: initial build used the package name as the
binary target; corrected to `--bin mlxl3-rs`. The initial ad hoc result
inspector forgot to map saved `text` to `cache_context`; corrected without
changing engine outputs or rerunning measurements.
