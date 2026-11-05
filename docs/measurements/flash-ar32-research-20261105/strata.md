# Latest Strata: no new engine delta since AR30

Source-only audit, 2026-11-05, under AR32-RESEARCH-20261105. Local comparison: qualified MLXL3 82fff5a54c91efd2a3f3d34f75a83aa5251785cc, accepted thirteen defaults ON and experimental AR20/AR30 options OFF, plain AR. Read opti.md, AR20 SSD/matrix/negative histories and AR30 runtime research first. No prototype, clone, model download, test, compilation, prover or benchmark; only this report was written. Cancelled profiling stays cancelled.

**Conclusion:** the latest stable Strata release targets the exact commit already studied in AR30. Main is three commits newer, with **zero inference-code changes**. There is no newly released Metal/SSD/AR kernel to import and no newly established local boost.

## Verified pins and delta

Primary GitHub API and release HTML checked; main HEAD rechecked unchanged at the end, approximately16:29 UTC.

| Identity | Verified source |
|---|---|
| Current main | [9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e](https://github.com/Niko1221/Strata/commit/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e), committed2026-11-05T13:54:54Z |
| Prior AR30 pin | [c3b3f5035506f3e966f1b4b7b188aeaabae92f1d](https://github.com/Niko1221/Strata/commit/c3b3f5035506f3e966f1b4b7b188aeaabae92f1d) |
| Latest stable | [v0.2.11](https://github.com/Niko1221/Strata/releases/tag/v0.2.11), published2026-11-05T12:49:59Z; app/Terminal0.2.11, engine/Control0.1.57 |
| Release tag | [Annotated tag4a4160162e4a402e082f41aa0a88e6eed629d913](https://api.github.com/repos/Niko1221/Strata/git/tags/4a4160162e4a402e082f41aa0a88e6eed629d913) targets **c3b3f503…**, our existing pin |
| Shipped Mac files | Tag annotation identifies [3e0210663b07deafce51310ae622862c2fe05901](https://github.com/Niko1221/Strata/commit/3e0210663b07deafce51310ae622862c2fe05901); later launch-plan changes concern non-Mac UMA PCs. Release tag, current source and package build are distinct identities. |

The [exact compare](https://github.com/Niko1221/Strata/compare/c3b3f5035506f3e966f1b4b7b188aeaabae92f1d...9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e) is ahead3/behind0, eleven files:

- dc9dfcfd290afe3e42363d739c5904cf4ae5fa05: funding metadata and sponsor links.
- 5d37280db1f70ef6b55d69ac1d9ad8630d7a62f0: Docker copies two existing preparation scripts into its context, with a packaging test; those scripts and engine math are unchanged.
- 9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e: README supporters.

No changes to stream.cpp, window.cpp, arena_budget.h, mtl_rt.mm or AR shaders.

## Performance evidence does not establish a new Mac AR gain

Release speed work uses branch/package-like builds and was not fully rerun on shipped0.1.57 packages. New Mac quant support lacks a real-model Mac speed run. Large prompt gains are CUDA/HIP prefill; CUDA K-quant explicitly leaves decode unchanged. FP16 weight caching is Linux NVIDIA multi-request/MTP, five paired interleaved blocks, about3.8–6.6% aggregate improvement, explicitly absent on Metal. Radeon route-tail skipping reports nonzero KL and changes executed experts, incompatible with our exact AR contract. These are upstream author attestations, not local EXL3 measurements. [Release protocols and limits](https://github.com/Niko1221/Strata/releases/tag/v0.2.11).

The older M5 Max/Q2_0 spin comment reports about5% with matching tokens but lacks complete protocol details in the code. It already motivated B10; its20ms budget is not proposed again. [run_graph](https://github.com/Niko1221/Strata/blob/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e/metal/src/mtl_rt.mm#L1009-L1036).

## Two historical mechanisms, ranked honestly

**1. Conditional reserve: cached CPU dispatch descriptions.** Strata caches a shape/head/state-dependent dispatch vector, updates QSA positions/grid, then executes it. Real callers: [Window cache hit/build1380–1429](https://github.com/Niko1221/Strata/blob/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e/metal/src/engine/window.cpp#L1380-L1429) → [run_ds1546–1548](https://github.com/Niko1221/Strata/blob/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e/metal/src/engine/window.cpp#L1546-L1548).

Native gpu.rs:1063/1161 and A06 projection.rs:1490–1520 reconstruct bindings/cohort descriptions. C04 caches only completed resident scratch, stayed neutral and is OFF. A minimal future description cache could cover one fixed resident segment while preserving existing encoders/commits. It must rebind current owned views and positions, validate all bindings before emitting, invalidate on shape/flags/KV growth/reset/shrink, and never retain stale state or returned outputs. Require independent numerical/state/rollback checks and actual cache-hit counters before timing. This is **already AR30 runtime R5**, with no new isolated speed attestation, not a fresh optimization justified by this release.

**2. Exclude direct port: segmented GPU graph with CPU I/O handshakes.** run_ds → [run_graph_seg1050–1083](https://github.com/Niko1221/Strata/blob/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e/metal/src/mtl_rt.mm#L1050-L1083) → [at_cut496–498](https://github.com/Niko1221/Strata/blob/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e/metal/src/engine/window.cpp#L495-L498) → [ExpertStream::at_layer619–713](https://github.com/Niko1221/Strata/blob/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e/metal/src/engine/stream.cpp#L619-L713). It pre-encodes a CB with shared-event cuts, pins the selected cohort and waits for demanded reads before continuation. Read failures exit the process.

Our ready-cohort path already overlaps shared/hot compute and can dispatch an early wave before the slowest miss. Precommitting a GPU wait behind uncancellable pread adds the previously identified timeout/lifetime risk. The safer uncommitted single-miss reserve was already documented in AR20; no new evidence warrants reopening it.

## Deduplication and decision

Contiguous records, quiescent backing reuse and isolated read splitting already exist as A11/A12. Mixed-size GGUF cache classes do not remove waste from our homogeneous1,248,012-byte EXL3 records. Depth-zero demand has no missing urgent-versus-prefetch priority class. Global cache/common-queue histories remain qualified negatives; B12 increases MAIN/HOLD decode misses1.7674%/1.4473%. Shared-encoder barriers remain the cancelled P04/same-expert territory, not a new trial. F_RDAHEAD is already B11; old F_NOCACHE losses remain. Sources: [classes124–162](https://github.com/Niko1221/Strata/blob/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e/metal/src/engine/stream.cpp#L124-L162), [read277–306](https://github.com/Niko1221/Strata/blob/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e/metal/src/engine/stream.cpp#L277-L306), [prior runtime research](../flash-ar30-20261105/research-runtime.md), [SLRU audit](../flash-ar20-20261105/slru-audit/README.md).

**No fresh engine candidate from the latest Strata delta.** Retain item1 only as an old conditional reserve. No MTP/DFlash, P04, HC L3/L5, new profiling or promised11–12tok/s boost. A future implementation still needs its own issue/matrix/opti preregistration and correctness gate.
