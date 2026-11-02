# Prepared MLX evaluation — T01B / POSTP09-PREPARED-EVAL-01

Status: closed on the user's request to change direction. The experimental
path remains off by default. Component/state qualification passed; the AR
pilot was unfavorable. MTP/DFlash speed qualification is incomplete.
Dynamic + Cold U remains enabled independently. No benchmark remains running.

The prototype changes only nonresident `mlxl3_rs_expert_*_slots1` custom
dispatches. `MLXL3_PREPARED_METAL=1` enables it in a fresh process. The existing
128-entry factory cache retains one metadata/pipeline plan per entry, keyed
by the input dtype, scalar status, constant/device address class, output dtype
and device. Factory identity already includes source, header and argument names.
There are no captured input arrays or shared scratch buffers in the plan.

The first evaluation delegates to MLX's original primitive and retains that
exact pipeline. Later contiguous calls allocate outputs through MLX, bind
the current arrays and shape metadata, and dispatch through MLX's command
encoder. Noncontiguous inputs delegate to the original primitive. Array
dependencies, barriers, allocation, stream, shader and rounding remain intact.
The original factory and primitive are still constructed on each call: this
first prototype isolates repeated evaluation work, not the entire graph cost.

This deliberately depends on the pinned MLX 0.32.2 internal ABI. The bridge
has a version static assertion. Some internal helpers and CustomKernel RTTI
are not exported; the cast is limited to fresh results from that exact factory,
with its primitive name checked. No global MLX installation was patched.

Primary sources inspected: MLX tag v0.32.2, tree
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`,
[evaluation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/custom_kernel.cpp),
[signature generation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/common/metal_kernel.cpp),
[pipeline/library lookup](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/device.cpp).
The evaluation source is saved in this directory. Apple Inc.'s upstream MIT
implementation supplies the binding order; the shaders are unchanged.

## Verification and the defect caught before timing

The C++ test compares 72 combinations / 144 outputs with both the original
MLX primitive and an independent scalar oracle, including Float16/Float32,
1–32 rows, strided views, constant/device boundaries, pending graphs across
plan replacement and reverse sibling evaluation. Scalar arguments and invalid
launch dimensions/options are covered. A deliberately wrong ndim binding
fails the oracle; the mutation never touched production source.

The initial ASan/UBSan process returned zero but emitted null-member-call UB
from `NS::SharedPtr` releasing a null Metal pointer. It is **not a passing
sanitizer result**. R2 uses a unique owner with a nonnull release deleter;
`asan-r2.log` passes with `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`
and contains no sanitizer diagnostic. Initial logs and identities are retained.

The FFI cache test now also exercises the prepared path with 320 keys / 330
requests. Existing physical Cold/failure/recovery, independent target state,
MTP and DFlash tests are reused; only completed inspected results count.
Kani on the actual Rust slot-oracle entry failed in the known compiler ICE
before obligations (8.42 s, peak ~893 MB). CBMC on the C++ source failed to
parse platform headers. These paths remain formally unverified; sampled
numeric tests and sanitizers are not complete proofs of Metal or concurrency.

## Planned component timing

Same release binary, prepared flag 0/1, order A-C-C-A. Each process checks
48 GU/down dtype/stride cases against independent single-row EXL3 projections.
Timed contiguous Float32 input chains use 1/2/4/8/16/32 routes, two GU
projections per route or one down projection. Each shape has 8 warmups and
8 batches of 32 evaluated calls. Build, async submission and final wait wall
times are recorded separately; output readback/parity occurs outside timing.
These are full projection chains without SSD, not pure GPU times or model
tok/s. No inference of a 12 ms saving is justified.

The original authorized MTPHEAD anchor, 32 MiB cumulative swap writes,
256 MiB reads, AC power, memory admission and quiet guards remain in use.
All commands/options are in the plans; identities pin sources and binaries.
No publication, installed app change or activation of this experimental path.

## Component results (complete)

R2 passes 311 ordinary MLX/chat tests (204 ignored, including physical tests
selected separately), format, strict Clippy and release build. All seven R2
FFI/cache/Cold/error/target/MTP/DFlash selections pass. The model oracle observes
177 natural Cold waves. The sanitizer test and 4 micro processes are complete,
with 192 dtype/stride oracle cases and 12,288 measured calls across 384 batches.

The micro sometimes shows a few microseconds less submission wall time, but
large total-time differences are driven by unstable wait intervals. For example,
GU/4-routes takes 264.8 and 520.0 us in the two A processes versus 265.8 and
274.0 us in C. Down/4-routes is 515.6/245.5 us in A versus 240.4/225.0 us in C.
Do not label those apparent 31–39% average reductions as a launcher speedup.
Down/1-route is +10.23% on the aggregate; its C processes are 209.1/252.0 us
versus A 207.8/210.5 us. No sample was dropped. `micro-analysis.json` retains
every batch, phase and process.

Adaptation audit: same pipeline, grid, threadgroup, binding order, output
allocator and MLX barriers; no shader math or SSD change. One extra primitive
and shared owners are built on each call, and the original factory is still
invoked. Thus the prototype can only remove a subset of the previously profiled
submission work. Model timing is required; the micro alone cannot settle it.

The first ad hoc parser missed a marker prefixed by the libtest test name and
stopped on 95/96 records. The corrected parser requires exactly 96 distinct
records per process and finite positive timing values. Measurements were not
repeated or edited. Also see `identity-metadata-erratum.json`: three descriptive
fields were inherited from an older campaign; recomputed source/binary hashes
were correct. `identity-final.json` fixes those fields and additionally pins
the installed MLX dylib and ABI header for the 128-output pilot.

## Model pilot (closed, partial matrix preserved)

All six AR/MTP/DFlash A/C bridge17 processes completed, with 24 warmup/measured
responses checked against external gold and children joined. They showed mixed
short timings and are not treated as speed evidence. The pre-registered pilot
uses 128 outputs / 127 decode transitions, MAIN/HOLD, one warmup and measured
response per prompt, A-C-C-A in each of the three modes. The same CLI R2 and
default Dynamic+ColdU apply to both arms; only `MLXL3_PREPARED_METAL` differs.
All plans are in `speed-manifest.json`. No automatic long/confirmation campaign
or activation follows an ambiguous result.

The completed AR A-C-C-A pilot gives 11.16049 → 10.96964 tok/s (−1.710%),
and mean prefill time 7.65568 → 8.09237 s (+5.704%). MAIN is −0.643% decode /
+2.014% prefill time; HOLD −2.759% / +9.656%. See `pilot-AR.json` and the
individual reports. The first A4 process stopped on a live power guard before
any complete response; the separately preregistered A4 retry completed with
the same binary, flags and resource anchor. Both attempts are retained.

MTP A1 completed. MTP C2 was interrupted when the user redirected the work:
the controller received SIGINT, cleaned up and joined its child (exit 130).
Its partial report has two warmups and one measured response; it is not a
completed comparison. MTP 3/4 and all DFlash128 pilot processes were not run.
The initial stop command refused a case-sensitive Python-path assertion;
the controller identity was rechecked before the successful stop.

The adaptation audit found no additional concrete binding, barrier or shader
defect after R2. The prototype still constructs the original factory/primitive
and adds a wrapper; it does not eliminate per-node MLX scheduling or dispatch.
Compression and background activity varied during the pilot, so no exclusive
cause is assigned to the timing changes. The unfavorable AR result does not
disprove preparation of whole chains; it does not justify enabling this port.
No new speed result is claimed for MTP/DFlash and no installed app was changed.
