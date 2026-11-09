# Issue 162 GPU reconstruction (PR 163)

## Preregistration, 2026-11-09

Base: published PR head 2ddde23, integrated main 0fd9fbd1, with concurrent
reconstruction changes. No historical measurements validate this checkout.
Read AGENTS.md, opti.md (including QWEN36-144 composition and M5 qualifications),
the current production implementations and issue 162 requirements supplied by
the coordinator. The formal-proof skill is absent from the cloud skill catalog
and inspected local skill paths; retain proof harnesses and attempt Kani anyway.

- P03: explicit opt-in Qwen profile with strict individual overrides, resolved
  once at model loading for all CLI/bridge consumers. Preserve device guards,
  dependencies and unchanged default policy. Expose effective flags. Desktop
  setting wiring requires coordination with its owner, not a silent default.
- P04: reuse four immutable GDN pipelines within a Metal owner for identical
  geometry. Source and compile options are immutable for that owner; no cache
  sharing between devices. Weights, recurrent state and snapshot identity stay
  per layer. Check cache hits and distinct geometry on physical Metal later.
- P05: reconcile existing PLE trace-only allocation implementation; no duplicate
  rewrite. Draft hidden captures belong to the native reconstruction worker.
- P06: extract the existing gate-dot encoder and predict from logits with top16
  in the same command, without softmax/top10. Preserve half conversion and exact
  demand-route implementation. Physical oracle compares legacy prediction and
  verifies command counts, routes and rounding.
- P07: retain an opt-in sorting-network router candidate with exactly the
  canonical float key and score reduction order, bounded padded geometry and
  unchanged fallback. CPU independent sorted-key oracle across E=1..1024,
  special values and k boundaries plus ignored physical F16/F32 tests. No
  promotion without physical numerical qualification and measurements.

Planned controls: scoped formatting/diff checks, CPU contract/oracle tests and
mutation, full CPU suite through coordinator, strict Clippy, targeted Kani
attempt, native coverage inventory review. Metal shader compilation, GPU parity,
model continuation, startup time, throughput and process memory are NOT measured
on this Linux cloud. No Apple build or benchmark claim; no push/merge/release.

## Implemented source and contracts

P03: `qwen_profile.rs` resolves all eight flags strictly, even flags whose
dependencies disable them. `MLXL3_NATIVE_QWEN_PROFILE=reference|optimized` is
read once by `Qwen35::load`, common to native CLI and bridge. Absent means
reference; explicit per-flag `0`/`1` overrides either profile. Optimized selects
the seven historically measured composition flags, NOT device greedy. Joining,
grouped-stage, device-greedy capability and context-bounded device-KV dependencies
remain enforced. CLI stderr records requested profile and actual flags; bridge
ready adds `qwen_profile` and `qwen_effective_flags`, retaining legacy fields.
Desktop wiring is in the separate runtime reconstruction report. This exposes
an experimental choice, not a newly qualified default or a speed claim.

P04: the already-validated Qwen loader geometry is passed to GDN directly, so
config is not read thirty more times. A cache on the owning Metal instance
contains four immutable PSOs per `[hidden,key_heads,value_heads,conv_length]`.
Device, embedded shader source and `fast_math=false` are immutable within that
cache's lifetime; epsilon is a runtime argument, not a source specialization.
Only successful compilation of all four PSOs is inserted. Layer weights, owner
identities, convolution and recurrence remain separate. Physical reuse/distinct
geometry/device/rejected-shape fixture added, NOT executed on Linux.

P05: PLE already contains trace-only allocations and stores plus the physical
trace-off continuation regression. It is preserved, not counted as new work.
Draft collection is owned by the native worker.

P06: `encode_dot` is the same arithmetic and paired-dot selector for demand and
prediction. Prediction now encodes dot and top16 into one retained command.
Demand continues dot, softmax and canonical normalized top10. The CPU half
conversion is unchanged. Physical fixture compares legacy prediction IDs/bits
with F16/F32 and half-conversion variants, exact ties/zeros/extreme finite inputs,
one submission and three allocations (plus optional conversion). NOT executed.

P07: `MLXL3_NATIVE_ROUTER_NETWORK=1` compiles/selects a direct-Metal bitonic
candidate. Absent/0 retains reference. Non-power-of-two widths pad to at most
1024; insufficient pipeline thread capacity falls back to reference. Padding
keys are strictly below all canonical finite/infinite/NaN keys. The key contains
expert ID, signed zeros compare equally, NaNs retain the old high ordering, and
scores keep ascending top-k dtype-specific serial addition/division. No fusion
with softmax, MLX router replacement or default promotion. Physical F16/F32,
strided-row, special-value and normalization fixture is retained, NOT executed.

## Verification results so far

Commands used `PATH=/home/agent/.cargo/bin:$PATH`, existing
`CARGO_TARGET_DIR=/tmp/pr163-native-target`, debug info disabled and incremental
off. Logs: `docs/measurements/pr163-recovery/gpu/`.

- `cargo test --locked --no-default-features --lib qwen_profile`: 2 passed,
  covering strict parsing, every one of 256 flag masks and four capability pairs.
- Same command with `router_network`: 1 passed in 1.44 s. CPU network simulator
  compared with independent float ordering for E=1..1024 and every k=1..E,
  including zeros, infinities, NaNs and subnormals. This is a finite CPU oracle,
  NOT execution or proof of the Metal shader.
- Isolated mutation of the production geometry helper to always return width1:
  rejected with exit101 at E=2; untouched production source rerun passed in
  1.46 s. Mutation never touched the shared checkout's runtime source.
- Both targeted `cargo kani --lib --no-default-features --harness ...` attempts
  failed before proving anything: `no such command: kani`. Harnesses retained
  for profile dependencies and bounded padded dispatch/XOR partner indices.
- Scoped rustfmt and diff checks passed. First strict all-target CPU/chat Clippy
  failed on concurrent N03's unused `ReadService::set_cancellation`, reported to
  its owner; no suppressions added here. After the owner's cancellation tests
  exercised that method, identical strict Clippy rerun passed in 6.72 s
  (`clippy-final.log`). This is Linux CPU/chat Clippy, not Apple type-checking.
- Full `cargo test --locked --no-default-features --features chat --lib`:
  381 passed, 10 ignored, 0 failed in 42.31 s (`cpu-suite.log`), on the integrated
  working tree at invocation. Subsequent unrelated worker edits require the
  coordinator's final aggregate check.

No physical Metal shader compilation, numerical parity, model continuation,
CLI/bridge model smoke, cold/warm startup measurement, throughput or process
memory measurement has run. P03 acceptance beyond portable selection and P07
profiling remain open. New GPU fixtures must pass on a Mac before qualification.
Integration status: code local; no push/merge/release or installed app changed.

This reconstruction's trials are closed: portable checks passed, formal proof
blocked by missing Kani, physical correctness/performance not verified. The
sorting candidate remains retained and unmeasured. It is not evidence that the
quadratic router was the bottleneck, and no historic percentages are added.

## Physical follow-up commands (not run here)

Run each selected ignored fixture alone on supported Apple hardware, followed by
the existing full Qwen/logit/state/continuation and predictive-SSD gates before
any benchmark or default promotion:

- `cargo test --locked --lib shared_gdn_pipelines_preserve_geometry_and_owner -- --ignored --nocapture --test-threads=1`
- `cargo test --locked --lib prediction_is_same_command_dot_top16 -- --ignored --nocapture --test-threads=1`
- `MLXL3_NATIVE_ROUTER_NETWORK=1 cargo test --locked --lib network_router_matches_reference_bits_and_strides -- --ignored --nocapture --test-threads=1`

Preserve raw failure logs and compare identical artifacts/options. The network
fixture exercises boundary geometries with exact reference bits; the CPU test
exhausts all E/k. Neither is a full-model performance result. Do not run GPU
measurements concurrently with compilation or other inference.
