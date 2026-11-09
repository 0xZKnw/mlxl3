# PR167 — second audit implementation, 2026-11-09

Baseline: `0e84cbee62b3924939c3af9e0f35f57d922539a2`, branch
`codex/issue165-fast-native`. **Local, uncommitted implementation; not a new PR SHA.**
**Follow-up:** [runtime checks and speed](../pr167-reaudit-speed-20261109/README.md) now document the later MB3 dispatch fix and measurements. The compilation-only state below is the earlier handoff.

Request: implement `MLXL3_PR167_AUDIT_FIX_CODEX (1).md`; latest user steering
prioritizes writing the candidates together, with quality/proof/speed tests afterward.
This supersedes the original observe-first sequence for this local prototype only.

No new GPU inference, profiling capture or throughput experiment has run on this
revision. Previous measurements at the baseline do not validate these changes.

| Audit item | Code now present | Remaining / status |
| --- | --- | --- |
| Detailed collection | Scoped lazy MoE JSON/timelines and read timing, including worker collection flag captured at enqueue; clean throughput disables stage retention and detailed collection; diagnostic opts in | Source compiles; on/off numerical/state equivalence and worker/error paths unverified |
| Target attribution | Decode-only policy, shape/PSO, target/draft service counts, detail counts, chain and KV-fork deltas; optional decode diagnostic file; expert-axis observer | Capture and attribution analysis not performed; host intervals are not exclusive GPU time |
| MB3 isolation | Family/shape selection, native or NT1 geometry, optional 128-thread QMV PSO limit; MB2 remains default | Per-family correctness, resource/actual-dispatch checks and ablation not measured |
| Target scheduling | Optional bounded current-HC write → next-GDN projection command, using the existing owned segment and retirement path | Prefix/rejection/EOS/cancellation/late-error/continuation and lifetime checks unverified; full QSA/PLE grouping is not implemented |
| Q04 block overhead | Stack picker inputs, pooled shared picked-ID storage with retained scratch leases; sequential logical decisions preserved; counters distinguish prior-row substitutes | Physical scratch reuse, failure/cancel safety and speed unverified; full GPU logical-cache state machine is not implemented |
| Q02 | Separate test-only MUL1 deferred-affine and centered-affine profiles in the canonical tile body, per-split prepared-input correction; unsupported projection fallback rejected | Independent numerical oracle, AR/Bn/Hot/cold consistency, quality and speed unverified; existing H128/fusions reused, no new H128 algorithm |
| Depth / draft / KV | Optional D0–D4 complete-round-cost controller with calibration and periodic probes; optional draft publication guard; fork allocation/copied-byte/wait counters; first-consumer read attribution by retained/rejected rows | D0 retains and maintains draft state: it is not an unloaded-draft AR baseline. Controller/lifetimes/cancellation/acceptance checks unverified; GPU token-ID chaining and grouped repair remain follow-up |

Exact scheduling/metadata candidates and the new approximate Q02 profiles are
independent controls. Nothing here establishes a useful gain or a quality bound.
Q01 stays withdrawn. Production defaults and installed application were not changed.

## Controls

- `MLXL3_NATIVE_BENCH_DETAILS=0|1`: throughput collection, default off unless a diagnostic file is requested.
- `MLXL3_NATIVE_Q04_DIAGNOSTIC_OUTPUT=<path>`: measured decode diagnostic, requires detailed collection.
- `MLXL3_NATIVE_VERIFY_MB3=1` plus `MLXL3_NATIVE_MB3_SCOPE=all|resident|gdn-qkv|gdn-z|gdn-out|gate-up|down|consumers` and `MLXL3_NATIVE_MB3_NT=native|1`.
- `MLXL3_NATIVE_QMV_THREADS128=1`: restrict relevant QMV pipeline compilation to 128 threads, not unrelated larger-thread kernels.
- `MLXL3_NATIVE_BLOCK_HC_GDN_CHAIN=1`: bounded block command composition, default off.
- `MLXL3_NATIVE_Q02_PROFILE=exact|mul1-deferred|mul1-centered`: default exact, test binary only.
- `MLXL3_NATIVE_MTP_ADAPTIVE=1`, `MLXL3_NATIVE_MTP_DRAFT_PUBLICATION=1`: independent opt-ins, default off.

## Checks actually completed

- `cargo fmt --all`: passed.
- `cargo check --locked --tests --features chat`: passed after fixing two new
  `&&[MatrixView]` iterator errors in the GDN refactor. Initial failure retained
  in `compile.log`; successful affected reruns in `compile-r2.log` and
  `compile-r3.log`. This is compilation, not execution of tests or a release build.
- Apple Metal source compilation passed for `MLXL3_Q02_PROFILE=0,1,2`, each with
  `MLXL3_DEVICE_RESIDENT_ROUTE=1`; corresponding logs are retained. All three emit
  112 unused-variable/template warnings; no numerical property follows from this.
- Ruff formatting/lint of `tests/test_kani_ci.py`: passed.
- Reviewed source inventory refreshed: 309 files match. Changes are individually
  listed in `inventory-review.txt`; no exclusions added.
- `git diff --check`: passed before this report was written; final check follows.
- CI now compiles all three Q02 profiles in its existing macOS job on PRs and
  pushes to main. Not pushed or remotely run for this local revision.

Commands use Xcode `/Applications/Xcode.app/Contents/Developer`, Cargo from
`/Users/justin/.cargo/bin`, shared target
`/Users/justin/Documents/mlxl3-pr161-validation/target`, and the existing Python
virtual environment. No model was loaded or downloaded.

## Deferred validation

Tests for scoped/lazy collection, family selection and depth economics, plus
symbolic Kani harnesses for the actual CPU helpers, have been written but not run.
The helper obligations are registered in the existing Kani manifest; compilation
of ordinary Rust does not compile or prove `cfg(kani)` harnesses.
Strict Clippy, the native/Python suites, Kani attempts and physical component/model
oracles are deferred under the user's code-first instruction. No claim of proof,
parity, quality, GPU ownership safety or performance is made for this revision.

Next validation should first close the exact-candidate state/resource checks,
then independently qualify Q02 before speed comparisons. Keep one GPU workload,
freeze binary/source identities and preserve the baseline/interrupted evidence.
No push, merge, release or installed-app replacement was performed.
