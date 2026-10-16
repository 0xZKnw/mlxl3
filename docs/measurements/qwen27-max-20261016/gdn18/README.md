# Q27-18 — prepared API filter, not applied or executed

Only files below this temporary directory were written. The worktree and installed app were not changed. Before/after copies, source archive copies, exact patch, targeted GPU mutation and SHA manifest are present.

## Scope and call graph

The current prepare_qkv header body is factored into prepare_header without arithmetic changes. The new explicit prepare_and_step_with_gates API uses the own Rapid archive kernel, parameterized for Hk16/Hv32 or48/Dk128/Dv128/conv4/T1..4; validates shapes and FP16/FP32 dtypes; prepares the same compact RecurrentTape accepted by its existing replay.

The archived baseline gated_delta.rs and gated_delta_packed.metal are byte-identical to the current worktree before this preparation. The shader's new path is guarded by MLXL3_GDN_FUSED_PREP, defined only by this new API. Existing callers still use the stock macro-disabled path. There is deliberately NO Qwen caller or unused environment switch/TLS override before the micro filter. This is structurally OFF in production. After qualification, a separate minimal integration should reuse the historical MLXL3_GDN_PREP_RECURRENCE opt-in and test-local override, in a separately journaled trial.

Adapted Hv48 constants: convolution width10240, value width6144, state786432 FP32 values, launch[32,16,48], threadgroup[32,16,1]. Each value head now shares one key head with two other value heads. All floating arithmetic and F16 boundaries remain those of MLX0.32.2 and the local packed/prepare kernels. No weights, dtype, sums or sampler change.

Patch counts before formatting: contracts+65/-0, gated_delta+488/-43, shader+61/-2. Most additions are quality fixtures; the43 removed lines are the existing helper-header body moved unchanged to a function.

## Apply only after parent preflight and snapshot

From the worktree root:

    git apply --check /tmp/mlxl3-q27-max-20261016/gdn18/gdn18-api.patch
    git apply /tmp/mlxl3-q27-max-20261016/gdn18/gdn18-api.patch

Before applying, compare current hashes against manifest.json; preserve intervening edits and adapt hunks instead of replacing full files. The after copies are references, not instructions to overwrite unrelated work.

## Checks the parent must execute, with no GPU measurement concurrent

No formatting, compilation, lint, tests or proofs were run by the preparing agent.

    cargo fmt --all
    cargo fmt --all -- --check
    cargo test --locked --no-default-features --lib contracts::fused_gdn_rejects_unqualified_geometry -- --exact
    cargo kani --lib --no-default-features --harness fused_gdn_prepare_recurrence_has_bounded_launch_and_storage
    cargo kani --lib --features mlx --harness fused_gdn_real_array_api_preserves_output_shapes
    cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
    cargo test --locked --release --features mlx,chat
    cargo test --locked --release --features mlx,chat --lib --no-run --message-format=json
    cargo build --locked --release --features mlx,chat

Kani guard: six full-domain i32 symbols, no assumes, no loops; exact eligibility, launch512, convolution widths8192/10240, bounded prepared keys and states524288/786432; five covers (T1/Hv32, T1/Hv48, T3, T4/Hv48, rejectedT5). The real Array API harness uses mapped symbolic T1..4 and Hv32/48 without assumes or stubs; its historical Kani/FFI ICE may recur, which remains an unverified API, not a success. Attempt source MSL verification as required, retaining its real-source unsupported diagnostic, not a rewritten C proof.

## Numeric oracle before timing

Run the compiled test binary directly with a parent watchdog300s and bounded stop/join, one physical GPU job at a time:

    MLXL3_MTP_TEST_MODEL='/Users/justin/Library/Application Support/io.mlxl3.desktop/Models/Qwen3.8-27B-exl3-6a9ca9d0' \
    MLXL3_ALLOCATOR_CACHE_MIB=256 MLXL3_GDN_FUSED_MICRO=0 \
    <actual-built-lib-test-binary> \
    gated_delta::tests::fused_preparation_recurrence_matches_split_and_all_prefixes \
    --exact --ignored --nocapture --test-threads=1

Exactly one test must run and pass. Hv48 loads the real layer0 convolution/A_log/dt_bias from the supplied Qwen27B checkpoint. Hv32 is explicitly a synthetic historical geometry control, not revalidated old-model performance. Zero/nonzero conv/state, T1..4, retention false/true:32 output/state/conv comparisons;40 rollback prefixes checked against separate T1 transitions, plus independent prefix convolution states. Keys/delta/decay compare against the existing split implementation. Shapes/dtypes/counts/nonempty/finitude/bits are checked. T0/T5, malformed scalar/history, qkv/state dtype, short A_log and invalid replay prefixes are rejected.

The expected replay states are gathered from serial step_with_gates transitions, not another invocation of the same replay kernel. Both sides of every numeric comparison must be finite and nonempty. This is sampled physical testing, not a proof of Metal or universal numeric parity.

After application and before final positive controls, apply mutation-k-scale.patch to the prototype shader only, rebuild the test, and require this exact oracle to fail. Then restore the known shader bytes, refresh mtime as needed, rebuild, verify binary/SHA identity and rerun the affected positive oracle. Do not benchmark the mutant or stale cached test binary.

Also run the existing stock-path physical regressions alone after the shared shader change:

- gated_delta::tests::packed_step_with_gates_matches_graph
- gated_delta::tests::tape_replay_matches_every_serial_prefix

No model-level integration or generation throughput is validated by this API filter.

## Optional micro filter, only after positive numeric/mutation restoration

Use the same exact ignored test with MLXL3_GDN_FUSED_MICRO=1 under a parent watchdog300s. All32/40 quality comparisons run before the first timing. Four JSON objects appear, Hv32/Hv48 ×T1/T2; three warmups per arm, then40 pairs, AB on even pairs and BA on odd pairs. T1 has no retained tape, T2 retains the tape as MTP verification does.

Each timed call constructs/evaluates the actual split or fused preparation+recurrence; evaluates output/state/conv and retained keys/delta/decay, then synchronizes. Timings include host/MLX/GPU; they are neither GPU-only nor a token/s estimate. Fixed initial-state fixtures, no artificial dependent chain. Baseline and candidate use exactly the same inputs and parameter arrays.

The parent report must require exit0, one selected test and all four unique complete micro objects with40 finite positive times per arm before saying complete/parity true. Parse each sample matrix with the existing speed_gate: median>=3%, at least32/40 wins, both AB/BA medians>=3%, drift<=3%. Hv32 is only a control; only Hv48 qualifies this27B candidate. Partial JSON after later failure must remain failed, regardless of an individual quality_complete field.

If Hv48 is not useful and stable, archive and withdraw this API prototype. If useful, separately pre-register the production Qwen integration, TLS override, full128-state/cache/prefix/continuation checks, bridge cancellation/recovery/budgets and actual MTP1 ABBA/BAAB before performance promotion.
