# Current native sampling diagnostic (P00 first slice)

This cfg(test)-only observer runs on the PR119 native baseline, excluding the
stopped expert-encoder prototype. It preserves numerical output. It is a
sampling-focused P00 slice; it does not rank every engine bottleneck or report
production throughput. The unchanged production reference is f4b592bc.

MAIN87/context4096,3GiB/4readers/lookahead0/prefixoff, warm/control/observed,
8generated tokens/7decode rows. Old optimization variables are absent, using
real M5 defaults. The physical fixture passes once (1passed/0failed/0ignored),
49.25s process wall time, with identical IDs/fullhidden/logits and84continuation
states/28,803,072finite F32words. First-state and first-logit independent
archived hashes also match.7rows/336layers/7head+logprobs+CPUselect spans,
positive actual allocation counters,7head GPU intervals and noMLX are checked.
All owned children are joined and the process group empty.

## Observations from seven instrumented decode rows

| Observed scope | Mean per row |
|---|---:|
| Row host interval |165.091ms|
| Head host interval (inclusive) |3.806ms|
| Projection completion host wait |3.447ms|
| Head GPU command duration |3.241ms|
| Actual head vector copy |0.047ms /496640bytes|
| Canonical log-probability host interval |0.348ms|
| Its nested completion wait |0.239ms|
| CPU greedy selection |0.119ms|
| Native Metal allocation API envelope |11.296ms|

The head GPU projection remains necessary for P04. This observation gives no
basis for a large speed claim from removing the small copy/selection scopes.
The log-probability host interval includes GPU work/wait that a replacement
must still handle. Parent/child intervals overlap; the table is not additive
and does not estimate recovered time. All data are intrusive test timings.

There are39701 native Metal allocation requests/2,467,297,434requested bytes
in the observed rows, rather than that amount of resident RAM. Existing SSD
records show1,369,069,164demanded record bytes,3360route occurrences and2263hits.
CPU SSD wait is not isolated; these bytes are not physical NAND traffic.
Central-helper encode CPU coverage is partial. Completed-command GPU times
cover central/projection/MoE commands only. No devicewide utilization,
per-kernel time or CPU/GPU intersection is inferred. GPU/CPU epochs are not
aligned. The deduplication key is(address,start,end); addresses can be reused.

## Reproduction and validation

`freeze.json` gives source hashes, the frozen libtest83ddec44 identity and build
environment. `run-01/report.json` records exact physical argv, bounds and
controlled environment. Full event data and target-only stdout/stderr are
lossless gzip with hashes; raw originals remain in build/native8/raw-run-01.
There is no Instruments trace, global environment dump or other-app data.

Executed from this worktree, PATH includes /Users/justin/.cargo/bin:

```sh
export CARGO_TARGET_DIR="$PWD/build/native8/cargo-target"
export MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx
export MACOSX_DEPLOYMENT_TARGET=26.2
cargo clean --package mlxl3-native
cargo fmt --all
cargo clippy --locked --features chat --all-targets -- -D warnings
cargo test --locked --features chat --lib flash_profile:: -- --test-threads=1
cargo test --locked --release --features chat --lib --no-run --message-format=json
cargo test --locked --features chat
```

Strict Clippy passes; lifecycle6pass; full direct suite346pass/48ignored.
Inventory247 matches after updating exactly2test-only references. GPU tests
remain ignored in that CPU suite; the one physical selection ran separately.
A30s-bounded Kani attempt on the actual lifecycle in a disposable module copy
stopped with compiler ICE intrinsics.rs243 in2.90s before obligations; no
clock/Metal stubs were used. Property NOT VERIFIED; tests are not a proof.
See the receipt, log and retained wrapper. No redundant compatibility matrix
or production CLI rebuild was run for test-only hooks. Existing CI PR/main
triggers remain; exact pushed-head CI is for root to inspect, and unrelated
baseline proof failures must remain separate. No app or release changed.
