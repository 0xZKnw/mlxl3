# PR167 — second audit speed pilot (2026-11-09)

Baseline `0e84cbee62b3924939c3af9e0f35f57d922539a2`, branch `codex/issue165-fast-native`.
Frozen release test binary SHA-256: `505aac5bffe851ed057754f708361a5fdb3390bb662dc8adf83626311157fe4b`.
Measured source hashes: [identity.json](identity.json). The final CPU-only compile
restriction is identified separately in [delivery-source-delta.json](delivery-source-delta.json).
The timed binary was not rebuilt after that non-GPU guard change; CPU and Metal
compilation/lint were rerun. No application/default/merge/release change.

## Results

| Configuration | Decode tok/s | Draft acceptance | Accepted/proposed | Decode seconds | Responses |
|---|---:|---:|---:|---:|---:|
| AR exact | 10.380 | — | — | 6.070 | 2 |
| D2 MB2 | 8.977 | 69.2% | 72/104 | 7.018 | 2 |
| D2-details-on | 8.814 | 69.2% | 36/52 | 7.148 | 1 |
| D2-chain | 9.605 | 69.2% | 36/52 | 6.559 | 1 |
| D2-chain-mb3-qkv | 9.722 | 69.2% | 36/52 | 6.480 | 1 |
| D4-chain | 9.039 | 51.9% | 42/81 | 6.970 | 1 |
| D4-adaptive-publication | 9.454 | 58.5% | 38/65 | 6.664 | 1 |
| D1-block-q20-chain | 10.406 | 77.1% | 27/35 | 6.054 | 1 |
| AR-q02-deferred | 9.836 | — | — | 6.405 | 1 |
| D2-q02-deferred | 8.697 | 69.2% | 36/52 | 7.244 | 1 |
| AR-q02-centered | 9.455 | — | — | 6.663 | 1 |
| D2-q02-centered | 9.109 | 69.2% | 36/52 | 6.916 | 1 |

D2 HC→GDN composition: **+7.00%** versus the current D2 MB2 control.
With family-scoped MB3/NT1/128-thread QMV compilation: **+8.30%**.
The incremental family/NT/compiler effect is only
+1.22% with one response; it is not established independently.
Neither exceeds current exact AR throughput. D1/Q04 uses a different depth and an
approximate policy, so its percentage versus D2 is not a single-change gain.

Q02 deferred: AR−5.24%, D2−3.12%; centered: AR−8.91%, D2+1.47%.
These prototypes have no demonstrated useful gain as a whole. Retain them as
optional research, with model quality explicitly deferred by the user.

## Protocol

Apple M5 /24GiB, AC power, no thermal/performance warning recorded by pmset.
Qwen3.8-Flash-Next EXL3 ~2.05bpw/head4/MUL1, existing checkpoint artifact
`d6c73ad963bf3f23`; LFU53 /3GiB caches,4 readers, resident routing on,
HC workspace off. Same47-token MAIN prompt, context capacity4096, final active
offset110. Each fresh process generates64 warmup outputs then64 measured outputs,
with63 timed decode steps. Model load is outside response prefill/total time.
Raw reports and summary separate prefill/TTFT, decode, total, Metal allocation and
process footprint. Service bytes are not unique disk reads or NAND traffic.

There are14 arms /28 trajectories. Repeated AR/D2 controls have2 measured responses;
every candidate has1. All28 match the appropriate AR policy/profile reference;
27/28 also match exact AR. Both Q02 profiles happen to produce exact-AR words
on this prompt. This does not establish their quality or logits/state parity to
the exact model. Q04 remains its existing approximate20% policy.

Control drift: AR-1.425%, D2+0.259%.
This is a short single-prompt pilot, not a sustained or long-context speed claim.
Do not add historical percentages or promote the small centered-Q02 D2 signal to
a validated gain. Q02+chain composition and exhaustive family ablation are unmeasured.

## Validation and preserved failures

- Release library375 passed /139 ignored. Three new ordinary CPU tests pass.
- Kani0.68 /CBMC6.11:2 actual helpers pass. MB3 family76 checks and2 covers;
  depth481 checks (6 unreachable),1 cover, unwind8. Costs/options/limits/rounds
  symbolic; returned depth never exceeds budget or4. These are bounded CPU
  checks, not Metal/MTP/allocator/concurrency proofs. Initial unstable CLI export
  failure is retained; retry adds the required option.
- Eight explicit physical controls pass: independent F16/F32 picker; ordinary
  MB2 projections; corrected family MB3/NT1/128-thread projections; deferred and
  centered Q02 scalar/B2–B4 projections; expert-consumer failed-fill recovery;
  baseline and composed full-model prefix/state controls. Each projection run
  compares11,538,432 finite FP16 words. Q02 compares its own profile across
  scalar/block execution; independent affine-error and task-quality oracles remain pending.
- Full-model controls compare target hidden/GDN/QSA and draft states, logical
  cache state, forced rejection ranks, EOS, final cancellation, rollback and resume.
- Initial family MB3 control detected a dispatch bug: scoped PSO selection still
  used global MB3 presence for dispatch Y. Unselected MB2 B3 paths omitted row3.
  Dispatch now uses the same per-call predicate as PSO selection. Failed binary
  identity/log/status retained; affected fixture and full-model controls rerun.
  Effective NT observation was also corrected.
- Q02 admission rejected unsupported K7/8 fixtures before the old post-load skip.
  Retry retains all42 nongeneric K2–6 fixtures, the original test's domain, with6
  exclusions/reasons saved. No runtime guard was removed. The timed model has
  no MUL1 K≥7 projection. This admission failure is retained separately.
- Timed runs assert64 nonempty outputs, real work/geometry,63 retained target
  rows, active offset, read drain, service-counter sums and zero detailed events
  in clean mode. MB3 and HC-chain actual use is checked. Every child is bounded,
  joined and monitored for competing builds/provers/engines; no series interruption.
- Final strict Metal/CPU Clippy and formatting pass. The first CPU Clippy failure
  (two GPU-only accessors) and its platform-guard fix remain explicit.
  CPU strict command: `cargo clippy --locked --no-default-features --features chat --lib --tests -- -D warnings`. A broader no-chat integration-target attempt additionally failed because existing OpenAI transport code requires the chat-gated vision_input module; its diagnostic remains in cpu-clippy-r2.log.
  Focused Python6 passed /44 Linux-only skipped /2 subtests. Prior broad suites
  remain historical; no new full Python suite, quality campaign or CLI/app qualification.

## Reproduction and remaining work

`cargo test --locked --release --lib --features chat --no-run --message-format=json`
builds the test executable. Ordinary run uses `--test-threads=1`.
Kani commands/status/output are in `kani-r2-status.json`, `kani-r2.log`, `kani-r2.json`.
Physical commands/settings are in `physical*.py` and `physical-status*.json`;
throughput is in `matrix.json`, `speed.py`, `speed/status.json` and each raw report.
Xcode `/Applications/Xcode.app/Contents/Developer`, Rust1.98.1, existing Cargo
validation target and Python3.12 environment; no download.

No Q02 quality approval is implied. Existing H128/fusions are reused; full GPU
policy transitions, GPU token chaining, full QSA/PLE grouping and grouped KV repair
remain follow-up. Detailed capture and production CLI/bridge/app qualification
remain unmeasured. This delivery pushes research code/results only.
