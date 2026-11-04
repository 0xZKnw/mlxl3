# Issue 125 L1 — cross-layer HC write/read/GDN

**Qualified locally and enabled by default on exact Apple M5.** A completed MoE branch's final HC write can be encoded with the next regular GDN's HC read and attention body. This preserves the existing P13 read/GDN join and removes another 34 command buffers and waits per row. Shaders, arithmetic, routing, requested allocations and SSD scheduling are unchanged.

The pending value owns streams/branch/gate, stays within `FlashNext::hidden`, and returns the newly written streams to the consumer residual. It never enters snapshots or prefix state. PLE/QSA consumers, public component APIs, trace/capture exports and lookahead retain synchronous completion. The whole-token transaction restores producer and consumer states after a deferred error. The three old inputs remain retained through the consumer residual (~51,216 B in F32); this is bounded to one boundary, not accumulated over 34 layers. No further cleanup variant was measured or selected.

Set `MLXL3_NATIVE_CROSS_LAYER=0` to roll back. Absent means ON only for `Apple M5`; strict explicit `1` also keeps the hardware guard. The P13 flag must be active. Other hardware, exports, lookahead and non-GDN consumers retain the existing path.

## Short AR32 result

Apple M5/24 GiB, checkpoint `65c895314393431c09050b2e04e250836b3a6eb4`, EXL3 average 2.05 bpw. Context4096, cache3 GiB/48×53 slots, four readers, dynamic0/lookahead0/prefixoff, T0/topk1/penalty1. One warmup and one measured response per prompt/process; A/B/B/A. All 16 full warmup/measured outputs match, 32 generated/31 decode tokens, all processes/readers joined. AC100%, no thermal/performance warning recorded; temperature unmeasured. No owned compilers, provers or other inference overlapped GPU runs.

| Prompt | P13 reference samples (tok/s) | L1 samples (tok/s) | Median before → after | Change |
|---|---|---|---|---|
| MAIN | 6.25255, 6.16877 | 6.53622, 6.61763 | 6.21066 → 6.57693 | +5.897% |
| HOLD | 6.37753, 6.38345 | 6.50470, 6.63336 | 6.38049 → 6.56903 | +2.955% |

All candidate samples exceed all controls. Control drift MAIN−1.340%/HOLD+0.093%; candidate drift+1.246%/+1.978%. Median complete-response latency−4.860%/−4.526%, TTFT−4.605%/−5.133%. Metal allocated bytes are identical in all eight measured responses:6,608,617,472 B. This is allocated Metal memory, not equal RSS. These are short screens, not sustained or long-context claims; do not add historical P13 gains.

[Protocol](performance/protocol.json), [summary](performance/summary.json), [campaign/process cleanup](performance/campaign.json), and the four report directories preserve each sample.

## Exact code and artifacts

Base P13 `c146e9a4a351fb159477c50fe83117447dcbed2b`; research parent `65b50fa5413747c709729707fde5cd033d8b022e`. Original personal checkout untouched. PRs122/123/124 were still open when this work began; the new PR is stacked on PR124.

- Reference CLI: `3dfea63ea3c2be58e3126f91686817d53d21fcf137c1eb62d8991e1d1647c4c4`.
- Measured L1 flag1 CLI: `cb41f7553fe757e281e3029b54607037c3b4f542c4315ec2b2b11360921f26c0`; libtest `db53ba919cfeadded43a777c09902b913d303521a318d487074f42351d6e8b85`.
- Final default CLI: `9e2c538c85851159bdb53a054e97485c6accfff8260ebc7f858aaa2fa5b9758c`; libtest `06478421f77b060d0dea3c1e561a7a600c2ff6d8a2dc3344757c92f68ec22e95`.

[Measured freeze](freeze.json) and [final freeze](freeze-default.json) identify 187 native source files. Their only native source difference is the Metal initialization policy in `native/src/gpu.rs`; shaders, numerical/ownership methods and tests are identical. Final default artifacts were physically requalified with all nine native optimization flags absent. Speed was measured on flag1 before this policy-only change; no second performance gain is claimed.

Rust/cargo1.98.1, Kani0.68.0/CBMC6.11.0, Python3.12.14, macOS27.2/build26B123, Xcode27.2/SDK27.2, deployment target26.2. Build environment: private `CARGO_TARGET_DIR`, `MACOSX_DEPLOYMENT_TARGET=26.2`, `MLXL3_MLX_ROOT=/Users/justin/Documents/mix-stq1_0/.venv/lib/python3.12/site-packages/mlx`. Each target was cloned from stopped P13 artifacts and package-cleaned before use. No shared writable target.

## Correction before speed, then final default qualification

[Initial two gates](quality/report.json) passed before timing. The untraced model compares four imposed tokens248044/1/42/7, complete finite248320-half logits and10240-float hidden, plus four84-state continuations:115,212,288 F32 state words and1,986,560 logits. Independent initial84-state archive and known initial logit hash remain required. Real counters are34triples+2P13doubles+14separate finalwrites per row; command/wait counts drop34. Tests retain an older prefix, reject an invalid deferred gate before commit, inject a failure after layer20, restore state/hidden/offset, recover exact continuation, and exercise top-level export/capture fallbacks. The real CLI two-token golden is `9f850af1cbba074b`, context4096, native backend/noMLX, 48×53slots.

[Final six gates](quality-default/report.json) pass with `MLXL3_NATIVE_CROSS_LAYER` and all eight previous native flags absent:

1. Actual default model4rows,34/2/14, complete states/logits/hidden, rollback/prefix/captures.
2. Existing F16/F32 HC/GDN component oracle, finite traces, foreign-owner fallback, uncommitted-state rejection and recovery.
3. Production CLI four goldens, unsupported-request rejection, cancel at context/delta, recovery and ledger.
4. Frozen independent bridge-budget fixture against the **new final CLI**, two goldens and memory-saver recovery.
5. AR-only production prefix: two goldens and one real cache hit.
6. Existing depth1 predictive/full-state/continuation fixture and drained readers.

All six child process groups are empty. The depth guard has a symbolic CPU check; the unchanged predictive fixture does not export a physical cross-layer counter. Real GPU Failed/OOM was not injected.

Preflight caught the reused budget plan pointing at the old P13 CLI. It was corrected **before execution**; [receipt](quality-default/preflight-correction.json). No stale-binary result was accepted. The first mutation-preparation command stopped before editing because an unscoped string also matched a proof cover; the corrected mutation removed the real production export guard, failed the exact one-test oracle, then restored and passed. All attempts remain in `opti.md`.

## CPU, bounded verification and CI

[Verification receipt](verification.json): fullCPU337pass/8ignored; DIRECT353/54; MLX389/272. Three strict Clippy variants, fmt, actual release CLI/libtest builds, normal reviewed inventory251 and manifest tests2 pass. Linux-only orchestration tests44 are skipped on this Mac. The ignored suites are not presented as executed GPU coverage; the six named physical gates above were selected explicitly.

Kani checks actual production helpers: eligibility has25obligations and6covered guard cases over symbolic bools and fullusize depth; strict qualified default/flag has240obligations and7covers over the existing UTF8≤4/unwind10 domain. Seven standard UTF8 guards are unreachable, no functional assertion is unreachable. No stubs or narrowed caller assumptions. This is bounded CPU verification, not a proof of Metal/OS/concurrency or complete generation. The real export-guard mutation is detected and the restored source passes.

Quality runner rejects21 malformed/missing/zero/boolean/wrong-count evidence variants; unchanged cleanup/strict-JSON helpers retain their separately archived39+15selfchecks. Ruff initially found import formatting and an unused import in private controllers; corrected versions have identical non-import AST and the21validator checks pass again without launching GPU. CrossHair was attempted but the module is absent, so Python source verification remains unverified. [Controllers and receipts](controllers/).

`.github/workflows/rust.yml` already runs PR/main CPU/direct/MLX and sharded Kani checks. The new real guard proof is required in the obligation manifest and partition regression. Exact-head hosted CI is pending publication; inherited parent Kani failures/timeouts are separately documented in PR122, not silently repaired or presented as green.

## Reproduction

```sh
cargo test --locked --no-default-features
cargo test --locked --features chat
cargo test --locked --features mlx,chat
cargo clippy --locked --no-default-features --all-targets -- -D warnings
cargo clippy --locked --features chat --all-targets -- -D warnings
cargo clippy --locked --features mlx,chat --all-targets -- -D warnings
cargo kani --lib --no-default-features --harness cross_layer_selection_requires_all_transaction_guards
cargo kani --lib --no-default-features --harness hc_gdn_flag_preserves_qualified_defaults_and_exact_literals
cargo build --locked --release --no-default-features --features direct-metal,chat --bin mlxl3-rs
python scripts/check_native_coverage.py
```

The physical [plan](quality-default/plan.json) and archived controller contain exact environment, commands, timeouts, artifacts and expected nonempty markers. Physical tests require the local checkpoint and an exclusive Apple GPU slot. The speed driver is unchanged SHA8560057f…; use its protocol before rerunning a measurement.

[Lossless log manifest](log-manifest.json) preserves raw and gzip hashes; uncompressed copies remain in private `build/issue125-l1/raw-logs`. No app replacement, merge, release or model-weight modification. L2/L3/L4/L5 are separate trials and are not validated by this report.
