# MLXL3 Engine 2.0.6-beta.1 — release validation

## Source and artifact

- Runtime/source commit: `b13eb5f3d6b1782a8f268c0832ad038b74ba1a22` (merged PR #169).
- Runtime information: version2.0.6, MLX backend, chat enabled, aarch64, commit `b13eb5f3d6b1`.
- Archive: `MLXL3-Engine-v2.0.6-beta.1-arm64.tar.gz`.
- Size: **67873120 bytes**.
- SHA-256: `c56ad2f3e0f6e5f1e87535618ed4a789f759a9471788bdbb5f73286eb214d0a2`.
- Release tag: `engine-v2.0.6-beta.1`; engine prerelease, `latest=false`.
- No Desktop DMG or installed application replacement.

## Changes actually integrated

Only the qualified prefill GEMM/reconstruction and grouped-QMM paths plus their FFN bundle prerequisite and checks were ported onto current main. Both default on for standalone LFM3 text8 bpw/MCG8/M5, with shape/dtype/batch guards and strict0 opt-outs. The older experimental worktree's RoPE, last-head-row, small-prefill, draft, asynchronous decode, sampler and KV-slab changes were not copied wholesale. No checkpoint weights or quantization were changed.

## Local checks

- Strict Clippy on all targets: CPU/chat and MLX/chat passed; formatting and reviewed318-file source inventory passed.
- CPU library:426 passed,10 ignored; additional targets37/17/37/1 passed.
- MLX library:453 passed,233 ignored. Ignored physical checks are not counted as executed.
- Python packaging/proof-manifest/inventory:70 passed,13 platform/backend skips,2 subtests passed.
- Kani0.68/CBMC6.11: three actual admission/default/geometry harnesses passed,31+170+241 obligations,2+3+4 covers;13 library obligations unreachable. UTF-8 override domain at most2 bytes, unwind6; grouped-layout unwind13. No arbitrary assumes added. This is bounded CPU model checking, not a Metal/MLX/pipeline proof.
- Physical M5: independent CPU reconstruction oracle, grouped QMM production/strided/fallback checks, FFN bias/logical-padding/unqualified fallback checks passed.
- Full model:35 runs over7 contexts and5 profiles (stock/P1/P2/combined/implicit default),9 captures of35 states and128000 logits/run; seven implicit-default runs. Max KL0.000121898704≤0.001 and max state relativeL2 0.005016183≤0.01. Prefill grouped-QMM component/qualified first-capture equality was checked bit for bit; full approximate model behavior is not claimed universally bitexact.

## Exact packaged runtime

Archive members, sizes, SHA-256, arm64 architecture, ad-hoc signatures and manifest version/tag passed. The extracted runtime loaded the extracted MLX library and reported the merged source identity. Source file hashes match the physically checked implementation.

The quality controller passed42 negative/positive process/PID/environment/response checks. It exercised the actual extracted `mlxl3` executable, comparing flags0 against implicit defaults without setting the new enable variables. Eight CLI forwards (5/128/256/499 tokens in both profiles) passed numerical checks, including the independent ExLlama499 reference. The packaged short34→64 and long1090→128 responses matched in full text/hash/count in both profiles.

**Interrupted coverage:** the OFF French request reached the2048-token cap rather than natural EOS. The default French request was interrupted when a separate packaged `mlxl3` inference appeared on the Mac. The controller stopped its own child and retained `failed_or_interrupted`, `parity:false`; it did not stop the external job. Therefore complete French/EOS parity and default-profile cancellation/recovery were **not fully verified on this final archive**. OFF-profile error/cancellation/recovery completed. No performance claim is derived from these quality timings.

An earlier library-suite controller falsely classified its own isolated child as external; that failed receipt is retained and the suite passed after permitting only the owned process group. Initial tool patch/import-wiring/self-check failures are preserved. The packaged-process guard includes both `mlxl3` and `mlxl3-hybrid`; historical snapshots inspected contained neither, but gaps between snapshots remain unknown.

## Performance and remaining limits

The earlier8–11% prefill signal belongs to a separate experimental source and warmed protocols. It is not a measured release-versus2.0.5 speedup. No new speed campaign was run for this release; decode and total-generation gains are not established. Complete historical French comparisons are separate from the interrupted final-archive check above. The new temporary reconstruction path may increase scratch memory. Broader languages, long-context limits, cold-JIT costs and complete GPU correctness are not proved by these fixtures.

Exact-merge hosted CI was inspected: Flash profile compatibility succeeded; the other workflows were still queued/running at publication preparation. Do not read local success as all hosted CI green. The user explicitly requested merging and publishing before waiting for the full remote pipeline.

Evidence: `docs/measurements/engine-206-20261111/`; package build/extraction receipt and quality-controller source are archived with the release preparation. Automatic Discord notification is verified separately after publication.
