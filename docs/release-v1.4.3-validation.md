# MLXL3 1.4.3 — validation record

This record distinguishes source integration checks from packaged-artifact checks. The release body and asset manifests supply the final source SHA, artifact sizes and SHA-256; a local successful benchmark does not imply an installed or published update.

## PR32 review

Reviewed base `8c08c4c45c1f7795f476ca514e7c7d3b8cebce7b`, head `0f53d2cadc9f5a556340abdc4d86763400a3f4a4`, production callers, error paths, fixtures and workflows. No blocking defect found. Seven memory-policy CPU tests and 240 independent Swift context-advice cases passed. The four inspected head jobs passed, including **52/52 Kani harnesses**. [Native CI](https://github.com/0xZKnw/mlxl3/actions/runs/38063221792), [Desktop CI](https://github.com/0xZKnw/mlxl3/actions/runs/38063221788).

PR32 merged as `260cbf42cd4b5de0833ace8ee3f0f1ec7299a368`. This review did not rerun 27B inference. Its prior qualified and rejected experiments remain in [the memory report](adaptive-memory-qwen-20261009.md).

## Integrated source checks

The source integration includes PR31/32 and DFlash2, with six merge conflicts reviewed in full. The following local checks passed after the integration repairs:

| Check | Command / result |
| --- | --- |
| Rust format | `cargo fmt --all -- --check` |
| CPU strict lint | `cargo clippy --locked --all-targets --features chat -- -D warnings` |
| CPU tests | `cargo test --locked --features chat`: **74 passed, 3 ignored** |
| Shipped strict lint | `cargo clippy --locked --release --all-targets --features mlx,chat -- -D warnings` |
| Shipped build/tests | `cargo build --locked --release --features mlx,chat`; `cargo test --locked --release --features mlx,chat`: **94 passed, 77 ignored** |
| Python | `.venv/bin/python -m pytest -q`: **380 passed, 4 skipped** (three unavailable PonyEXL3 adapter cases, one absent local Ling model) |
| Desktop | `scripts/check-desktop.sh`: strict Swift checks and the complete fixture suite passed; context oracle **240 cases**, idle-memory oracle **114 cases**, native rendering and production StudioModel flows |
| C++ key cache | Independent five-field tuple ordering: **262144 comparisons**, 256 keys/128-entry eviction; 128 hits: **768 old allocations → 0 candidate allocations**. ASan/UBSan passed. These are correctness/allocation checks, not new timing results. |

Rust test logs precede a failed summary command (`rg` unavailable in the restricted PATH, final wrapper exit127); the successful test counts above come from the actual completed suites. Ignored tests are not claimed as executed.

The archived source/runtime identity for the eight physical tests is `3ba367d3db77655c27b8b3e818dff6bbe3fa8eaf`. Later CI-concurrency changes do not alter inference code. [Lossless integration logs, JSON, reproduction source and screenshots](measurements/release-1.4.3/integration-evidence.tar.gz), [per-file sizes and SHA-256](measurements/release-1.4.3/integration-evidence-index.json).

## Deliberate failure checks and fixes

- DFlash Tune lost the optional Memory saving flag at the Desktop/NDJSON boundary. A real StudioModel fixture failed before repair (exit133). The unchanged ON/OFF test passed after the shared bridge and native caller were connected. Draft state and conversation preservation were also checked.
- A signal-handler write could reenter the fixture's buffered stdout. A deterministic signal inside `raw.write` reproduced `RuntimeError` before repair; the handler now sets a flag and the request loop emits the acknowledgement. All **21 checker tests** passed, including silent/invalid/incomplete/nonfinite/divergent events and child cleanup.
- GCC rejected inlined replacement allocation operators in the C++ counter fixture. Preventing their inlining preserved the allocation boundary without suppressing strict diagnostics; Linux CI subsequently passed.
- The first parallel Kani invocation lacked its required `--output-format terse` and failed before verification. This setup failure remains archived. The corrected two-job run stopped on a hosted-runner shutdown signal; the log identifies no failed obligation or counterexample. A sequential rerun keeps every input domain, assertion and unwind bound. It remained in progress at publication; the user explicitly requested publication without waiting for CI.

## Physical Apple GPU checks

M5 Air, 24 GiB RAM, macOS27.2 beta, SDK26.5/deployment26.2, MLX0.32.2, Rust1.98.1, Python3.12.14. The physical-test report records **battery power, 69%**, not AC. Only Qwen3.6-35B-A3B EXL3 2.49bpw was loaded, with its Q4 MTP/DFlash heads. No local compiler/prover or other model ran concurrently. Each test ran alone with `--exact --ignored --nocapture`, a 150-second limit, a nonzero selected-test assertion and process cleanup.

All **8/8** tests passed:

1. Cache-view compaction preserves every bit and releases parent storage.
2. Packed embeddings preserve all F16 bit patterns and permutations.
3. Full Metal cache-key/eviction checks preserve outputs.
4. A compacted 256-token prefill checkpoint preserves 248320 finite logits and 80 state arrays, and obeys the complete budget.
5. Eight packed-embedding model positions preserve 248320 finite logits and 80 state arrays bit for bit.
6. MTP recursive sessions, including forced D0↔D1..3 transitions, match independent greedy/head-cache oracles.
7. DFlash captures, 24 verifications and 108 commits across prefills23/24/256 match 80 serial state arrays.
8. Nine adversarial context-copy blocks exercise accepted lengths1..3 and subsequent neural continuation with exact draft caches.

One checkpoint retained196509696→69799936 bytes for69636096 logical bytes. One exact embedding allocation measured1017118720→897417216 active bytes. These are scoped memory observations, not a global process-RAM or speed guarantee. Durations in correctness logs are **not throughput measurements**. The new campaign deliberately excludes 27B GPU inference; pinned geometry and earlier qualification support its availability.

## Formal scope and gaps

Kani0.68.0 checks real Rust CPU implementations. COPY transition harnesses cover initial lengths0..20, arbitrary u32 tokens, accepted prefixes0..7, arbitrary anchor/copied/enabled states and every committed cell, with unwind33 and preserved reachability checks. Disabled-copy width covers all usize values; the active lookup uses a bounded20-token independent forward oracle. No production stub or new input assumption was introduced.

The local LEN7 attempt exceeded its180-second limit and was terminated/reaped: **unverified**, not a counterexample. At publication, exact-source Linux/macOS jobs had passed; Desktop and Kani were still in progress. Publication proceeded on the user’s explicit instruction to skip waiting for CI; no fully green CI or completed COPY proof is claimed. Other package harnesses cover budgets, acceptance, shapes, indices, calibration and memory-policy arithmetic. Bounded CPU checks do not prove MLX, Metal, FFI, allocation, concurrency or universal generation behavior.

The real C++ source was attempted with `goto-cc`/CBMC; its installed frontend cannot parse the platform libc++ context. CrossHair is unavailable locally; no compatible Swift source verifier is installed. These remain **unverified**, with tests and diagnostics preserved. The inherited direct-API Q4 row-alignment fixture failure is recorded in [the DFlash report](dflash2-exact-optimisations-2026-10-08.md); normal production paths passing does not close it.

## Packaged-artifact validation and publication

The official `scripts/build-macos-dmg.sh` completed. The signed app and engine were built from clean tracked source **`ef8cc2b92e9be12dd41f97bf8cc250d0ec43f3f7`**. PR33 merged as **`657a337c363cbcdd97385285d3a4d963b3dd4d60`**; release tags identify the actual build source, an ancestor of that merge, rather than relabeling the binary as another commit.

Deep-strict signatures, arm64-only executables/libraries, plist1.4.3/build25, source manifests, math fonts, every runtime SHA/size, exact archive members and app/archive runtime identity passed. `hdiutil verify` and read-only mounting passed; **14 app files** matched the build bit for bit, the relocated runtime executed, and detach returned0. The full `scripts/check-desktop.sh` suite also passed with `MLXL3_TEST_ENGINE_ARCHIVE` pointing at the real1.4.3 archive: installation, relocation, execution, rejection/fallback and hostile-archive cases used disposable directories.

The signed35B bridge ran **47 DFlash default checks**, **43 optional context-copy checks**, **MTP with full Tune**, and **two conversational turns**, all successfully. Commands are retained in signed-bridge-suite.json. DFlash exercised budgets1/2/3/8/17/64, all policies, full Tune, cancellation, cache hits/misses, memory_saver, invalid requests and foreign27B draft-header rejection. The target remained35B throughout; no27B inference was launched. Each child was bounded and reaped, with no local compiler/prover or other inference concurrent. Default Qwen pipeline was ON; experimental options remained OFF except COPY in its explicit second pass.

All **33 complete responses** from the archived independent engine reference matched token hashes, committed history and nonzero token counts. The mapping shifts request IDs by2 after40 to account for new memory-policy cases. An initial incorrect cardinality guard failed before comparing outputs; that failed report is retained. SystemPython3.9 similarly failed the first metadata helper because file_digest was unavailable; rerunning it with configuredPython3.12 passed. Neither initial failure was mislabeled as a completed validation. Durations from these checks are not throughput claims.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| MLXL3-Desktop-v1.4.3-b25-Apple-Silicon.dmg | 73356632 | `85e5b067c3640a4fe8fb4660e09577181a9f21dcc38d6ada49096b8a669549bb` |
| MLXL3-Engine-v1.4.3-arm64.tar.gz | 66910324 | `22e6728247a212dadb1ed3d5a9f335ee2bf858455271d08b72e92db9738f2d91` |
| MLXL3-Engine-v1.4.3-arm64.json | 13587 | `6f0ee44c82bd925f69603d3ef9a453126210dd962fedead0865d2773d0f73593` |
| MLXL3-Release-v1.4.3-validation.json | 16537 | `45ce4db05089216d0fb0e995511cd55d6d4e7af5f3a1752e2fa3442f9c296273` |

The machine-readable validation record is attached to both releases. [Engine1.4.3](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.3) uses latest=false; [Desktop1.4.3/build25](https://github.com/0xZKnw/mlxl3/releases/tag/v1.4.3) is the latest Desktop update. Installed copies and user histories were not replaced. Both releases explicitly disclose CI pending at publication, verifier limits, unavailable tests and the inherited Q4 fixture.

## Performance interpretation

No new speed campaign was run. The retained key-view microbenchmark's approximately25% CPU lookup reduction is not a tokens/s claim. The [isolated 35B campaign](dflash2-key-view-tps-isolated-2026-10-10.md) failed its drift criterion despite exact outputs. It establishes neither a new global gain nor DFlash2 superiority over MTP. Rejected GPU/Q4/KV8 prototypes remain rejected.

Final evidence: [lossless publication archive](measurements/release-1.4.3/publication-evidence.tar.gz), [SHA/size index](measurements/release-1.4.3/publication-evidence-index.json). Unchanged entries remain in the earlier integration archive. Remote API asset sizes/digests and both tag targets were checked, and `/releases/latest` selected Desktop1.4.3. CI snapshots record status at publication/closure; no wait for later jobs.
