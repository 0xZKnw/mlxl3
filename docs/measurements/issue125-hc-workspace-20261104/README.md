# Issue 125 L5: per-HC workspace

Status: CPU checks and full suites pass; frozen artifacts, physical correctness and speed are pending. The strict `MLXL3_NATIVE_HC_WORKSPACE=0|1` setting is snapshotted once and defaults OFF; explicit1 is qualified only on exact Apple M5. This is an isolated prototype based on65b50fa5413747c709729707fde5cd033d8b022e; no PR, activation or app installation has occurred.

Each HC owns normalized10240, down320, active320 and up10240 temporaries. Its cache remains visible to the memory ledger even while leased. A lease retains the actual command buffer and permits reuse only after Completed. Dropping it in any other status, including NotEnqueued, invalidates the cached owners: an old caller can still commit its retained command safely, while subsequent calls allocate different buffers. Busy, dtype mismatch and trace use fresh buffers. Mixed and gate outputs remain independent and can be retained by callers.

The request plan explicitly reserves the future F32 cache:97HC instances ×84480B =8194560B. Idle cleanup clears mixer and both HCs of each layer after draining; snapshots and prefixes retain states and final outputs, never these temporaries. The L3 composition must request three buffers with no up owner and reserve4221440B when its F32 fusion is active. No shader or arithmetic changes occur in L5.

The existing seven-row post-P13 diagnostic associated388workspace requests/row with1.172ms/row of host allocation API calls. This is neither removable wall time nor a throughput estimate. The prototype also adds Rc/RefCell access, view clones and a command-buffer retain; its net cost is unmeasured.

## CPU evidence

`qa/focused-report.json` binds sources and lossless compressed logs. Direct/chat library check, strict direct all-targets Clippy, formatting, inventory252files, four deterministic policy tests and two Kani manifest tests pass. The first all-targets build found a test-only usize/u64 counter mismatch; its log is preserved and the signature was corrected.

Kani0.68.0 checks the actual state, capacity, parser and admission helpers:4harnesses,481checks,12covers. Domains: arbitrary key/completion booleans and u64 counts; parser valid UTF8 of at most four bytes, unwind10; state unwind6. No assume or stubs. Seven unreachable standard-library alignment/UTF8/format guards occur in the parser harness; functional assertions and all covers are reachable. This does not prove Metal retention, GPU synchronization, real lease Drop behavior, OS concurrency or full generation.

An exact copy of the real Key/State source prefix was tested in a disposable Rust standalone harness. Original passes; deleting the busy guard fails the independent second-lease assertion; restored original passes. This mutation checks the helper, not the Metal wrapper.

Final CPU source (including poisoned-cache and post-encode-error physical fixtures) passes all three strict Clippy modes and full suites: CPU340passed/8ignored, direct356/54, MLX compatibility392/272. Formatting, inventory252files, manifest2tests, Ruff and two actual portable CI policy functions pass. Physical tests are compiled but ignored; Linux-specific Kani process orchestration is not run locally. The admission Kani harness starts from actual native_ar(64,4096) and varies u64 count/up mode; the unchanged geometric-plan proof covers its earlier domain separately.

## Physical gates prepared, not run

After the root GPU grant, run the frozen direct libtest with `MLXL3_FLASH_MODEL` pointing at the qualified65c89531checkpoint, one test per process, bounded and exclusive:

```
NATIVE_LIBTEST gpu::hc::workspace_tests::physical_workspace_reuses_only_completed_private_temporaries --exact --ignored --nocapture --test-threads=1
MLXL3_NATIVE_HC_WORKSPACE=1 NATIVE_LIBTEST gpu::memory::tests::native_ledger_tracks_aliases_and_request_shrink_preserves_replay --exact --ignored --nocapture --test-threads=1
```

The first gate checks six varied F16/F32 reverse-strided cases against the unchanged OFF path, nonempty finite full mixed/gate words, actual7→3requests after warmup and selected-path counts. It retains old outputs, poisons the temporary buffers, rejects abandoned backing aliasing, exercises busy/dtype/trace fallback, drops encoded work before commit and later commits it after a fresh call, injects a post-acquisition precommit error, recovers, and clears/rewarms the cache. The existing ledger gate additionally checks complete84states, a retained snapshot, idle release, actual admission and request shrink. Full-model independent initial-state/logit archives and real CLI golden/cancel/prefix checks must run on the final composition before32-token timing. The L3 composition changes F32 warm requests6→3 and must adjust exact expectations without weakening path checks.

Use the common32-token MAIN/HOLD A/B/B/A screen only after correctness: same3GiB/48×53, four readers, dynamic0, depth0, prefixoff, T0/K1/penalty1, one warm plus one measured response per prompt/process. No build/prover/other inference may overlap. Preserve drift, outputs and hashes; do not add standalone gains.
