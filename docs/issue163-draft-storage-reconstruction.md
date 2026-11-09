# Issue 162 P01/P02: compact draft storage and atomic ring commits

## Scope and integration

Reconstructed locally on 2026-11-09, against the current PR163 recovery checkout. No prior lost-checkout result is reused. No Mac, GPU, model benchmark, commit, push, or installed-app change was performed by this worker.

- **P01 implemented as an opt-in candidate**, `MLXL3_NATIVE_DRAFT_COMPACT=1`; absent/`0` preserves dense storage and any other value errors. Eligible FC, attention/MLP, convolution-projection and selector-projection matrices retain BF16 words, or packed affine-Q4 nibbles plus BF16 scale/bias per group of 64. Small elementwise weights, codebooks and Markov weights retain their existing dense representation. This is not a GPU draft port and not an assertion that every draft weight uses two/four bits.
- Each compact projection decodes into one temporary FP32 operand and calls the **unchanged** Accelerate SGEMM with the same complete matrix dimensions, transpose, accumulation path and final BF16 boundary. No tiled matmul or alternate accumulation order was substituted. Decoding adds CPU work; performance must be qualified before default promotion.
- The ledger counts actual retained payload capacities and reports `projection_decode_scratch_bytes` separately as the largest prospective transient FP32 operand. It is not current allocation, total peak memory or RSS. Explicit request-memory admission still refuses this draft mode under its existing policy; no budget bypass was introduced.
- **P02 ring append implemented:** stage all five layers' K/V producers first, validate every destination/update before the first write, then commit in place without fallible operations. Initial backing allocation is staged and published only after success. Existing non-ring behavior is unchanged. A normal ring append no longer clones ten 2048×1024 FP32 planes (80 MiB of payload); this is a source-level copy count, not measured latency or process-memory reduction.
- **P02 prefix capture implemented:** estimate all target/logits/draft/MTP/token/conversation payloads before entering an admission-controlled closure that clones the draft prefix. Recheck actual retained capacities afterward. Snapshot ownership remains independent, and checkpoint state excludes speculative proposals.

## Files and production boundaries

- `native/src/draft_storage.rs`: compact codec, actual checked multi-plane ring commit, portable tests and Kani harness.
- `native/src/native_bf16.rs`: immutable dense/compact weight adapter; unchanged Matrix arithmetic; Apple-only real SGEMM parity test.
- `native/src/native_draft.rs`: opt-in load, staged append, decoded operand adapter and retained/scratch/prefix memory accounting. This file also contains the separate worker's N01/N03/P05 changes.
- `native/src/memory_policy.rs`: generic `capture_if_fits` admission boundary used by the actual prompt-cache path.
- `native/src/native_prompt_cache.rs`: admission before expensive snapshot copies.
- `native/src/lib.rs`: portable codec/commit module registration.

## Verification completed on Linux

Environment: `/home/agent/.cargo/bin/cargo`, `CARGO_TARGET_DIR=/tmp/pr163-native-target`, `CARGO_PROFILE_DEV_DEBUG=0`, `CARGO_PROFILE_TEST_DEBUG=0`, `CARGO_INCREMENTAL=0`; no GPU/model loaded.

1. `cargo test --locked --no-default-features --features chat --lib draft_storage -- --nocapture`: **4 passed, 0 failed, 0 ignored** after fixture correction. Tests use production codec/commit functions:
   - Exhaustively decode all 65,280 finite BF16 bit patterns against direct IEEE high-word expansion, including signed zero. Assert retained capacity is two bytes/value and reject malformed/nonfinite inputs.
   - Independently reconstruct legacy affine values for constant, positive-only, negative-only, mixed, zero and deterministic random groups; compare every FP32 result bit. Assert packed payload is 36 bytes/group of 64 rather than 256 bytes; incomplete group rejects without mutation.
   - Compare ten actual mutable ring planes against independent unbounded histories through append sizes 1, 2046, 3, 2048, 2051, 17, 4097. Check wraparound and multi-window truncation and that every backing pointer stays unchanged.
   - Inject malformed final-plane shape, nonfinite last producer, malformed destination and overflowing start; all ten planes remain unchanged, then a valid retry succeeds.
2. `cargo test --locked --no-default-features --features chat --lib memory_policy::capture_tests -- --nocapture`: **1 passed, 0 failed, 0 ignored**. Below-budget and arithmetic-overflow rejection do not invoke the capture closure; exact boundary invokes it once; a capture error propagates.
3. `cargo clippy --locked --all-targets --no-default-features --features chat -- -D warnings`: **passed**.
4. `git diff --check`: **passed** at review time.

### Failures and unavailable checks retained

- Initial codec suite: 3 passed/1 failed because the test incorrectly classified BF16 word `0xff01` as nonfinite. It is a finite negative value; rejection fixture was corrected to NaN `0xff81`. The exhaustive finite-word decode itself had already passed. No production numerical change was made to satisfy this fixture.
- `cargo kani --lib --no-default-features --harness invalid_ring_producer_preserves_all_planes`: **not executed**, tool unavailable (`no such command: kani`). The harness calls actual `commit_ring`, symbolic u16 start and a malformed producer, checks nonmutation of all 2048 cells with unwind 2050 and covers wrap boundary. It does not prove allocator, producer math, SGEMM, GPU or concurrency behavior. Formal-proof skill was not present in the available catalog/path.
- First write attempt encountered the read-only filesystem and made no edits. Authorized escalated write subsequently succeeded.
- Linux Clippy/tests do **not** compile the macOS-only native draft/Accelerate/target wrapper. New `compact_weight_operands_and_sgemm_match_legacy_bf16_and_affine` test exercises actual dense/compact adapters plus Accelerate on Apple CI; it remains unrun locally. Existing `.github/workflows/rust.yml` runs full chat tests on Ubuntu and macOS-26, so portable tests and that Apple test are selected by existing CI. No hosted CI result is inferred from the workflow.
- Physical target logits/KV, pending-state recovery, prefix reuse and continuation, full-model memory residency/peak and end-to-end cycle latency remain **unverified without Mac/model**. Compact storage stays opt-in. No TPS or RAM-process gain is claimed.

Raw local verification logs: `/tmp/pr163-storage-logs/{storage-tests,capture-tests,clippy,kani}.log`. Parent performs final integrated suites, manifest/inventory reconciliation and exact-head CI separately.

## Preregistered mutation validation

`PR163-P01-P02-MUTATION-20261109`: changed only the production packed-Q4 low-nibble decode mask from 15 to 14. The independent dense-reference test **failed as expected**, exit 101, 0 passed/1 failed, at the bitwise operand comparison. Restored the exact pre-mutation source and reran all four storage/ring tests: **4 passed, 0 failed, 0 ignored**. Byte comparison confirms the file is identical to the preregistered source after restoration. This is fault-detection evidence, not a performance experiment.

Logs are preserved in `docs/measurements/issue163-draft-storage-20261109/`, with final component source SHA-256 identities. `native_draft.rs` remains a shared worker file; its final integrated identity and validation belong to the parent integration report. All jobs launched by this worker completed; no active benchmark or hardware task remains.
