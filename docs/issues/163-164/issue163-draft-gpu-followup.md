# P01 compact draft GPU projection follow-up

## Preregistered design (2026-11-09)

Baseline: PR163 `b28d33b`, compact CPU weights reconstruct an FP32 operand
before Accelerate SGEMM. No earlier physical result validates this follow-up.

Candidate: explicit experimental GPU option, disabled by default and requiring
compact storage. One synchronous Metal projection reads packed BF16 or affine
Q4 directly. Host activation, attention, normalization and transactional cache
publication remain unchanged. Each projection uploads compact bytes; this first
implementation does not claim device residency or remove synchronization.
FP32 affine decode uses separate multiply/add, and final output is rounded once
to BF16 ties-to-even. FP32 reduction order can differ from Accelerate, so neither
bitwise model parity nor speedup is assumed. Failure must return before any cache
publication; no fallback hides a failed GPU execution.

Planned checks: physical ignored fixture with independent scalar exact-integer
oracle, BF16 halfway rounding, non-square and multiple-row shapes, both formats,
malformed storage and nonfinite activation rejection; compare with production
CPU projection on exactly representable fixture. Rust formatting and Apple CI
compilation, CPU suite where applicable, Kani/source verifier availability.
No model, physical Metal execution or benchmark is available on this Linux
cloud checkout. No Mac access, default promotion, push or merge.

The required formal-proof skill and language reference were sought in the
available skill catalog and local skill installation but were not present.
This is a verification limitation, not a proof waiver or success.

## Implementation and verification

The production `Draft::linear` now routes compact projections through a private
Metal backend only with both `MLXL3_NATIVE_DRAFT_GPU=1` and
`MLXL3_NATIVE_DRAFT_COMPACT=1`. Other flag values fail closed. Small dense operands
and the default path retain Accelerate. `append` still stages all projected
updates before publishing cache planes. This is a GPU **projection prototype**,
not a complete GPU draft backbone: normalization, attention and activation
storage remain on CPU. It uploads weights and scans affine values each call,
synchronizes each projection, and uses a simple serial dot per output. These
costs can regress speed. No throughput benefit is claimed.

`MemoryPayload.projection_gpu_peak_bytes` reports the observed peak sum of the
four shared Metal buffers (compact weight, affine parameters/dummy, input,
output). This includes coexistence of host payloads and GPU copies but excludes
driver/compiler overhead; it is not RSS. No device weight cache is retained.
The existing CPU decode scratch field remains a conservative potential operand
size, not evidence it was allocated during a GPU projection.

Portable regression passed: one test checks valid non-square multirow projection
and six invalid layouts including empty, truncated, mismatched and oversized
dimensions. Kani 0.68.0/CBMC6.11.0 is available after restoring the tool PATH:
the actual production shape helper passed one harness, 20 obligations, zero
failures and two of two covers, verification 4.884323s. Scope: all u8 dimensions
and u16 lengths, no assumptions. This does not prove MSL, Metal execution,
floating-point parity, allocator behavior or model state transitions. Initial
`cargo` not found was a PATH issue, not missing Kani.

Commands use PATH=/home/agent/.cargo/bin and the shared cloud target directory:
`cargo test --lib --no-default-features native_draft_gpu::admission_tests`;
`cargo kani --lib --no-default-features --harness draft_gpu_projection_admission_matches_dimensions`.
Rustfmt and diff whitespace checks pass. A physical ignored test covers both
formats with an independent f64 exact-dyadic dot oracle, production CPU comparison,
two BF16 halfway cases and malformed/nonfinite inputs. It has not been run.
Apple compilation, physical kernel oracle, real bridge/logits/cache continuation
and full-model performance remain unverified. No benchmark or default promotion.

Verification-artifact follow-up preregistration: repeat the unchanged focused
test/proof solely to retain complete logs after the tool's first output was
truncated. This does not broaden the domain or add an independent validation.

Follow-up closed: the same one regression and 20-obligation/two-cover proof
pass again (proof 2.5277781s). Logs: `measurements/pr163-draft-gpu/admission.log`
and `kani.log`. Strict `cargo clippy --lib --tests --no-default-features
--features chat -- -D warnings` passes (`clippy.log`). The earlier no-chat
all-tests lint invocation failed on existing OpenAI/vision feature gating and a
concurrent SSD test's collapsible-if warning; the SSD owner corrected its warning.
No GPU test ran on Linux. `xcrun`/standalone MSL verifier unavailable; Metal source
and Apple API compilation await Apple CI and physical validation. Code local,
experimental option off by default; no performance-qualified result.

## Autorelease lifetime correction preregistration

Independent review identified missing local autorelease pools around Metal
creation/submission. Headless callers do not guarantee an outer pool, so command
and encoder references could retain each projection's buffers indefinitely.
Before publication, wrap backend creation and the full synchronous projection in
`objc::rc::autoreleasepool`, as existing GPU APIs do. Return only owned Device,
Queue and Pipeline handles from creation and an owned CPU Matrix from projection;
wait and validate command completion before draining the projection pool.
Repeat formatting, portable admission regression and strict chat lint. Physical
repeated-projection allocation/driver retirement remains unavailable locally.

Correction closed locally: both entrypoints now own local pools; no borrowed
Metal references leave either closure. The projection waits before copying the
owned CPU result and before draining its pool, including command failure paths.
Portable admission regression passes (one test), strict chat lint and rustfmt /
diff checks pass. Logs: `measurements/pr163-draft-gpu/admission-autorelease.log`
and `clippy-autorelease.log`. The pure shape helper is unchanged, so its existing
bounded proof remains applicable and was not rerun. Apple compilation and
physical memory-retirement testing remain pending.
