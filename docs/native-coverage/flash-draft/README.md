# Native Flash speculative draft

This component runs the raw DFlash2 backbone without MLX. System Accelerate
performs CPU matrix multiplication with explicit BF16 rounding at the maintained
operator boundaries. The native Metal Flash target verifies proposals and commits
only the accepted prefix plus the target token. This is correctness work. No
throughput improvement is claimed.

The production JSONL bridge shares its existing generation, tokenization,
streaming, cancellation, sampling validation and mode events. Native DFlash uses
the same option parser as the reference implementation. Its CPU draft has an
owning KV cache, the existing ring/adaptive policies and transactional updates.
A failed draft append restores the target. Prefix identity prevents reuse after
a foreign session, reset or mismatched anchor.

## Evidence

[checks.json](checks.json) records the exact source, binary hashes and limits.
The numerical run uses the local Flash EXL3 checkpoint, the pinned raw DFlash2
checkpoint, an M5 with 24 GiB and a 0.6 GiB expert cache. Test commands load one
model at a time, with compilation and provers stopped during GPU qualification.

- The independent NumPy opt44 backbone fixture passes its historical 2% relative
  RMSE limit for hidden/layer outputs and 1% limit for all ten KV arrays. Results
  are numerical tolerance checks. They are not bitwise backbone parity.
- The separately captured MLX 0.32.2 RMS fixture checks 12,928 BF16-expanded words
  exactly at widths 128, 2560 and 10240. A temporary mutation removing the pre-gain
  BF16 rounding fails. The SGLang residual norm retains its distinct FP32 variance.
- The physical production session passes budgets 1, 2, 3 and 17, comparing emitted
  tokens, retained hidden bytes and all 248,320 continuation logits per case with
  ordinary native decoding. Acceptance and rejection are both exercised.
- The final real bridge passes AR/DFlash token hashes at budgets 2 and 17, six
  invalid requests, cancellation during context and delta, and both recovery
  generations. Loaded-library inspection contains no MLX library.
- The existing ten-step MTP head and MTP session physical regressions also pass.

Reproduce the backbone with `MLXL3_FLASH_DRAFT_ORACLE`,
`MLXL3_FLASH_DRAFT_KIND=dflash2` and `MLXL3_FLASH_DRAFT_MODEL` set to the independent
fixture and pinned raw draft, then select
`native_draft::native_backbone_matches_independent_numpy_layers_and_kv` with
`--exact --ignored --nocapture`. The session test is
`native_draft::speculative_session_preserves_ordinary_tokens_states_and_failure_recovery`.
Both require a direct build with `--features chat` on Apple Silicon.

The bridge command is
`python scripts/check-mtp-bridge.py ENGINE MODEL DRAFT --mode dflash --tokens 2,17 --context 256 --prompt-file PROMPT`.
The unchanged default `--mode mtp` keeps existing callers compatible.

## Verification limits

Kani checks the actual BLAS extent helper on two arbitrary 64-bit dimensions,
with no assumptions. All 19 obligations and two covers pass. The first encoding
timed out at 90s and remains in the evidence. The equivalent wide-product oracle
passes without reducing the input domain. This CPU model check covers neither
Accelerate arithmetic nor Metal or full generation. Full-library Kani reproduces
the existing macOS compiler ICE. CrossHair finds no checkable functions in the
process orchestration script.

The complete physical session uses mode 1 with default Q4/ring/adaptive settings.
Other modes and option combinations remain tracked for separate qualification.
The CPU draft expands its quantized coefficients into FP32 host storage. Its
memory use and speed are deferred optimization work. DSpark backbone support in
the fixture test does not activate a DSpark production session. PR #100 request
admission must also be integrated before enabling drafts under a bounded AR-only
request budget. No installed app was replaced and no release was published.
