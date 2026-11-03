# Native Metal trace analyzer — preregistration

2026-11-03. Read opti.md and the existing source/host profiling limits before
implementation. This is verification of exported trace data, not another engine
optimization or GPU campaign. The HC source/artifacts remain frozen elsewhere.

Use only ElementTree and the standard library. Validate each schema, row width,
nested table-local id/ref identity and actual process.pid, with positive target
coverage. Join explicit command-buffer/encoder IDs. Preserve unnamed/unmapped
work without guessing names from GPU event-label, frame-number or shader catalog.
`mlxl3.kernel/<function>` is the agreed diagnostic single-dispatch encoder label
contract; other labels remain unassigned. ShaderTimeline zero rows must not turn
into per-kernel times or a successful zero-work result.

Calculate unions per encoder and globally. Report raw sum, within-encoder
nested/duplicate overlap and cross-encoder overlap separately; these are encoder
GPU intervals, not exclusive kernel counters or a speed validation. CPU Encoding
durations stay separate. Reject malformed/missing/cyclic references, wrong
schemas and empty target coverage. Publish results atomically/exclusively, with
failed status on analysis error. No raw TOC, environment, other-process data or
private trace is committed; fixtures are synthetic and summaries target-only.

Verification plan: deterministic interval fuzz/exhaustive small cases against
independent integer point coverage; nested/duplicate/adjacent/negative endpoints;
synthetic full CLI journeys and malformed/missing refs/schema/PID cases. Attempt
CrossHair symbolic search on the actual pure helper with explicit scope/practical
timeouts; this is not a proof of XML parsing or the engine. Run format/Ruff and
focused/relevant full Python suites, then replay the old unnamed exported trace
without changing any inference or rerunning a capture.

Frozen unnamed baseline (PID filtered): 39890 total GPU rows, 8395 target;
7917/477/1 at depths 0/1/2; 8312 CPU Encoding rows; 8394 GPU rows join 8310 CPU encoders;
80 target shader metadata rows and 0 shader timeline rows. Raw interval sum
2144923578 ns, per-encoder/global union to be reproduced independently. Global
union 2059396071 ns. Conditions are not an exclusive speed experiment: root noted
a cleanup overlap and retained that limitation. Preserve this baseline unchanged.

Implementation/CPU validation: local, not yet published. No new GPU/engine work.

The CLI is `python scripts/analyze_native_metal_trace.py EXPORT_DIRECTORY --pid
TARGET_PID --output NEW_REPORT.json`. Export all four tables named above into
that directory. Existing reports are never replaced. A malformed/incomplete
export produces a failed report and exit 1, rather than successful attribution.

The old trace replay reproduces [the sanitized summary](old-unnamed-summary.json):
8395 target GPU rows, raw sum 2144923578 ns, per-encoder union sum 2144836661 ns,
global union 2059396071 ns, within-encoder overlap 86917 ns and cross-encoder
overlap 85440590 ns. Of 8395 GPU rows, 8394 join explicit encoder identities;
1 encoder is unmapped. Every row remains unnamed. The shader catalog's 80 rows and
timeline's 0 rows cannot establish per-kernel timing. These durations are encoder
activity within a partial trace, not a speed measurement; background cleanup
overlapped this capture, as recorded by the parent profiling report.

Executed checks:

- `python -m ruff format --check scripts/analyze_native_metal_trace.py tests/test_native_trace_analyzer.py`
  and `python -m ruff check` on the same files: pass. `python -m py_compile`:
  pass. Initial Ruff run found three missing explicit subprocess `check=False`
  arguments; corrected before tests.
- `python -m pytest -q tests/test_native_trace_analyzer.py`: 31 passed.
  This includes 784 exhaustive interval pairs on integer endpoints [-3,3], plus
  500 deterministic fuzz lists (seed 7503, up to 7 intervals, endpoints [-10,10]),
  against independent integer point coverage. Expected success, nested refs,
  PID filtering, malformed refs/schema/cells/width/IDs/durations, empty workload,
  missing input and production CLI/atomic exclusive report publication covered.
- Relevant complete reporting/profile suites: 286 passed in 3.93 s using
  `python -m pytest -q tests/test_native_trace_analyzer.py tests/test_flash_trace_report.py tests/test_flash_profile.py tests/test_decode_p09_attribution.py tests/test_flash_profile_export.py`.
- Disposable actual-source mutation replacing union accumulation with raw
  interval summation fails the independent regression: duplicate [-3,-2)
  yields 2 instead of 1. The first receipt-wrapper assertion expected a different
  pytest error prefix and failed after that expected test failure; the saved
  diagnostic was inspected and confirmed. Original source was never mutated.
- `crosshair check tests.test_native_trace_analyzer.crosshair_two_interval_union
  --report_all --per_condition_timeout=20`: one condition confirmed over all
  paths. This invokes the actual helper against independent two-interval
  inclusion/exclusion, for arbitrary integer endpoints with ordered intervals.
  First attempt failed with dynamic module registration `KeyError`; registration
  was fixed and this command rerun successfully.
- `crosshair check scripts.analyze_native_metal_trace.interval_union_ns
  --report_all --per_condition_timeout=20`:both arbitrary-list postconditions
  **not confirmed**, no counterexample reported. This is an inconclusive
  symbolic counterexample search, not proof of arbitrary-length union logic.

CrossHair was the existing project installation. No verifier dependency was
added. XML parsing, filesystems, real diagnostic label correctness, concurrency,
Metal execution and inference are not formally verified. The CI now runs the
focused tests and strict formatting/lint on PRs and main pushes; hosted results
will be inspected only after publication of an exact commit.
