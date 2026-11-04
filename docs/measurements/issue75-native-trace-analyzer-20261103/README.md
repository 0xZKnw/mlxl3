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
Rebased from the original main base onto `17c66327`; analyzer and test source are
unchanged by that rebase. Both workflow check sets and journal histories were
preserved. After the GPU queue, rerun focused checks for the removed no-op test
loop and the rebase; no engine build or inference is involved.

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

## 2026-11-04: existing named capture and exported encoder aggregation

The first named analysis is preserved in [named-summary.json](named-summary.json).
It joined every target GPU row, but recognized only `hc_write_float` and
`flash_logsumexp` as individual encoder labels. XML inspection found that the
exporter had concatenated logical encoder names using ` & ` within physical
encoder identities. This is observed export behavior, not evidence of an
individual kernel's duration.

The extension accepts a whole exact sequence of prefixed ASCII identifiers,
preserving order and repeated names. A generic or malformed constituent leaves
the whole sequence unknown. [The new summary](named-aggregate-summary.json)
reports 32 aggregate sequences separately from the two individual groups. All
6,955 target GPU rows still join explicit identities. The remaining 192 unknown
intervals have a 331,247 ns union. GPU totals, CPU encoding, joins and catalog /
timeline counts remain identical to the initial summary: raw duration
1,705,357,809 ns; per-encoder union sum 1,705,188,434 ns; global union
1,658,390,805 ns; observed active-event span 4,942,294,458 ns.

Largest groups in this partial capture:

| Exact group description | Exported encoder count | Whole-group interval union |
| --- | ---: | ---: |
| GDN sequence, 17 names including 3 QMV calls | 576 | 399.385080 ms |
| HC read sequence, 7 names | 1,535 | 205.210660 ms |
| MoE sequence, 91 names including 27 QMV calls | 150 | 128.249165 ms |
| MoE sequence, 101 names including 30 QMV calls | 124 | 113.730035 ms |
| MoE sequence, 81 names including 24 QMV calls | 142 | 107.679619 ms |
| Prepare / QMV NT2 / finish sequence | 15 | 106.271123 ms |

These are complete group costs. They are not divided among constituent kernels,
exclusive GPU percentages, full-generation totals or estimates of recoverable
time. The capture begins before the decode gate releases and covers only part
of the observed decode. The test build adds CPU bookkeeping. See the
[sanitized protocol and conditions](named-protocol.json), including frozen
binary identity and [the existing root generation/state parity gates](named-quality-summary.json). Raw trace,
TOC, XML and environments stay local; table hashes are in the summaries.

The extension has [49 focused and 304 relevant reporting tests](aggregate-validation.json)
passing, including 84 finite exhaustive ordered name sequences and malformed,
mixed-generic, delimiter, repetition, integration and CLI cases. The new
integration regression failed on the previous source; a disposable mutation
sorting and deduplicating the actual parser output is also
[detected](aggregate-mutation.json). Format, Ruff and compilation pass.

CrossHair confirms the actual parser's ordered output only for the 16
combinations of two names from `{a, Z, _, a1}`. Two soundness conditions for
arbitrary Unicode labels of at most 128 characters remain **not confirmed**
after 20 seconds each; this is inconclusive symbolic search, not a general
proof. An initial environment import failure and a direct private-helper run
without analyzed-condition messages are retained and do not count as success.
The existing PR/main CI runs focused tests and the bounded name-domain check.
No new inference or capture was run for this extension, and no engine source,
default, application or release was changed.
