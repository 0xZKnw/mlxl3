# Cold GPU1 bridge inspection verification

1 November 2026. The new `benchmark-inspection.py` exposes
`preflight(entry, manifest_path=None, identity_path=None)` and `inspect(...)`.
Defaults are this directory's `manifest.json` and `identity.json`.

The COMPOSE preflight is reused in an isolated instance with A/B orders and both
GDN/F02 disabled. Cold readiness is A=0, B=1; bridge17 uses AB and pilot128 ABBA
for each of AR, MTP and DF. The existing exact command, model/draft paths,
three-GiB cache, four readers, quiet/headroom/time limits, binary hashes and
128-token independent golden checks remain in force. The three required binary
identities are `final_cli`, `python_runner` and `libtest`.

GPU1 policy validation delegates to the new fixed-anchor loader: manifest and
identity must agree with its exact anchor path/hash and 32-MiB limit. The anchor,
policy and reused/new controllers must be frozen sources. Required dependencies
include COMPOSE `benchmark-inspection.py`, F02 `benchmark-inspection.py` and
`bounded-gpu-runner.py`, and this directory's `gpu-runner.py` and inspector.

Each inspection loads a fresh F02 inspector instance and injects its own expected
environment plus the new preflight. The golden, mode, generated-count, memory,
VM-budget and supervisor cleanup checks in the original inspection body are
unchanged. The Cold flag is checked in the actual report as well as the plan;
GDN/F02 remain zero and prefill heads remains one. Failures enter the original
durable `try/finally`: a partially checked response cannot announce success or
`external_golden: true`. Existing result files cannot be overwritten. The
inherited journal label still says F02; the preregistered campaign and inputs are
checked as GPU1, and no historical result is reused as a GPU1 measurement.

Checks used the repository Python/Ruff environment:

- `python .../post-p08-cold-ready-20261101/inspection_selftest.py`: **18 positive
  cases and 21 expected rejections pass**, covering every arm/position/mode and
  wrong or missing Cold flags, GDN/F02 contamination, wrong order, unjoined
  readers/processes, golden divergence, incomplete rows/report, wrong mode or
  command, mismatched anchor/budget, missing frozen dependency or binary, swap
  overshoot and unauthorized policy with its source hash refreshed. Receipt:
  `inspection-selftest-r3.json`.
- A disposable source mutation disables both actual Cold-report checks. The
  known invalid report then escapes rejection, confirming that the regression
  oracle distinguishes the required behavior. The production source is
  unchanged and the disposable copy is removed.
- The inherited F02 `inspection_selftest.py` passes **17 positive and 72 expected
  rejection cases**. Receipt: `inspection-inherited-selftest.json`.
- Ruff format/check, `py_compile`, workflow YAML parse and targeted
  `git diff --check` pass. CI runs these checks and the new self-test next to the
  GPU1 runner tests; no remote CI run is claimed.

The first synthetic run failed because the disposable macOS `/var` fixture paths
were not canonicalized to `/private/var` before installing the fixed loader
paths. This was a fixture error; paths now use `resolve()`, matching the real
loader. The original traceback is retained in `inspection-selftest.stderr`;
the corrected r2/r3 receipts are separate. No historical report was edited.

CrossHair was already attempted for the GPU1 adapter and is absent; it was not
installed or relaunched. These deterministic CPU checks are not a formal proof
or numerical model parity. No GPU, native compilation, inference or real
campaign execution was performed by this inspector task. Synthetic journaling
is confined to a disposable working directory.
