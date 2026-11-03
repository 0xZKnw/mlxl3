# GPU1 anchor adapter verification

1 November 2026. `gpu-runner.py` reuses the existing F02 `run`, resource guards,
supervisor, cumulative VM checks and failure cleanup through `runpy`. The only
replacement in that function's globals is `load_baseline`. The new wrapper
requires both `anchor-gpu1.json` and `policy-gpu1.json` in `identity.sources`
before delegating; F02 then verifies every source hash before calling the loader.

API: `run(plan: Path, anchor_path: Path, identity_path: Path)`. Command interface:

```sh
python docs/measurements/post-p08-cold-ready-20261101/gpu-runner.py PLAN ANCHOR IDENTITY
```

The fixed anchor and policy are in this same directory. The policy must contain
experiment `POSTP08-COLD-READY-01-GPU1`, the exact authorization
`Oui, nouvelle campagne avec plafond de 32 Mio`, absolute anchor path, its SHA-256,
32 MiB Swapout and 256 MiB Swapin ceilings, `no_reanchor: true`, and the unchanged
original F02 anchor path/hash as `previous_anchor`. The anchor must contain the
same experiment, a nonempty timestamp with timezone, and nonnegative integer
`vm_bytes.Swapouts` and `vm_bytes.Swapins`. Booleans and floats are not integers
for this contract. The adapter neither selects a baseline nor writes an anchor.

Commands used `/Users/justin/Documents/mix-stq1_0/.venv/bin/python` and its Ruff:

- `python -m pytest -q docs/measurements/post-p08-cold-ready-20261101/gpu_runner_selftest.py`:
  **45 passed**. Final log `gpu-runner-selftest-r2.log` (0.09 s). Disposable
  fixtures cover positive load/delegation, policy changes and missing fields,
  wrong authorization, invalid counters/timestamps, anchor transplantation and
  mutation, required identity entries, original-run hash-check ordering, and
  inherited budget boundaries. A disabled-authorization mutation is detected;
  the actual source is unchanged and the original check passes afterward.
- `python -m pytest -q docs/measurements/post-p08-dense-epilogue-20261031/bounded_guard_selftest.py`:
  **48 passed** (2.17 s), including actual CPU child success, timeout, overshoot,
  headroom refusal and process reaping. Log `gpu-runner-inherited-guard-selftest.log`.
- Ruff format/check and `py_compile` pass on both new files. An initial Ruff
  `S102` finding in the mutation test was fixed by loading a disposable source
  copy instead of using `exec`; the final 45-test run follows that correction.
- Workflow YAML parses and `git diff --check` passes. Targeted CI format, lint,
  compile and pytest lines were added next to the readiness checks. Remote CI
  has not run for these local changes.
- `python -m crosshair check docs/measurements/post-p08-cold-ready-20261101/gpu-runner.py:validate_documents
  --analysis_kind=asserts --per_condition_timeout=1 --per_path_timeout=0.1`:
  exit 1, **CrossHair module absent**, recorded in `gpu-runner-crosshair.log`.
  No symbolic verification or formal proof is claimed, and no installation or
  expensive prover was attempted.

No native compilation, GPU test, model load, anchor creation, or real campaign
launch was performed by this adapter task. The original F02 logs and policy
remain unchanged. These tests verify admission behavior, not GPU performance or
the correctness of model outputs. The authorized campaign still needs its fixed
anchor, policy and source identity created and checked by the parent runner.
