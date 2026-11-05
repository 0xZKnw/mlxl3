# DEFAULT-AR10 controller source qualification

Controller and snapshot SHA-256:
`7405384a65476150e6d3163e3b22b33bb54368df57a60a5b737e86e227a8110a`.
The controller requires `--run-after-root-grant`; none of its production phases
was launched by this agent. It never writes opti.md. Parent owns the physical
campaign and the preregistration already written by the defaults source agent.

18/18 deterministic checks passed in `sourcecheck.json`, using the actual
controller functions imported without a grant. The fixture source is preserved
in `sourcecheck.py.txt`. Checks cover inherited flag removal, forced ON/OFF,
grant refusal, valid model evidence with truthful zero observed activity,
invalid counts/default marker/activity booleans, duplicate JSON, wrong idle/CLI
evidence, empty benchmark reports, and actual run_phase success/failure/timeout/
failed cleanup/spawn failure with fake Popen/wait/stop. No real process, model,
GPU, Cargo, build or prover ran in these checks. They are sampled adversarial
tests, not formal proof; cleanup OS behavior is reused from the existing helper
and is not independently re-proved by fake processes.

Commands run with the qualified Python 3.12 venv:

```text
PYTHONDONTWRITEBYTECODE=1 /Users/justin/Documents/mix-stq1_0/.venv/bin/python docs/measurements/ar10-defaults-20261105/sourcecheck.py.txt
/Users/justin/Documents/mix-stq1_0/.venv/bin/python -m ruff check --no-cache build/ar10-defaults/controller.py
/Users/justin/Documents/mix-stq1_0/.venv/bin/python -m ruff format --check build/ar10-defaults/controller.py
```

Initial source lint reported two BLE001 diagnostics at the interrupt capture
and cleanup-exception capture. Their intentional persistence/rethrow behavior
was documented with targeted noqa comments; final Ruff check and format check
pass. The first format pass reformatted the newly created controller before
the 18 checks and final source hash. No source changes followed the stable
signal to parent. The previously stated docs/ar10-defaults path abbreviation
was corrected to the canonical docs/measurements/ar10-defaults path before any
check or model run. No earlier physical validation or speed result is claimed.

Parent launch command (not executed by this agent), from the PR132 worktree:

```text
/Users/justin/Documents/mix-stq1_0/.venv/bin/python build/ar10-defaults/controller.py --run-after-root-grant
```

The five phases are default/no13 model, explicit13OFF idle, default/no13 CLI,
default/no13 MAIN32 and forced13ON MAIN32, using the same frozen CLI for both
speed rows. Both speed groups have one warmup and one measured run; text/hash
must equal the fixed prior MAIN golden. Two samples establish no reliable boost
or sustained throughput. Freeze binary/source checks precede any physical run.
