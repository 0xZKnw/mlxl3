# PR130 supervision cleanup regression

The [failed macOS job](https://github.com/0xZKnw/mlxl3/actions/runs/41997784225/job/127147306720) ran `drafts4-quality/guards-supervisor.txt`. Its line 554 masked the underlying failure with `KeyError: reaped`; the missing row/stdout context was not retained, so the exact CI cleanup exception cannot be recovered. The checked-out quality source matches merge `7de8ac53`.

The concrete source defect is reproducible: group signaling or waiting can raise before the sole `reaped` assignment, then the outer `finally` writes the incomplete row. Tests execute the real constructor and cleanup AST, with a real disposable child exiting between the poll and signal. Original sources produced 12 errors in the first four-test regression (raw `cleanup-before.log`), including ESRCH and missing cleanup reports.

All four draft supervision copies now start with `reaped: false`, tolerate a vanished group, wait at most five seconds after TERM and five after KILL, and persist a failed cleanup receipt in `finally`. Only an actually joined child (or no launched child) becomes true. Interrupted fixtures synchronize on a published PID rather than a fixed sleep; absent fields still fail validation and include context. No guard is replaced by `get(True)`. The workflow runs the focused tests on PRs and main.

Validation: `python -m pytest -q tests/test_flash_draft_supervisor.py` passes 4 tests/16 subtests, covering ESRCH, permission failure, no/already exited child and two bounded timeout waits. Ruff format/check passes for the test and all eight supervisor sources. `python scripts/check_native_coverage.py` passes 264 files; its six regression tests also pass.

The full `drafts4-quality/guards-supervisor.txt` entry ran under a 55-second parent deadline, passed in 24.46 seconds with 18 joined subprocess scenarios, 144 diagnostic cases, 12 event checks, 31 component cases, 17 head cases and 76 native-validator regressions. Its temporary historical outputs were backed up and restored; the new proof and stdout are retained beside this file, with exact source SHA-256 values. The other three copies exercise the same production cleanup through the focused tests; their full historical matrices were not rerun.

CrossHair was attempted and is absent (`No module named crosshair`); no formal proof is claimed. These are CPU supervision checks; no Metal, model, Cargo, performance, application or publication path ran.
