# PR163 cleanup permission race

Trial: PR163-CI-EPERM-20261109, preregistered by the integration owner in opti.md.
This is a correctness repair, not a performance experiment. Cloud Linux only;
no Mac, Metal, installed application, commit or publication changed here.

## Cause and contract

`run` launches a new-session direct child and calls `cleanup` after success or
failure. Cleanup signals discovered owned PIDs before signalling the direct
child's process group. Darwin can report EPERM for that second signal while
the process is exiting. Previously this exception escaped the finally block
before the bounded `child.wait`, producing `failed_cleanup` unnecessarily.
The deterministic regression reproduces this control-flow defect; Darwin
kernel timing itself is not reproduced by these Linux tests.

Permission failures from PID/group signals are now deferred, not treated as
proof of cleanup. Success still requires the direct child to be reaped and
a fresh process snapshot showing no owned group, exact target, remembered
descendant or unique registry command. Discovery failures remain fail-closed.
A surviving denied child or descendant still raises unconfirmed cleanup.
Both TERM and KILL attempts and their bounded waits remain in place. No
permissions, signal ownership boundaries or process-group reuse guards change.

## Checks

The baseline command selected the new group-permission regression and denied
descendant test: `python -m pytest -q -p no:cacheprovider
tests/test_native_ar_profile.py -k 'group_permission_error_still_reaps or
permission_error_does_not_hide'`. Result: **1 failed, 1 passed, 27 deselected**.
The failing stack ends at the old `cleanup` line 247, `os.killpg(child.pid, sig)`,
with `PermissionError: [Errno 1] exiting`. This baseline output is retained in
the task execution transcript, not a standalone raw log.

After the fix, the initial full controller suite passed **29 tests**. A fourth
regression then added the persistently denied direct child case, confirming
both bounded waits are attempted. Final commands, executed using
`/tmp/pr163-recovery-venv/bin/python`:

- `-m ruff check scripts/profile_native_ar.py tests/test_native_ar_profile.py`
- `-m ruff format --check scripts/profile_native_ar.py tests/test_native_ar_profile.py`
- `-m pytest -q -p no:cacheprovider tests/test_native_ar_profile.py tests/test_native_json_process.py`

Final result: **57 passed in 13.43s**, Ruff check and formatting passed.
Raw final output: [final.log](final.log). `git diff --check` passed.
Coverage includes actual disposable processes, silent timeout, detached
descendants, unrelated witness survival, reused PID/group guards and report
failure/parity states. New tests use production `cleanup` and `run` entrances.

Initial sandbox execution could not create temporary files; the reviewed
execution with temporary-file/process access succeeded. Ruff was initially
absent, then installed from PyPI at CI-pinned 0.16.5. Its first check reported
SIM117 in the new test; combining the contexts fixed it before the final run.

Formal-proof skill was absent from both skill catalogs and local skill search.
`python -m crosshair --help` failed: `No module named crosshair`. These tests
are not a deductive proof of OS signalling or concurrency. Hosted exact-head
CI and physical Darwin behavior remain unverified here. The existing fresh
cleanup budget/discovery guards are retained; this patch does not reclassify
any previous failed CI result as success or claim C01-C03 collectively closed.

Final source SHA-256:

- profile_native_ar.py: `62a3473848864419f986bb38d3f9f7731781548d736405a0779970d1a3385a7a`
- test_native_ar_profile.py: `52e1cf97f1913ca4f458dabd480746a3c70a8a79ce43e4d028a2a3409534e81d`
