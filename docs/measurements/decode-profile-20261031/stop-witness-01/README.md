# STOP-WITNESS-01 — 31 October 2026

Two real, serial xctrace stop observations on a bounded Python CPU helper:

| SIGINT destination | Graceful exit | Exit | Saved trace / TOC export |
| --- | ---: | ---: | --- |
| Whole xctrace process group | 4.392 s | 0 | yes / 0 |
| xctrace PID alone | 3.949 s | 0 | yes / 0 |

The combined instruments, recording options, 2 s rolling window and 10 s time
limit were identical. The stop followed the startup log, a live service child,
and two seconds of collection. The graceful join bound was 20 s. The helper
was bounded to 120 s; no model or GPU computation was requested.

DTServiceHub had a separate process group in both cases: xctrace24091 / hub24102
and xctrace24315 / hub24326. Sending SIGINT to the xctrace group therefore did
not directly signal that service. The proposed shared-PGID mechanism is refuted
for this witness. These two observations do not explain the model's 60 s stop
failure, establish a causal difference between stop methods, or justify changing
the production controller.

The original P08 anchor remained unchanged: zero new Swapouts and Swapins,
AC power, quiet checks, admission10.5 GB and RSS ceiling10 GiB. All owned helper,
tracer and observed service PIDs disappeared; both direct children were joined.
The five archived source hashes still match. See [r2/result.json](r2/result.json),
[r2/inspection.json](r2/inspection.json), both xctrace logs and exported TOCs.

The first attempt is preserved separately in [result.json](result.json). Its
group trace was saved, but a terminal RSS0 sampling race triggered the existing
RSS guard, followed by a transient cleanup PermissionError/zombie. The original
PIDs were subsequently absent; [post-failure-cleanup-inspection.json](post-failure-cleanup-inspection.json)
and the recovery TOC preserve that check. No parent-only case ran in that first
attempt. The preregistered r2 observation accepts RSS0 only after a terminal
state or actual exit, while retaining the live RSS ceiling; `capture.py` was
not changed.

The witness scripts are archived operational evidence, not a general capture
runner. Formatting passed; r2 Ruff reported BLE001 for the cleanup catch that
records an error and continues to the other owned group. The executed source is
preserved byte for byte with that warning. No formal proof, model parity,
sampling attribution or performance claim follows from this witness.
