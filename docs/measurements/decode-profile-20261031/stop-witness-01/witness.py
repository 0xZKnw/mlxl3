"""Preregistered CPU-only xctrace stop liveness witness; no model/perf inference."""

import json
import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
API = runpy.run_path(str(ROOT / "docs/measurements/decode-profile-20261031/capture.py"))
ANCHOR = ROOT / "docs/measurements/decode-profile-20261031/gpu-witness-01/anchor.json"
OPTIONS = ROOT / "docs/measurements/decode-profile-20261031/recording-options.json"
BASELINE = json.loads(ANCHOR.read_text())["vm_bytes"]
HELPER = """import time
end = time.monotonic() + 120
value = 1
while time.monotonic() < end:
    for index in range(1000):
        value = (value * 1664525 + index + 1013904223) & 0xffffffff
    time.sleep(0.01)
print(value, flush=True)
"""
RESULT = OUT / "result.json"
record = {
    "status": "incomplete",
    "experiment": "DECODE-PROFILE-08-STOP-WITNESS-01",
    "limits": "two CPU helper stop-liveness cases; no model/GPU throughput or sample attribution",
    "original_vm_baseline": BASELINE,
    "cases": [],
    "sources": {},
}


def save():
    RESULT.write_text(json.dumps(record, indent=2) + "\n")


def processes():
    raw = subprocess.check_output(["ps", "-axo", "pid,ppid,pgid,comm"], text=True, timeout=5)
    rows = []
    for line in raw.splitlines()[1:]:
        pid, parent, group, command = line.split(maxsplit=3)
        rows.append({"pid": int(pid), "ppid": int(parent), "pgid": int(group), "command": command})
    return rows


def owned_rows(rows, leaders):
    ids = set(leaders)
    changed = True
    while changed:
        before = len(ids)
        ids.update(row["pid"] for row in rows if row["ppid"] in ids)
        changed = len(ids) != before
    return [row for row in rows if row["pid"] in ids or row["pgid"] in leaders]


def checked(case, children, admission=False):
    API["guard"](
        case,
        BASELINE,
        10_500_000_000,
        [child.pid for child in children if child.poll() is None],
        admission,
    )


def cleanup(case, children):
    errors = []
    for child in reversed(children):
        try:
            API["STOP"](child)
            # A service child can remain in the owned group after its leader exits.
            residual = [row for row in processes() if row["pgid"] == child.pid]
            if residual:
                os.killpg(child.pid, signal.SIGTERM)
                end = time.monotonic() + 2
                while time.monotonic() < end and any(
                    row["pgid"] == child.pid for row in processes()
                ):
                    time.sleep(0.05)
                if any(row["pgid"] == child.pid for row in processes()):
                    os.killpg(child.pid, signal.SIGKILL)
            case.setdefault("cleanup", []).append(
                {
                    "pid": child.pid,
                    "reaped": child.poll() is not None,
                    "exit": child.returncode,
                    "residual_before": residual,
                }
            )
        except BaseException as error:  # noqa: BLE001 -- still clean the other owned group
            errors.append(repr(error))
    case["cleanup_errors"] = errors
    case["remaining_owned_rows"] = owned_rows(processes(), [child.pid for child in children])
    assert not errors and not case["remaining_owned_rows"], "owned process/group not cleaned"


def main():
    assert not RESULT.exists(), "refusing to overwrite witness"
    helper = OUT / "cpu-helper.py"
    helper.write_text(HELPER)
    for path in (
        Path(__file__),
        helper,
        OPTIONS,
        ANCHOR,
        ROOT / "docs/measurements/decode-profile-20261031/capture.py",
    ):
        record["sources"][str(path)] = API["sha"](path)
    xctrace = "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace"
    record["xctrace_version"] = subprocess.check_output([xctrace, "version"], text=True, timeout=5)
    record["uname"] = subprocess.check_output(["uname", "-a"], text=True, timeout=5)
    save()
    try:
        for mode in ("group", "parent"):
            case = {"mode": mode, "status": "incomplete", "join_limit_seconds": 20}
            record["cases"].append(case)
            children = []
            trace = OUT / f"{mode}.trace"
            case["trace"] = str(trace)
            checked(case, children, admission=True)
            with (
                (OUT / f"{mode}-helper.log").open("xb") as helper_log,
                (OUT / f"{mode}-xctrace.log").open("xb") as log,
            ):
                try:
                    target = subprocess.Popen(
                        [sys.executable, str(helper)],
                        stdout=helper_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    children.append(target)
                    command = API["tracer_command"]("combined", target.pid, trace, OPTIONS)
                    command[command.index("--window") + 1] = "2s"
                    command[command.index("--time-limit") + 1] = "10s"
                    case.update(
                        command=command,
                        target_command=[sys.executable, str(helper)],
                        target_pid=target.pid,
                        started_epoch=time.time(),
                    )
                    save()
                    tracer = subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "DEVELOPER_DIR": "/Applications/Xcode.app/Contents/Developer",
                        },
                    )
                    children.append(tracer)
                    case["tracer_pid"] = tracer.pid
                    ready = None
                    deadline = time.monotonic() + 20
                    while True:
                        assert tracer.poll() is None and target.poll() is None, (
                            "early helper/tracer exit"
                        )
                        rows = owned_rows(processes(), [target.pid, tracer.pid])
                        text = (OUT / f"{mode}-xctrace.log").read_text()
                        hub = [row for row in rows if "DTServiceHub" in row["command"]]
                        if "Ctrl-C to stop" in text and hub:
                            ready = ready or time.monotonic()
                            if time.monotonic() - ready >= 2:
                                break
                        assert time.monotonic() < deadline, "initialization timeout"
                        checked(case, children)
                        time.sleep(0.2)
                    case["before_stop_processes"] = rows
                    case["stop_epoch"] = time.time()
                    case["stop_signal"] = "SIGINT"
                    case["stop_destination"] = (
                        "process_group" if mode == "group" else "tracer_pid_only"
                    )
                    began = time.monotonic()
                    if mode == "group":
                        os.killpg(tracer.pid, signal.SIGINT)
                    else:
                        tracer.send_signal(signal.SIGINT)
                    save()
                    while tracer.poll() is None and time.monotonic() - began < 20:
                        checked(case, children)
                        time.sleep(0.1)
                    case["graceful_stop_seconds"] = time.monotonic() - began
                    case["graceful_completed_before_deadline"] = tracer.poll() is not None
                    case["exit_before_cleanup"] = tracer.poll()
                    case["after_stop_processes"] = owned_rows(processes(), [target.pid, tracer.pid])
                    case["status"] = (
                        "graceful_exit" if tracer.poll() is not None else "graceful_timeout"
                    )
                finally:
                    try:
                        cleanup(case, children)
                    finally:
                        save()
            checked(case, [])
            case["log"] = (OUT / f"{mode}-xctrace.log").read_text()
            case["output_saved_log"] = "Output file saved as:" in case["log"]
            if trace.is_dir():
                toc_command = [
                    xctrace,
                    "export",
                    "--input",
                    str(trace),
                    "--toc",
                    "--output",
                    str(OUT / f"{mode}-toc.xml"),
                ]
                case["toc_command"] = toc_command
                with (OUT / f"{mode}-toc.log").open("xb") as toc_log:
                    completed = subprocess.run(
                        toc_command,
                        stdout=toc_log,
                        stderr=subprocess.STDOUT,
                        timeout=15,
                        check=False,
                    )
                case["toc_exit"] = completed.returncode
            save()
            print(
                json.dumps(
                    {
                        key: case.get(key)
                        for key in (
                            "mode",
                            "status",
                            "graceful_stop_seconds",
                            "exit_before_cleanup",
                            "toc_exit",
                            "output_saved_log",
                        )
                    }
                ),
                flush=True,
            )
        record["status"] = "complete_stop_liveness_observations_not_model_validation"
    except BaseException as error:
        record.update(status="failed_or_interrupted", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
