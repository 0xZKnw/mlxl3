import hashlib
import json
import os
import resource
import runpy
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "docs/measurements/slotstream-20261026/supervisor.py"


@pytest.mark.parametrize("size", [0, 1, (256 << 10) - 1, (256 << 10) + 1, (1 << 20) + 17])
def test_campaign_source_hash_streams_all_bytes_without_read_bytes(tmp_path, monkeypatch, size):
    path = SCRIPT.parents[1] / "post-sushi-20261028/gpu-runner.py"
    digest = runpy.run_path(str(path))["file_sha"]
    payload = (bytes(range(256)) * ((size + 255) // 256))[:size]
    file = tmp_path / "source.bin"
    file.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    def forbidden(*_):
        raise AssertionError("campaign hashing must not allocate the whole file")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert digest(file) == expected


def test_campaign_source_hash_rejects_missing_and_non_file_inputs(tmp_path):
    path = SCRIPT.parents[1] / "post-sushi-20261028/gpu-runner.py"
    digest = runpy.run_path(str(path))["file_sha"]
    with pytest.raises(FileNotFoundError):
        digest(tmp_path / "missing")
    with pytest.raises(IsADirectoryError):
        digest(tmp_path)


def test_campaign_source_validation_calls_streaming_hash_before_launch(tmp_path, monkeypatch):
    path = SCRIPT.parents[1] / "post-sushi-20261028/gpu-runner.py"
    context = runpy.run_path(str(path))
    identity = tmp_path / "identity.json"
    anchor = tmp_path / "anchor.json"
    identity.write_text(json.dumps({"sources": {"sentinel": "unchanged"}}))
    anchor.write_text(json.dumps({"vm_bytes": {"Swapouts": 0, "Swapins": 0}}))
    called = []

    def streamed(file):
        called.append(file)
        raise OSError("test read failure before any process")

    monkeypatch.setitem(context["run"].__globals__, "file_sha", streamed)
    with pytest.raises(OSError, match="test read failure"):
        context["run"](tmp_path / "plan-not-read.json", anchor, identity)
    assert called == [context["ROOT"] / "sentinel"]


@pytest.mark.parametrize("counter", ["Swapouts", "Swapins"])
@pytest.mark.parametrize("delta", [0, 1, 256 << 20, (256 << 20) + 1, -1, None, True])
def test_sushi_campaign_guard_has_zero_swapout_budget(counter, delta):
    path = SCRIPT.parents[1] / "post-sushi-20261028/gpu-runner.py"
    guard = runpy.run_path(str(path))["check_vm"]
    baseline = {"Swapouts": 100, "Swapins": 100}
    current = baseline.copy()
    current[counter] = 100 + delta if type(delta) is int else delta
    allowed = type(delta) is int and 0 <= delta <= (0 if counter == "Swapouts" else 256 << 20)
    if allowed:
        guard(current, baseline)
    else:
        with pytest.raises(AssertionError):
            guard(current, baseline)


@pytest.mark.parametrize(
    "fault", [None, "wrong_test", "zero_tests", "ignored", "truncated", "missing_marker", "exit"]
)
def test_native_quality_outcome_requires_the_selected_nonempty_test(tmp_path, monkeypatch, fault):
    module = runpy.run_path(str(SCRIPT))
    monkeypatch.chdir(tmp_path)
    name = "linear::banked_scales::tests::fixture"
    text = f"running 1 test\ntest {name} ... ok\ntest result: ok. 1 passed; 0 failed\nEXACT_FINITE"
    if fault == "wrong_test":
        text = text.replace(name, "unrelated::case")
    elif fault == "zero_tests":
        text = "running 0 tests\ntest result: ok. 0 passed; 0 failed\nEXACT_FINITE"
    elif fault == "ignored":
        text = f"running 1 test\ntest {name} ... ignored\ntest result: ok. 0 passed; 0 failed; 1 ignored\nEXACT_FINITE"
    elif fault == "truncated":
        text = f"running 1 test\ntest {name} ... EXACT_FINITE"
    elif fault == "missing_marker":
        text = text.replace("EXACT_FINITE", "")
    snapshot = {
        "pmset -g batt": "Now drawing from 'AC Power'\n",
        "vm_bytes": {"Swapins": 0, "Swapouts": 0},
    }
    previous = {
        "snapshot": lambda _: snapshot,
        "monitor": lambda process, record, parser: process.wait(timeout=5),
    }
    module["main"].__globals__["runpy"] = SimpleNamespace(
        run_path=lambda path: (
            previous if path.endswith("bench-supervisor.txt") else {"parse_vm": None}
        )
    )
    monkeypatch.setattr(resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=65536))
    plan = tmp_path / "quality.json"
    plan.write_text(
        json.dumps(
            [
                {
                    "name": "quality",
                    "command": [
                        sys.executable,
                        "-c",
                        f"print({text!r}); raise SystemExit({1 if fault == 'exit' else 0})",
                    ],
                    "env": {},
                    "expected_exit": 0,
                    "native_test": name,
                    "required_text": ["EXACT_FINITE"],
                }
            ]
        )
    )
    if fault is None:
        module["main"](plan)
    else:
        with pytest.raises(AssertionError):
            module["main"](plan)
    result = json.loads((tmp_path / "quality-result.json").read_text())
    assert result["status"] == (
        "completed_expected_outcomes_not_speed" if fault is None else "failed_or_interrupted"
    )
    record = result["commands"][0]
    assert record["reaped"] and record["exit_code"] == (1 if fault == "exit" else 0)
    with pytest.raises(ProcessLookupError):
        os.kill(record["pid"], 0)


def child():
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError])
def test_cleanup_accepts_exit_between_poll_and_signal(monkeypatch, error):
    module = runpy.run_path(str(SCRIPT))
    process = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    process.wait(timeout=5)

    class StalePoll:
        pid = process.pid
        calls = 0
        joined = False

        def poll(self):
            self.calls += 1
            return None if self.calls == 1 else process.poll()

        def wait(self, timeout):
            self.joined = True
            return process.wait(timeout=timeout)

    def exited(*_):
        raise error("process exited concurrently")

    monkeypatch.setattr(os, "killpg", exited)
    wrapped = StalePoll()
    module["stop"](wrapped)
    assert wrapped.joined and process.returncode == 0


def test_cleanup_terminates_and_joins_real_process():
    module = runpy.run_path(str(SCRIPT))
    process = child()
    try:
        module["stop"](process)
        assert process.returncode == -signal.SIGTERM
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def test_live_permission_failure_is_not_hidden(monkeypatch):
    module = runpy.run_path(str(SCRIPT))
    process = child()
    real_killpg = os.killpg

    def denied(*_):
        raise PermissionError("live process denied")

    try:
        monkeypatch.setattr(os, "killpg", denied)
        with pytest.raises(PermissionError, match="live process denied"):
            module["stop"](process)
        assert process.poll() is None
    finally:
        real_killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def test_monitor_failure_is_durable_and_child_is_joined(tmp_path, monkeypatch):
    module = runpy.run_path(str(SCRIPT))
    monkeypatch.chdir(tmp_path)
    plan = tmp_path / "matrix.json"
    plan.write_text(
        json.dumps(
            [
                {
                    "name": "fixture",
                    "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                    "env": {},
                    "expected_exit": 0,
                }
            ]
        )
    )

    def monitor(*_):
        raise RuntimeError("fixture memory guard")

    snapshot = {
        "pmset -g batt": "Now drawing from 'AC Power'\n",
        "vm_bytes": {"Swapins": 0, "Swapouts": 0},
    }
    previous = {"snapshot": lambda _: snapshot, "monitor": monitor}
    module["main"].__globals__["runpy"] = SimpleNamespace(
        run_path=lambda path: (
            previous if path.endswith("bench-supervisor.txt") else {"parse_vm": None}
        )
    )
    with pytest.raises(RuntimeError, match="fixture memory guard"):
        module["main"](plan)
    report = json.loads((tmp_path / "matrix-result.json").read_text())
    assert report["status"] == "failed_or_interrupted"
    record = report["commands"][0]
    assert record["status"] == "failed_or_interrupted"
    assert record["reaped"] and record["exit_code"] == -signal.SIGTERM
    assert "fixture memory guard" in record["error"]
    assert "failed_or_interrupted" in (tmp_path / "opti.md").read_text()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "PID PPID COMM\n",
        "bad header\n",
        "PID PPID COMM\n1 0\n",
        "PID PPID COMM\n0 0 x\n",
        "PID PPID COMM\n1 -1 x\n",
        "PID PPID COMM\nx 0 y\n",
        "PID PPID COMM\n1 0 x\n1 0 y\n",
    ],
)
def test_process_snapshot_rejects_incomplete_or_ambiguous_rows(text):
    module = runpy.run_path(str(SCRIPT))
    with pytest.raises(ValueError):
        module["parse_processes"](text)


def test_process_guard_distinguishes_our_descendants_and_foreign_work():
    module = runpy.run_path(str(SCRIPT))
    rows = module["parse_processes"](""" PID PPID COMM
1 0 /sbin/launchd
10 1 /usr/bin/python
11 10 /bin/zsh
12 11 /tmp/s4e-engine
20 1 /tmp/s4e-engine
30 10 /usr/bin/swift-frontend
40 1 /tmp/mlxl3-check.A/studio-check
50 51 /tmp/mlxl3-native
51 50 /tmp/python
""")
    blocked = module["competing_processes"](rows, 10)
    assert [pid for pid, _, _ in blocked] == [20, 30, 40, 50]
    assert [pid for pid, _, _ in module["competing_processes"](rows)] == [12, 20, 30, 40, 50]


@pytest.mark.parametrize("phase", ["before", "during"])
def test_quiet_guard_refuses_compiler_and_saves_failure(tmp_path, monkeypatch, phase):
    module = runpy.run_path(str(SCRIPT))
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "launched"
    plan = tmp_path / "quiet.json"
    plan.write_text(
        json.dumps(
            [
                {
                    "name": "quiet",
                    "command": [
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(marker)!r}).touch()",
                    ],
                    "env": {},
                    "expected_exit": 0,
                    "require_quiet": True,
                }
            ]
        )
    )
    calls = 0

    def process_snapshot(command, **kwargs):
        nonlocal calls
        assert command == ["ps", "-axo", "pid,ppid,comm"]
        calls += 1
        return "PID PPID COMM\n1 0 /sbin/launchd\n" + (
            "42 1 /usr/bin/swift-frontend\n" if phase == "before" or calls > 1 else ""
        )

    def monitor(process, record, parse_vm):
        parse_vm("")
        return process.wait(timeout=5)

    snapshot = {
        "pmset -g batt": "Now drawing from 'AC Power'\n",
        "vm_bytes": {"Swapins": 0, "Swapouts": 0},
    }
    previous = {"snapshot": lambda _: snapshot, "monitor": monitor}
    module["main"].__globals__["runpy"] = SimpleNamespace(
        run_path=lambda path: (
            previous
            if path.endswith("bench-supervisor.txt")
            else {"parse_vm": lambda _: snapshot["vm_bytes"]}
        )
    )
    monkeypatch.setattr(subprocess, "check_output", process_snapshot)
    with pytest.raises(RuntimeError, match="concurrent build"):
        module["main"](plan)
    report = json.loads((tmp_path / "quiet-result.json").read_text())
    record = report["commands"][0]
    assert report["status"] == record["status"] == "failed_or_interrupted"
    assert record["reaped"] and record["competing_processes"] == [
        [42, 1, "/usr/bin/swift-frontend"]
    ]
    if phase == "before":
        assert not marker.exists() and "pid" not in record
    else:
        assert record["quiet_checks"] == 2 and record["pid"] > 0
