import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "docs/measurements/flash-next-ssd-impl/opt-43-mtp-route-head/speed-heads/bench-supervisor.txt"
)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf"), True, 901, "600"])
def test_invalid_deadlines_are_rejected_before_waiting(seconds):
    monitor = runpy.run_path(str(SOURCE))["monitor"]
    with pytest.raises(AssertionError):
        monitor(None, {"timeout_seconds": seconds}, None)


@pytest.mark.parametrize("seconds", [None, 600, 900])
def test_default_and_explicit_deadlines_keep_short_child_waits(seconds):
    monitor = runpy.run_path(str(SOURCE))["monitor"]

    class Child:
        def wait(self, timeout):
            assert 0 < timeout <= 2
            return 0

    record = {} if seconds is None else {"timeout_seconds": seconds}
    assert monitor(Child(), record, None) == 0


def test_silent_real_child_times_out_and_is_reaped(monkeypatch):
    monitor = runpy.run_path(str(SOURCE))["monitor"]

    def output(command, **kwargs):
        return "Now drawing from 'AC Power'\n" if command[0] == "pmset" else "vm"

    monkeypatch.setattr(subprocess, "check_output", output)
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    child = subprocess.Popen(command)
    record = {
        "timeout_seconds": 0.02,
        "command": command,
        "before": {
            "vm_bytes": {"Swapins": 0, "Swapouts": 0},
            "pmset -g batt": "Now drawing from 'AC Power'\n",
        },
    }
    try:
        with pytest.raises(subprocess.TimeoutExpired) as error:
            monitor(child, record, lambda _: {"Swapins": 0, "Swapouts": 0})
        assert error.value.timeout == 0.02 and child.poll() is None
    finally:
        child.terminate()
        child.wait(timeout=2)
    assert child.poll() is not None
