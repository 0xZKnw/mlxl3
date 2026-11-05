"""Malformed system samples stay explicit; observation cleanup is mandatory."""

import json
import runpy
import subprocess
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "docs/measurements/flash-next-ssd-impl/opt-49-variance-diagnostic/sample-system.txt"
)
API = runpy.run_path(str(SOURCE))


@pytest.mark.parametrize("bad", [None, "cpu", "disk", "gpu", "timeout"])
def test_system_sample_keeps_errors_instead_of_inventing_a_cause(bad):
    def read(command, **kwargs):
        assert kwargs == {"text": True, "timeout": 5}
        kind = {"ps": "cpu", "iostat": "disk", "ioreg": "gpu"}[command[0]]
        if bad == "timeout":
            raise subprocess.TimeoutExpired(command, 5)
        if bad == kind:
            return "invalid"
        return {
            "cpu": "12 18.5 1234 /Applications/Test.app/Test\n",
            "disk": "disk0\nKB/t xfrs MB\n12.5 100 240.75\n",
            "gpu": '"Device Utilization %"=63,"Renderer Utilization %"=43,"Tiler Utilization %"=25',
        }[kind]

    data = API["snapshot"](read)
    assert len(data["errors"]) == (3 if bad == "timeout" else int(bad is not None))
    assert (data["gpu"] is None) == (bad in ["gpu", "timeout"])
    if bad is None:
        assert data["cpu"] == [{"pid": 12, "cpu_percent": 18.5, "rss_kib": 1234, "name": "Test"}]
        assert data["disk"]["mb_total"] == 240.75
        assert data["gpu"]["Device Utilization %"] == 63


@pytest.mark.parametrize("failure", [None, "supervisor", "join"])
def test_observer_always_saves_status_and_joins(tmp_path, monkeypatch, failure):
    namespace = API["main"].__globals__
    events = []

    class Event:
        def wait(self, seconds):
            events.append(("wait", seconds))

        def set(self):
            events.append("stop")

    class Thread:
        def __init__(self, **kwargs):
            assert kwargs["daemon"]

        def start(self):
            events.append("start")

        def join(self, timeout):
            events.append(("join", timeout))

        def is_alive(self):
            return failure == "join"

    def supervised(plan):
        events.append("supervise")
        if failure == "supervisor":
            raise RuntimeError("interrupted")

    monkeypatch.setattr(namespace["threading"], "Event", Event)
    monkeypatch.setattr(namespace["threading"], "Thread", Thread)
    monkeypatch.setattr(namespace["runpy"], "run_path", lambda _: {"main": supervised})
    if failure:
        with pytest.raises((RuntimeError, AssertionError)):
            API["main"](tmp_path / "plan.json")
    else:
        API["main"](tmp_path / "plan.json")
    report = json.loads((tmp_path / "system-samples.json").read_text())
    assert (
        report["status"]
        == {
            None: "completed_observations_not_attribution",
            "supervisor": "failed_or_interrupted",
            "join": "failed_sampler_join",
        }[failure]
    )
    assert "stop" in events and ("join", 20) in events
    with pytest.raises(AssertionError, match="overwrite"):
        API["main"](tmp_path / "plan.json")
