"""Calibration uses the existing production bridge benchmark, with fixed options."""

import runpy
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / (
    "docs/measurements/flash-next-ssd-impl/opt-47-layer-cache/calibration-bridge.txt"
)
LOAD = runpy.run_path
SPEED = LOAD(str(SOURCE.parent / "runtime/speed-supervisor.txt"))


def test_calibration_bridge_preserves_fixed_prompt_budget_and_defaults(monkeypatch):
    calls = []
    api = {"PROMPTS": {}, "run": lambda *a, **k: calls.append((a, k))}
    monkeypatch.setattr(runpy, "run_path", lambda path: api)
    monkeypatch.setattr(sys, "argv", [str(SOURCE), "/engine", "/model", "/output"])
    LOAD(str(SOURCE), run_name="__main__")
    assert api["PROMPTS"] == {
        "CAL": "Explain the difference between CPU caches and virtual memory. Give a numerical example."
    }
    assert calls == [
        (
            (Path("/engine"), Path("/model"), Path("/output")),
            {"tokens": 128, "repeats": 1, "timeout": 120, "cache_gib": 3, "prompt": "CAL"},
        )
    ]


@pytest.mark.parametrize("args", [["/engine", "/model"], ["/e", "/m", "/o", "extra"]])
def test_calibration_bridge_rejects_wrong_arity_before_loading_backend(monkeypatch, args):
    def forbidden(path):
        pytest.fail("loaded bridge despite invalid arguments")

    monkeypatch.setattr(runpy, "run_path", forbidden)
    monkeypatch.setattr(sys, "argv", [str(SOURCE), *args])
    with pytest.raises(ValueError):
        LOAD(str(SOURCE), run_name="__main__")


@pytest.mark.parametrize(
    "mutation", [None, "empty", "partial", "duplicate", "capacity", "expected", "extra"]
)
def test_bank_guard_requires_every_layer_and_exact_capacity(mutation):
    rows = [f"Flash Next: loading layer {i}/48, 53 exact expert slots" for i in range(1, 49)]
    expected = [53] * 48
    if mutation == "empty":
        rows.clear()
    elif mutation == "partial":
        rows.pop()
    elif mutation == "duplicate":
        rows[0] = rows[1]
    elif mutation == "capacity":
        rows[0] = rows[0].replace("53 exact", "54 exact")
    elif mutation == "expected":
        expected[0] = True
    elif mutation == "extra":
        rows.append(rows[0])
    if mutation is None:
        SPEED["validate_banks"]("\n".join(rows), expected)
    else:
        with pytest.raises(AssertionError):
            SPEED["validate_banks"]("\n".join(rows), expected)


def test_speed_wrapper_keeps_verified_entry_and_installs_only_directory_and_guard(monkeypatch):
    class Entry:
        def __init__(self):
            self.__globals__ = {}
            self.calls = []

        def __call__(self, stage):
            self.calls.append(stage)

    entry = Entry()
    monkeypatch.setitem(SPEED["main"].__globals__["API"], "main", entry)
    SPEED["main"](0)
    assert entry.calls == [0]
    assert entry.__globals__ == {
        "__file__": str(SOURCE.parent / "runtime/speed-supervisor.txt"),
        "check_report": SPEED["check_report"],
    }


@pytest.mark.parametrize("corrupt", [False, True])
def test_report_wrapper_requires_allocation_after_original_report_check(
    tmp_path, monkeypatch, corrupt
):
    directory = tmp_path / "bridge"
    directory.mkdir()
    lines = [f"Flash Next: loading layer {i}/48, 53 exact expert slots" for i in range(1, 49)]
    (directory / "stderr.log").write_text("\n".join(lines[:-1] if corrupt else lines))
    data, oracle = {"report": "original"}, {"oracle": "independent"}
    item = {"report": "bridge/report.json", "layer_slots": [53] * 48}
    calls = []

    def original(*args):
        calls.append(args)
        return {"status": "validated"}

    namespace = SPEED["check_report"].__globals__
    monkeypatch.setitem(namespace, "OLD_REPORT", original)
    monkeypatch.setitem(namespace, "__file__", str(tmp_path / "speed-supervisor.txt"))
    if corrupt:
        with pytest.raises(AssertionError, match="incomplete bank"):
            SPEED["check_report"](data, item, oracle)
    else:
        assert SPEED["check_report"](data, item, oracle) == {"status": "validated"}
    assert calls == [(data, item, oracle)]
