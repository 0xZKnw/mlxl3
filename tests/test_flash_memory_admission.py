"""Explicit load headroom blocks a campaign before loading, not after swapping."""

import json
import runpy
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))


def record(budget=4 << 30):
    return {
        "command": [sys.executable],
        "env": {},
        "load_headroom_bytes": budget,
        "minimum_live_free_bytes": 512 << 20,
        "maximum_compressor_bytes": 7_000_000_000,
    }


def gauges(free=4 << 30, compressor=7_000_000_000):
    return {"Pages free": free, "Pages occupied by compressor": compressor}


@pytest.mark.parametrize("budget", [1, 4 << 30, (1 << 64) - 1])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_load_headroom_uses_exact_free_boundary_without_counting_file_cache(budget, delta):
    r = record(budget)
    current = {**gauges(budget + delta), "File-backed pages": 1 << 64}
    saved = current.copy()
    if delta < 0:
        with pytest.raises(AssertionError, match="headroom"):
            API["memory_headroom"](r, current, launching=True)
    else:
        API["memory_headroom"](r, current, launching=True)
    assert r["memory_guard"]["status"] == ("failed" if delta < 0 else "passed")
    assert current == saved


@pytest.mark.parametrize("free", [(512 << 20) - 1, 512 << 20, (512 << 20) + 1])
@pytest.mark.parametrize("compressor", [6_999_999_999, 7_000_000_000, 7_000_000_001])
def test_live_boundaries_are_separate_from_initial_load_budget(free, compressor):
    r = record()
    passed = free >= 512 << 20 and compressor <= 7_000_000_000
    if passed:
        API["memory_headroom"](r, gauges(free, compressor))
    else:
        with pytest.raises(AssertionError):
            API["memory_headroom"](r, gauges(free, compressor))
    assert r["memory_guard"]["status"] == ("passed" if passed else "failed")


@pytest.mark.parametrize(
    "key", ["load_headroom_bytes", "minimum_live_free_bytes", "maximum_compressor_bytes"]
)
@pytest.mark.parametrize("value", [None, True, False, 0, -1, 1.5, "4GiB"])
def test_invalid_load_budgets_cannot_disable_the_model_guard(key, value):
    r = record()
    r[key] = value
    r["env"]["MLXL3_FLASH_MODEL"] = "unused-test-path"
    with pytest.raises(AssertionError, match="load_headroom_bytes"):
        API["memory_headroom"](r, gauges(), launching=True)
    assert r["memory_guard"]["status"] == "failed"


@pytest.mark.parametrize("key", ["Pages free", "Pages occupied by compressor"])
@pytest.mark.parametrize("value", [None, True, -1, 1.5, "missing"])
def test_incomplete_or_invalid_memory_gauges_fail_closed(key, value):
    current = gauges()
    if value == "missing":
        del current[key]
    else:
        current[key] = value
    r = record()
    with pytest.raises((AssertionError, KeyError)):
        API["memory_headroom"](r, current, launching=True)
    assert r["memory_guard"]["status"] == "failed"


@pytest.mark.parametrize(
    "key", ["load_headroom_bytes", "minimum_live_free_bytes", "maximum_compressor_bytes"]
)
@pytest.mark.parametrize("launching", [False, True])
def test_declared_null_policy_cannot_disable_guard_without_model_env(key, launching):
    r = {"command": [sys.executable], "env": {}, key: None}
    with pytest.raises(AssertionError, match="positive integer"):
        API["memory_headroom"](r, gauges(), launching=launching)
    assert r["memory_guard"] == {
        "status": "failed",
        "stage": "before_launch" if launching else "live",
    }


def test_unbudgeted_cpu_steps_and_preflight_sleeps_keep_their_existing_contract():
    for r in (
        {"command": [sys.executable]},
        {"command": ["/bin/sleep"], "env": {"MLXL3_FLASH_MODEL": "x"}},
    ):
        API["memory_headroom"](r, {})
        assert "memory_guard" not in r


@pytest.mark.parametrize(
    "fault",
    [
        "short",
        "missing_budget",
        "live_free",
        "live_compressor",
        "after_swap",
        "after_swapin",
        "none",
    ],
)
def test_real_sampler_admission_and_pressure_reap_owned_children(tmp_path, monkeypatch, fault):
    api = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))
    base = Path("docs/measurements/flash-next-ssd-impl")
    paths = {
        "sample": base / "opt-49-variance-diagnostic/sample-system.txt",
        "supervisor": base / "opt-45-audit-prerequisites/run-supervisor.txt",
        "monitor": base / "opt-43-mtp-route-head/speed-heads/bench-supervisor.txt",
        "parser": Path("benchmarks/benchmark_flash_memory.py"),
        "process": Path("scripts/native_json_process.py"),
        "bridge": Path("scripts/benchmark-flash-bridge.py"),
    }
    for path in paths.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    baseline = {"Swapins": 100, "Swapouts": 100}
    before = {**baseline, **gauges()}
    if fault == "short":
        before["Pages free"] -= 1
    live = before.copy()
    if fault == "live_free":
        live["Pages free"] = (512 << 20) - 1
    if fault == "live_compressor":
        live["Pages occupied by compressor"] += 1
    if fault == "after_swap":
        live["Swapouts"] += 1
    if fault == "after_swapin":
        live["Swapins"] += (256 << 20) + 1

    class FastEvent(threading.Event):
        def wait(self, timeout=None):
            return super().wait(min(timeout, 0.001) if timeout is not None else timeout)

    def vm_stat(command, **kwargs):
        assert command == ["vm_stat"] and kwargs == {"text": True, "timeout": 5}
        return "synthetic memory sample"

    original = runpy.run_path

    def load(path):
        namespace = original(path)
        selected = Path(path).resolve()
        if selected == (tmp_path / paths["sample"]).resolve():
            g = namespace["main"].__globals__
            g["snapshot"] = lambda: {"errors": [], "test_fixture": True}
            g["threading"] = SimpleNamespace(Event=FastEvent, Thread=threading.Thread)
        if selected == (tmp_path / paths["monitor"]).resolve():
            namespace["snapshot"] = lambda _: {
                "vm_bytes": before.copy(),
                "pmset -g batt": "Now drawing from 'AC Power'",
            }
            namespace["monitor"].__globals__["subprocess"] = SimpleNamespace(
                TimeoutExpired=subprocess.TimeoutExpired, check_output=vm_stat
            )
        if selected == (tmp_path / paths["parser"]).resolve():
            namespace["parse_vm"] = lambda _: live.copy()
        return namespace

    g = api["run_sampler"].__globals__
    monkeypatch.setitem(g, "ROOT", tmp_path)
    monkeypatch.setitem(g, "runpy", SimpleNamespace(run_path=load))
    monkeypatch.setitem(g, "subprocess", SimpleNamespace(check_output=vm_stat))
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "child-started"
    item = {
        **record(),
        "name": "admission",
        "command": [
            sys.executable,
            "-c",
            "import time; from pathlib import Path; Path('child-started').touch(); "
            + f"time.sleep({3 if fault.startswith('live') else 0})",
        ],
        "env": {"MLXL3_FLASH_MODEL": "path-never-loaded"},
        "expected_exit": 0,
        "timeout_seconds": 5,
    }
    if fault == "missing_budget":
        del item["load_headroom_bytes"]
    plan = tmp_path / "run.json"
    plan.write_text(json.dumps([item]))
    if fault == "none":
        api["run_sampler"](plan, baseline)
    else:
        with pytest.raises(AssertionError):
            api["run_sampler"](plan, baseline)
    result = json.loads(plan.with_name("run-result.json").read_text())
    r = result["commands"][0]
    launched = fault not in {"short", "missing_budget"}
    assert r["reaped"] and marker.exists() == launched and ("pid" in r) == launched
    assert r["before"]["vm_bytes"] == before and baseline == {"Swapins": 100, "Swapouts": 100}
    assert r["status"] == (
        "completed_expected_outcome" if fault == "none" else "failed_or_interrupted"
    )
    if fault.startswith("live"):
        assert r["memory_guard"]["status"] == "failed" and len(r["live_memory_samples"]) == 1
        sample = r["live_memory_samples"][0]
        assert type(sample["epoch"]) is float and sample["epoch"] > 0
        assert {k: v for k, v in sample.items() if k != "epoch"} == live
    if fault == "after_swap":
        assert r["maximum_live_swapout_bytes"] == 1 and "Swapouts changed" in r["error"]
    if fault == "after_swapin":
        assert r["maximum_live_swapin_bytes"] == (256 << 20) + 1 and "swapin limit" in r["error"]
    assert json.loads(plan.with_name("system-samples.json").read_text())["sampler_joined"]
    assert runpy.run_path is original


@pytest.mark.parametrize(
    "field", ["model", "load_headroom_bytes", "minimum_live_free_bytes", "maximum_compressor_bytes"]
)
def test_memory_admission_cannot_be_bypassed_by_omitting_the_vm_baseline(tmp_path, field):
    plan = tmp_path / "run.json"
    item = {"env": {"MLXL3_FLASH_MODEL": "x"}} if field == "model" else {field: 1}
    plan.write_text(json.dumps([item]))
    with pytest.raises(AssertionError, match="explicit cumulative VM baseline"):
        API["run_sampler"](plan)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


@pytest.mark.parametrize("start", [0, (1 << 64) - 1])
@pytest.mark.parametrize("limit", [0, 1, 256 << 20])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_cumulative_swapin_guard_preserves_original_limit_at_full_integer_boundaries(
    start, limit, offset
):
    baseline = {"Swapins": start, "Swapouts": 100}
    r = {"before": {"vm_bytes": baseline.copy()}}
    incoming = limit + offset
    current = {"Swapins": start + incoming, "Swapouts": 100}

    def monitor(child, record, parse):
        parse("sample")
        return 0

    wrapped = API["cumulative_monitor"](monitor, baseline, limit)
    if 0 <= incoming <= limit:
        assert wrapped(None, r, lambda _: current) == 0
    else:
        with pytest.raises(AssertionError):
            wrapped(None, r, lambda _: current)
    assert r["before"]["vm_bytes"] == baseline


@pytest.mark.parametrize("limit", [True, -1, 1.5, "256MiB"])
def test_cumulative_swapin_guard_rejects_invalid_limits(limit):
    with pytest.raises(AssertionError, match="invalid cumulative swapin limit"):
        API["cumulative_monitor"](lambda *_: 0, {"Swapins": 0, "Swapouts": 0}, limit)
