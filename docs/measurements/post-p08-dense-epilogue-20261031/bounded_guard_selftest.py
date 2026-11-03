"""Disposable CPU-only checks; no private fixture, Metal, engine or model is used."""

import importlib.util
import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).with_name("bounded-gpu-runner.py")


def load(path=SCRIPT):
    spec = importlib.util.spec_from_file_location("bounded_guard_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sut():
    return load()


@pytest.fixture
def campaign(sut, tmp_path, monkeypatch):
    anchor = tmp_path / "anchor.json"
    anchor.write_text(
        json.dumps(
            {
                "campaign": "POSTP08-F02-01-w1",
                "timestamp": "2026-10-31T17:19:28.906524+00:00",
                "vm_bytes": {"Swapouts": 262144, "Swapins": 0},
            }
        )
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "original_anchor": str(anchor),
                "original_anchor_sha256": sut.file_sha(anchor),
                "baseline_swapout_bytes": 262144,
                "maximum_new_swapout_bytes": 33554432,
                "incident_already_counted_bytes": 524288,
                "no_reanchor": True,
            }
        )
    )
    monkeypatch.setattr(sut, "ORIGINAL_ANCHOR", anchor)
    monkeypatch.setattr(sut, "ORIGINAL_ANCHOR_SHA256", sut.file_sha(anchor))
    monkeypatch.setattr(sut, "PROTOCOL", protocol)
    return anchor, protocol


@pytest.mark.parametrize("counter", ["Swapouts", "Swapins"])
@pytest.mark.parametrize(
    "delta", [0, 1, 32 << 20, (32 << 20) + 1, 256 << 20, (256 << 20) + 1, -1, None, True, 1.5, "0"]
)
def test_exact_cumulative_bounds_and_invalid_counters(sut, counter, delta):
    baseline = {"Swapouts": 262144, "Swapins": 0}
    current = dict(baseline)
    current[counter] = baseline[counter] + delta if type(delta) is int else delta
    limit = 33554432 if counter == "Swapouts" else 268435456
    if type(delta) is int and 0 <= delta <= limit:
        sut.check_vm(current, baseline)
    else:
        with pytest.raises(AssertionError):
            sut.check_vm(current, baseline)


@pytest.mark.parametrize("bad", [-1, None, True, 0.0, "0"])
def test_invalid_baseline_is_not_accepted(sut, bad):
    with pytest.raises(AssertionError, match="invalid VM baseline"):
        sut.check_vm({"Swapouts": 262144, "Swapins": 0}, {"Swapouts": bad, "Swapins": 0})


@pytest.mark.parametrize("counter", ["Swapouts", "Swapins"])
def test_overshoot_is_saved_before_rejection(sut, counter):
    baseline = {"Swapouts": 262144, "Swapins": 0}
    current = dict(baseline)
    limit = 33554432 if counter == "Swapouts" else 268435456
    current[counter] += limit + 1
    record = {}
    with pytest.raises(AssertionError, match=counter):
        sut.observe_vm(record, current, baseline)
    assert record["trigger_vm_bytes"] == current
    assert record["maximum_live_" + counter.lower()[:-1] + "_bytes"] == limit + 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_new_swapout_bytes", 33554433),
        ("maximum_new_swapout_bytes", "33554432"),
        ("maximum_new_swapout_bytes", True),
        ("maximum_new_swapout_bytes", None),
        ("baseline_swapout_bytes", 0),
        ("incident_already_counted_bytes", 0),
        ("no_reanchor", False),
        ("original_anchor_sha256", "0" * 64),
        ("original_anchor", "/another/anchor.json"),
    ],
)
def test_protocol_changes_are_rejected(sut, campaign, field, value):
    anchor, protocol = campaign
    document = json.loads(protocol.read_text())
    document[field] = value
    protocol.write_text(json.dumps(document))
    with pytest.raises(AssertionError):
        sut.load_baseline(anchor)


def test_original_anchor_hash_path_and_missing_cap(sut, campaign, tmp_path):
    anchor, protocol = campaign
    assert sut.load_baseline(anchor) == {"Swapouts": 262144, "Swapins": 0}
    copy = tmp_path / "copy.json"
    copy.write_bytes(anchor.read_bytes())
    with pytest.raises(AssertionError, match="original anchor path"):
        sut.load_baseline(copy)
    anchor.write_text(anchor.read_text() + "\n")
    with pytest.raises(AssertionError, match="changed original anchor"):
        sut.load_baseline(anchor)
    anchor.write_bytes(copy.read_bytes())
    document = json.loads(protocol.read_text())
    del document["maximum_new_swapout_bytes"]
    protocol.write_text(json.dumps(document))
    with pytest.raises(KeyError):
        sut.load_baseline(anchor)


def vm(extra=0, incoming=0):
    return {
        "Swapouts": 262144 + extra,
        "Swapins": incoming,
        "Pages free": 2_000_000_000,
        "File-backed pages": 9_000_000_000,
    }


def test_two_commands_keep_the_original_baseline_and_incident(sut):
    values = iter([vm(524288), vm(33554432), vm(33554433)])
    hooks = sut.guarded_hooks(
        {"snapshot": lambda _: {"vm_bytes": next(values)}, "monitor": lambda *_: 0},
        {"Swapouts": 262144, "Swapins": 0},
    )
    records = []
    for index in range(2):
        record = {
            "status": "running",
            "minimum_available_bytes": 10_500_000_000,
            "before": hooks["snapshot"](None),
        }
        records.append(record)
        hooks["before_launch"](record)
        record["status"] = "completed_expected_outcome"
        assert record["maximum_live_swapout_bytes"] == (524288 if index == 0 else 33554432)
    third = {"status": "running", "minimum_available_bytes": 10_500_000_000}
    third["before"] = hooks["snapshot"](None)
    with pytest.raises(AssertionError, match="Swapouts"):
        hooks["before_launch"](third)
    assert third["maximum_live_swapout_bytes"] == 33554433


def test_final_snapshot_overshoot_is_saved_in_active_record(sut):
    values = iter([vm(), vm(33554433)])
    hooks = sut.guarded_hooks(
        {"snapshot": lambda _: {"vm_bytes": next(values)}, "monitor": lambda *_: 0},
        {"Swapouts": 262144, "Swapins": 0},
    )
    record = {
        "status": "running",
        "minimum_available_bytes": 2_000_000_000,
        "before": hooks["snapshot"](None),
    }
    hooks["before_launch"](record)
    with pytest.raises(AssertionError, match="Swapouts"):
        hooks["snapshot"](None)
    assert record["maximum_live_swapout_bytes"] == 33554433


@pytest.mark.parametrize("counter", ["Swapouts", "Swapins"])
def test_reset_above_anchor_is_rejected_between_observations(sut, counter):
    first, later = vm(), vm()
    first[counter] += 10
    later[counter] += 9
    values = iter([first, later])
    hooks = sut.guarded_hooks(
        {"snapshot": lambda _: {"vm_bytes": next(values)}, "monitor": lambda *_: 0},
        {"Swapouts": 262144, "Swapins": 0},
    )
    record = {
        "status": "running",
        "minimum_available_bytes": 2_000_000_000,
        "before": hooks["snapshot"](None),
    }
    hooks["before_launch"](record)
    with pytest.raises(AssertionError, match="reset VM counter"):
        hooks["snapshot"](None)
    assert record["trigger_vm_bytes"] == later


def require_overshoot_rejection(module):
    rejected = False
    try:
        module.check_vm(vm(33554433), {"Swapouts": 262144, "Swapins": 0})
    except AssertionError:
        rejected = True
    assert rejected, "guard accepted cap+1"


def test_actual_guard_mutation_is_detected(sut, tmp_path):
    require_overshoot_rejection(sut)
    original = SCRIPT.read_text()
    needle = 'assert deltas["Swapouts"] <= MAXIMUM_NEW_SWAPOUT_BYTES, "Swapouts"'
    assert original.count(needle) == 1
    mutant = tmp_path / "mutant.py"
    mutant.write_text(original.replace(needle, needle.replace("BYTES,", "BYTES + 1,")))
    with pytest.raises(AssertionError, match="guard accepted cap\\+1"):
        require_overshoot_rejection(load(mutant))
    assert SCRIPT.read_text() == original


@pytest.mark.parametrize("scenario", ["success", "overshoot", "timeout", "headroom"])
def test_real_supervisor_records_and_reaps_cpu_children(
    sut, campaign, tmp_path, monkeypatch, scenario
):
    anchor, _ = campaign
    monitor_relative = sut.MONITOR.relative_to(sut.ROOT)
    monitor = runpy.run_path(str(sut.MONITOR))
    supervisor = runpy.run_path(str(sut.SUPERVISOR))
    snapshots = 0

    def output(command, **_):
        nonlocal snapshots
        if command == ["vm_stat"]:
            snapshots += 1
            sample = vm(33554433 if scenario == "overshoot" and snapshots > 1 else 524288)
            if scenario == "headroom":
                sample["Pages free"] = sample["File-backed pages"] = 0
            return json.dumps(sample)
        if command == ["pmset", "-g", "batt"]:
            return "Now drawing from 'AC Power'\n"
        return "fixture\n"

    monitor["monitor"].__globals__["subprocess"] = SimpleNamespace(
        check_output=output, TimeoutExpired=subprocess.TimeoutExpired
    )

    def quiet(record, *_):
        record["quiet_checks"] = record.get("quiet_checks", 0) + 1

    supervisor["main"].__globals__["check_quiet"] = quiet
    real_run_path = runpy.run_path

    def loader(path):
        if Path(path).resolve() == sut.MONITOR:
            return monitor
        if Path(path).resolve() == sut.SUPERVISOR:
            return supervisor
        if str(path).endswith("benchmark_flash_memory.py"):
            return {"parse_vm": json.loads}
        return real_run_path(path)

    monkeypatch.setattr(sut, "runpy", SimpleNamespace(run_path=loader))
    monkeypatch.setattr(sut, "ROOT", tmp_path)
    monkeypatch.setattr(sut, "MONITOR", tmp_path / monitor_relative)
    monkeypatch.chdir(tmp_path)
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "sources": {},
                "binaries": {
                    "cpu": {"path": sys.executable, "sha256": sut.file_sha(Path(sys.executable))}
                },
            }
        )
    )
    command = [sys.executable, "-c", "print('CPU fixture')"]
    if scenario in ("overshoot", "timeout"):
        command[-1] = "import time; time.sleep(30)"
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            [
                {
                    "name": "cpu-fixture",
                    "command": command,
                    "env": {},
                    "require_quiet": True,
                    "minimum_available_bytes": 2_000_000_000,
                    "timeout_seconds": 0.05 if scenario == "timeout" else 3,
                    "expected_exit": 0,
                    "required_text": ["CPU fixture"],
                }
            ]
        )
    )
    if scenario == "success":
        sut.run(plan, anchor, identity)
    else:
        with pytest.raises((AssertionError, subprocess.TimeoutExpired)):
            sut.run(plan, anchor, identity)
    result = json.loads((tmp_path / "plan-result.json").read_text())
    record = result["commands"][0]
    assert record["reaped"] is True
    assert result["status"] == (
        "completed_expected_outcomes_not_speed"
        if scenario == "success"
        else "failed_or_interrupted"
    )
    assert record["status"] == (
        "completed_expected_outcome" if scenario == "success" else "failed_or_interrupted"
    )
    if scenario == "overshoot":
        assert record["maximum_live_swapout_bytes"] == 33554433
        assert record["trigger_vm_bytes"]["Swapouts"] == 262144 + 33554433
    if scenario == "headroom":
        assert "pid" not in record
    else:
        assert record["quiet_checks"] > 0 and record["exit_code"] is not None
    assert "parity" not in result and "parity" not in record
