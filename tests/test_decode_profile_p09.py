"""P09 reuses the guarded runner without weakening its fixed VM accounting."""

import json
import runpy
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "docs/measurements/decode-profile-p09-20261101/run.py"


@pytest.mark.parametrize("fault", [None, "campaign", "bool", "negative", "missing", "path"])
def test_anchor_identity_and_counter_domains(tmp_path, fault):
    api = runpy.run_path(str(PATH))
    anchor = tmp_path / "anchor.json"
    data = {"campaign": "DECODE-PROFILE-09", "vm_bytes": {"Swapouts": 10, "Swapins": 20}}
    if fault == "campaign":
        data["campaign"] = "old"
    elif fault == "bool":
        data["vm_bytes"]["Swapouts"] = True
    elif fault == "negative":
        data["vm_bytes"]["Swapins"] = -1
    elif fault == "missing":
        del data["vm_bytes"]["Swapins"]
    anchor.write_text(json.dumps(data))
    api["load_baseline"].__globals__["ANCHOR"] = anchor
    if fault:
        with pytest.raises((AssertionError, KeyError)):
            api["load_baseline"](tmp_path / "other.json" if fault == "path" else anchor)
    else:
        assert api["load_baseline"](anchor) == {"Swapouts": 10, "Swapins": 20}


@pytest.mark.parametrize("capture", [False, True])
def test_authorized_cumulative_limits_and_counter_reversal(monkeypatch, capture):
    api = runpy.run_path(str(PATH))
    baseline = {"Swapouts": 10, "Swapins": 20}
    if capture:
        guard = api["CAPTURE"]["guard"]
        ns = guard.__globals__
        monkeypatch.setattr(
            ns["subprocess"], "check_output", lambda *a, **kw: "Now drawing from 'AC Power'"
        )
        monkeypatch.setitem(ns, "QUIET", lambda *args: None)

        def check(counters):
            monkeypatch.setitem(ns, "PARSE_VM", lambda raw: counters)
            return guard({}, baseline, 10_500_000_000, [])
    else:
        guard = api["CORE"]["check_vm"]
        ns = guard.__globals__

        def check(counters):
            return guard(counters, baseline)

    ns["MAXIMUM_NEW_SWAPOUT_BYTES"] = api["MAXIMUM_NEW_SWAPOUT_BYTES"]
    check({"Swapouts": 10 + 196608, "Swapins": 20})
    check({"Swapouts": 10 + (32 << 20), "Swapins": 20 + (256 << 20)})
    for counters in (
        {"Swapouts": 11 + (32 << 20), "Swapins": 20},
        {"Swapouts": 9, "Swapins": 20},
        {"Swapouts": 10, "Swapins": 19},
        {"Swapouts": 10, "Swapins": 21 + (256 << 20)},
    ):
        with pytest.raises(AssertionError):
            check(counters)


@pytest.mark.parametrize("capture", [False, True])
def test_runner_wires_authorized_budget_and_unchanged_anchor(tmp_path, monkeypatch, capture):
    api = runpy.run_path(str(PATH))
    anchor = tmp_path / "anchor.json"
    anchor.write_text(
        json.dumps({"campaign": "DECODE-PROFILE-09", "vm_bytes": {"Swapouts": 10, "Swapins": 20}})
    )
    monkeypatch.setitem(api["run"].__globals__, "ANCHOR", anchor)
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps({"sources": {str(anchor): "frozen", str(PATH.resolve()): "frozen"}})
    )
    plan = tmp_path / "plan.json"
    plan.write_text("{}" if capture else "[]")
    controller = api["CAPTURE" if capture else "CORE"]
    guard = controller["guard" if capture else "check_vm"]
    guard.__globals__["MAXIMUM_NEW_SWAPOUT_BYTES"] = 0
    calls = []

    def execute(*args):
        calls.append(args)
        assert guard.__globals__["MAXIMUM_NEW_SWAPOUT_BYTES"] == 32 << 20
        return "dispatched"

    monkeypatch.setitem(controller, "run", execute)
    assert api["run"](plan, identity) == "dispatched"
    assert calls == [(plan, anchor, identity)]
    assert api["load_baseline"](anchor) == {"Swapouts": 10, "Swapins": 20}


@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("missing", [None, "anchor", "entry", "original"])
def test_explicit_mtp_restart_preserves_guards_and_requires_frozen_identity(
    tmp_path, monkeypatch, capture, missing
):
    entry = PATH.with_name("run-mtp-head.py")
    restart = runpy.run_path(str(entry))
    anchor = tmp_path / "anchor-mtp-head.json"
    anchor.write_text(
        json.dumps({"campaign": "DECODE-PROFILE-09", "vm_bytes": {"Swapouts": 99, "Swapins": 77}})
    )
    monkeypatch.setitem(restart["run"].__globals__, "ANCHOR", anchor)
    sources = {"anchor": anchor, "entry": entry.resolve(), "original": PATH.resolve()}
    if missing:
        del sources[missing]
    identity = tmp_path / "identity.json"
    identity.write_text(json.dumps({"sources": {str(p): "frozen" for p in sources.values()}}))
    plan = tmp_path / "plan.json"
    plan.write_text("{}" if capture else "[]")
    api = restart["API"]
    controller = api["CAPTURE" if capture else "CORE"]
    guard = controller["guard" if capture else "check_vm"]
    calls = []

    def execute(*args):
        calls.append(args)
        assert guard.__globals__["MAXIMUM_NEW_SWAPOUT_BYTES"] == 32 << 20
        assert api["load_baseline"](anchor) == {"Swapouts": 99, "Swapins": 77}
        return "dispatched"

    monkeypatch.setitem(controller, "run", execute)
    if missing:
        with pytest.raises(AssertionError):
            restart["run"](plan, identity)
        assert not calls
    else:
        assert restart["run"](plan, identity) == "dispatched"
        assert calls == [(plan, anchor, identity)]
