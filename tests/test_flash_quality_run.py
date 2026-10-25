"""Disposable process tests for audit45's bounded physical-test supervisor."""

import copy
import json
import os
import resource
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "docs/measurements/flash-next-ssd-impl/opt-45-audit-prerequisites/run-supervisor.txt"
)


@pytest.mark.parametrize(
    "case",
    [
        "ok",
        "expected_failure",
        "interleaved",
        "silent",
        "interrupt",
        "bad_exit",
        "zero_tests",
        "power",
        "swap",
        "backward",
        "memory",
        "missing_head",
        "failed_head",
        "empty_head",
        "valid_head",
    ],
)
def test_supervisor_checks_outcomes_and_reaps_children(tmp_path, monkeypatch, case):
    main = runpy.run_path(str(SOURCE))["main"]
    monkeypatch.chdir(tmp_path)
    from types import SimpleNamespace

    monkeypatch.setattr(
        resource,
        "getrusage",
        lambda _: SimpleNamespace(ru_maxrss=(11 << 30) if case == "memory" else 64 << 10),
    )
    before = {
        "pmset -g batt": "Now drawing from 'AC Power'\n",
        "vm_bytes": {"Swapins": 0, "Swapouts": 0},
    }
    snapshots = [copy.deepcopy(before), copy.deepcopy(before)]
    if case == "power":
        snapshots[1]["pmset -g batt"] = "Now drawing from 'Battery Power'\n"
    if case in ("swap", "backward"):
        snapshots[1]["vm_bytes"]["Swapouts"] = (256 << 20) + 1 if case == "swap" else -1

    def monitor(child, record, parser):
        if case == "interrupt":
            raise KeyboardInterrupt
        return child.wait(timeout=0.5)

    monkeypatch.setattr(
        runpy,
        "run_path",
        lambda path: (
            {"snapshot": lambda parser: snapshots.pop(0), "monitor": monitor}
            if "bench-supervisor" in str(path)
            else {"parse_vm": None}
        ),
    )
    negative = case == "expected_failure"
    text = (
        "running 1 test\ntest fake::case ... FAILED\ntest result: FAILED. 0 passed; 1 failed\nreason"
        if negative
        else "running 1 test\ntest fake::case ... ok\ntest result: ok. 1 passed; 0 failed"
    )
    if case == "zero_tests":
        text = "running 0 tests\ntest result: ok. 0 passed; 0 failed"
    if case == "interleaved":
        text = "running 1 test\ntest fake::case ... DIAGNOSTIC\nok\ntest result: ok. 1 passed; 0 failed"
    code = f"print({text!r}); raise SystemExit({101 if negative or case == 'bad_exit' else 0})"
    if case in ("silent", "interrupt"):
        code = "import time; time.sleep(60)"
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            [
                {
                    "name": "probe",
                    "command": [sys.executable, "-c", code],
                    "env": {},
                    "expected_exit": 101 if negative else 0,
                    "native_test": "fake::case",
                    "required_text": ["reason"] if negative else [],
                }
            ]
        )
    )
    if case in ("missing_head", "failed_head", "empty_head", "valid_head"):
        item = json.loads(matrix.read_text())
        reference = tmp_path / "head.json"
        item[0]["native_head_reference"] = str(reference)
        matrix.write_text(json.dumps(item))
        if case != "missing_head":
            count = 248320 if case == "valid_head" else 0
            reference.write_text(
                json.dumps(
                    {
                        "status": "failed" if case == "failed_head" else "complete",
                        "native_chain": {"status": "passed_native_chain", "logit_count": count},
                        "draft_head": {"logits": {"count": count}},
                    }
                )
            )
    if case in ("ok", "expected_failure", "interleaved", "valid_head"):
        main(matrix)
    else:
        with pytest.raises(
            (AssertionError, subprocess.TimeoutExpired, KeyboardInterrupt, FileNotFoundError)
        ):
            main(matrix)
    result = json.loads((tmp_path / "matrix-result.json").read_text())
    assert result["status"] == (
        "completed_expected_outcomes_not_speed"
        if case in ("ok", "expected_failure", "interleaved", "valid_head")
        else "failed_or_interrupted"
    )
    command = result["commands"][0]
    assert command["reaped"] is True
    with pytest.raises(ProcessLookupError):
        os.kill(command["pid"], 0)
    with pytest.raises(AssertionError, match="overwrite"):
        main(matrix)
