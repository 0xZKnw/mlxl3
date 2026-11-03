"""Explicit read-swap amendment; keep GPU1's anchor, write budget and other guards."""

import copy
import json
import runpy
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
UPSTREAM = runpy.run_path(str(BASE / "gpu-runner.py"))
AMENDMENT = BASE / "diagnostic-swapin-amendment.json"
READ_LIMIT = 1 << 30


def validate(amendment, anchor_path, anchor_sha, previous_policy_sha):
    assert amendment["authorized_by_user"] == "Oui, lectures swap limitées à 1 Gio"
    assert amendment["experiment"] == "COLD-READY-DIAG-01"
    assert amendment["no_reanchor"] is True
    assert amendment["anchor_path"] == str(anchor_path)
    assert amendment["anchor_sha256"] == anchor_sha
    assert amendment["previous_policy_sha256"] == previous_policy_sha
    for key, expected in (
        ("maximum_new_swapout_bytes", 32 << 20),
        ("maximum_new_swapin_bytes", READ_LIMIT),
    ):
        assert type(amendment[key]) is int and amendment[key] == expected, key


def run(plan, anchor_path, identity_path):
    identity = json.loads(identity_path.read_text())
    root = UPSTREAM["ROOT"]
    frozen = {(root / name).resolve() for name in identity["sources"]}
    assert {AMENDMENT, Path(__file__).resolve()} <= frozen, "unfrozen amendment/runner"
    previous_loader = UPSTREAM["load_baseline"]
    sha = UPSTREAM["F02"]["file_sha"]

    def baseline(path):
        original = previous_loader(path)
        validate(
            json.loads(AMENDMENT.read_text()),
            path,
            sha(path),
            sha(UPSTREAM["POLICY_PATH"]),
        )
        return original

    for item in json.loads(plan.read_text()):
        assert type(item["maximum_new_swapin_bytes"]) is int
        assert item["maximum_new_swapin_bytes"] == READ_LIMIT
    core = UPSTREAM["F02"]
    core["run"].__globals__["load_baseline"] = baseline
    core["check_vm"].__globals__["MAXIMUM_NEW_SWAPIN_BYTES"] = READ_LIMIT
    UPSTREAM["run"](plan, anchor_path, identity_path)


def self_check():
    anchor = Path("/fixed/anchor.json")
    valid = {
        "authorized_by_user": "Oui, lectures swap limitées à 1 Gio",
        "experiment": "COLD-READY-DIAG-01",
        "no_reanchor": True,
        "anchor_path": str(anchor),
        "anchor_sha256": "a" * 64,
        "previous_policy_sha256": "b" * 64,
        "maximum_new_swapout_bytes": 32 << 20,
        "maximum_new_swapin_bytes": READ_LIMIT,
    }
    validate(valid, anchor, "a" * 64, "b" * 64)
    failures = 0
    for key in valid:
        for value in (None, False, "invalid"):
            bad = copy.deepcopy(valid)
            bad[key] = value
            try:
                validate(bad, anchor, "a" * 64, "b" * 64)
            except (AssertionError, KeyError):
                failures += 1
            else:
                raise AssertionError((key, value))
    core = UPSTREAM["F02"]
    guard = core["check_vm"]
    previous = guard.__globals__["MAXIMUM_NEW_SWAPIN_BYTES"]
    try:
        guard.__globals__["MAXIMUM_NEW_SWAPIN_BYTES"] = READ_LIMIT
        anchor_vm = {"Swapouts": 100, "Swapins": 200}
        guard({"Swapouts": 100 + (32 << 20), "Swapins": 200 + READ_LIMIT}, anchor_vm)
        for current in (
            {"Swapouts": 100 + (32 << 20) + 1, "Swapins": 200},
            {"Swapouts": 100, "Swapins": 201 + READ_LIMIT},
            {"Swapouts": 99, "Swapins": 200},
        ):
            try:
                guard(current, anchor_vm)
            except AssertionError:
                failures += 1
            else:
                raise AssertionError("budget or reset escaped")
    finally:
        guard.__globals__["MAXIMUM_NEW_SWAPIN_BYTES"] = previous
    return {"positive_cases": 2, "negative_cases": failures}


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        print(self_check())
    else:
        run(*(Path(value) for value in sys.argv[1:]))
