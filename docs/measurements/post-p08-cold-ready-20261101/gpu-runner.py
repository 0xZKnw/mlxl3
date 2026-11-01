"""COLD-READY GPU1: reuse F02 supervision with one explicitly authorized fixed anchor."""

import json
import runpy
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
F02_PATH = ROOT / "docs/measurements/post-p08-dense-epilogue-20261031/bounded-gpu-runner.py"
F02 = runpy.run_path(str(F02_PATH))
ANCHOR_PATH = Path(__file__).with_name("anchor-gpu1.json").resolve()
POLICY_PATH = Path(__file__).with_name("policy-gpu1.json").resolve()
EXPERIMENT = "POSTP08-COLD-READY-01-GPU1"
AUTHORIZATION = "Oui, nouvelle campagne avec plafond de 32 Mio"
PREVIOUS_ANCHOR = {
    "path": str(F02["ORIGINAL_ANCHOR"].resolve()),
    "sha256": F02["ORIGINAL_ANCHOR_SHA256"],
}


def validate_documents(policy, anchor, anchor_path, anchor_sha, expected_anchor, previous_anchor):
    """Validate decoded documents and identities without I/O or an implicit baseline."""
    assert type(policy) is dict and type(anchor) is dict, "policy and anchor objects"
    assert type(anchor_path) is str and Path(anchor_path).is_absolute(), "absolute anchor path"
    assert anchor_path == expected_anchor, "fixed anchor path"
    assert policy["experiment"] == EXPERIMENT and anchor["experiment"] == EXPERIMENT, "experiment"
    assert policy["authorized_by_user"] == AUTHORIZATION, "explicit authorization"
    assert policy["anchor_path"] == expected_anchor, "policy anchor path"
    assert policy["anchor_sha256"] == anchor_sha, "changed anchor"
    assert type(anchor_sha) is str and len(anchor_sha) == 64, "anchor hash"
    assert all(character in "0123456789abcdef" for character in anchor_sha), "anchor hash"
    assert policy["previous_anchor"] == previous_anchor, "previous anchor provenance"
    assert policy["no_reanchor"] is True, "reanchoring forbidden"
    for key, expected in (
        ("maximum_new_swapout_bytes", 32 << 20),
        ("maximum_new_swapin_bytes", 256 << 20),
    ):
        assert type(policy[key]) is int and policy[key] == expected, key
    timestamp = anchor["timestamp"]
    assert type(timestamp) is str and timestamp, "anchor timestamp"
    assert datetime.fromisoformat(timestamp).utcoffset() is not None, "timestamp timezone"
    baseline = {key: anchor["vm_bytes"][key] for key in ("Swapouts", "Swapins")}
    assert all(type(value) is int and value >= 0 for value in baseline.values()), "VM baseline"
    return baseline


def load_baseline(anchor_path):
    return validate_documents(
        json.loads(POLICY_PATH.read_text()),
        json.loads(anchor_path.read_text()),
        str(anchor_path),
        F02["file_sha"](anchor_path),
        str(ANCHOR_PATH),
        PREVIOUS_ANCHOR,
    )


def validate_identity_sources(identity):
    sources = identity["sources"]
    assert type(sources) is dict and sources, "nonempty identity sources"
    covered = set()
    for name, digest in sources.items():
        assert type(name) is str and name, "identity source path"
        assert type(digest) is str and len(digest) == 64, "identity source hash"
        assert all(character in "0123456789abcdef" for character in digest), "identity source hash"
        covered.add((ROOT / name).resolve())
    assert {ANCHOR_PATH, POLICY_PATH} <= covered, "anchor and policy must be identity sources"


# Preserve the original run, guards, VM budgets and supervisor; replace only this loader.
F02["run"].__globals__["load_baseline"] = load_baseline


def run(plan, anchor_path, identity_path):
    validate_identity_sources(json.loads(identity_path.read_text()))
    F02["run"](plan, anchor_path, identity_path)


if __name__ == "__main__":
    run(*(Path(value) for value in sys.argv[1:]))
