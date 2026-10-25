"""Reject incomplete or unstable component measurements before a full-model claim."""

import runpy
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "docs/measurements/flash-next-ssd-impl/opt-49-resident-gates/qualify-micro.txt"
)
QUALIFY = runpy.run_path(str(SOURCE))["qualify"]


def fixture():
    return {
        "status": "complete",
        "parity": "exact_route_ids",
        "layers": 48,
        "route_rows": 2,
        "route_count": 960,
        "resident_gate_bytes": 240 << 20,
        "no_expert_io": True,
        "warmup_sweeps": 4,
        "warmup_seconds": 2.1,
        "sweeps_per_sample": 32,
        "samples": [
            {"pair": p, "candidate": c, "seconds": 0.98 if c else 1.0}
            for p in range(5)
            for c in ([False, True] if p % 2 == 0 else [True, False])
        ],
    }


@pytest.mark.parametrize("case", ["eligible", "too_small", "slower", "drift"])
def test_component_thresholds_do_not_claim_tokens_per_second(case):
    data = fixture()
    for sample in data["samples"]:
        if sample["candidate"]:
            sample["seconds"] = {
                "eligible": 0.98,
                "too_small": 0.995,
                "slower": 1.01,
                "drift": 0.98,
            }[case]
    if case == "drift":
        data["samples"][0]["seconds"] = 1.1
    result = QUALIFY(data)
    assert (
        result["status"]
        == {
            "eligible": "eligible_component_only",
            "too_small": "rejected_component",
            "slower": "rejected_component",
            "drift": "inconclusive_drift",
        }[case]
    )
    assert result["full_model_eligible"] == (case == "eligible")
    assert result["throughput_gain_confirmed"] is False


@pytest.mark.parametrize(
    "case",
    [
        "incomplete",
        "parity",
        "memory",
        "io",
        "warmup",
        "warmup_nan",
        "missing",
        "duplicate",
        "reversed",
        "bool",
        "nan",
        "infinity",
        "zero",
        "empty",
    ],
)
def test_invalid_component_data_never_qualifies(case):
    data = fixture()
    if case == "incomplete":
        data["status"] = "incomplete"
    elif case == "parity":
        data["parity"] = "diverged"
    elif case == "memory":
        data["resident_gate_bytes"] += 1
    elif case == "io":
        data["no_expert_io"] = False
    elif case == "warmup":
        data["warmup_seconds"] = 1.99
    elif case == "warmup_nan":
        data["warmup_seconds"] = float("nan")
    elif case == "missing":
        data["samples"].pop()
    elif case == "duplicate":
        data["samples"][0] = data["samples"][1]
    elif case == "reversed":
        data["samples"].reverse()
    elif case == "empty":
        data["samples"].clear()
    else:
        data["samples"][0]["seconds"] = {
            "bool": True,
            "nan": float("nan"),
            "infinity": float("inf"),
            "zero": 0,
        }[case]
    with pytest.raises(AssertionError):
        QUALIFY(data)
