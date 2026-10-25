import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "smallm_profile", Path(__file__).resolve().parents[1] / "benchmarks/analyze_smallm_profile.py"
)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def fixture():
    samples = []
    orders = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]
    for phase, count in [("warmup", 2), ("measure", 8)]:
        for round_ in range(count):
            for rows in range(1, 5):
                for config in orders[round_ % 4] if phase == "measure" else range(4):
                    samples.append(
                        {
                            "rows": rows,
                            "config": config,
                            "round": round_,
                            "phase": phase,
                            "seconds": 4 * rows * (0.8 if config else 1),
                            "blocks": 4,
                            "logits": rows * 248320,
                            "hidden": 5120,
                            "state_arrays": 128,
                            "batch": bool(config & 1),
                            "mb3": bool(config & 2),
                            "bit_exact": True,
                        }
                    )
    return {"status": "complete", "parity": True, "samples": samples}


def test_exact_stable_factorial_cost_and_gates():
    result = profile.analyze(fixture())
    assert len(result["comparisons"]) == 16
    for row in result["comparisons"]:
        assert row["rho_vs_AR"] == pytest.approx(row["rows"])
        assert row["paired_gain_percent"] == pytest.approx(25 if row["config"] else 0)
        assert row["speed_gate_passed"] is bool(row["config"])


def test_drift_cannot_be_promoted_despite_parity_and_positive_gain():
    report = fixture()
    for sample in report["samples"]:
        if sample["phase"] == "measure" and sample["round"] >= 4:
            sample["seconds"] *= 1.04
    result = profile.analyze(report)
    assert result["parity"] is True
    assert all(not row["speed_gate_passed"] for row in result["comparisons"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("seconds", 0),
        ("seconds", True),
        ("seconds", float("nan")),
        ("seconds", float("inf")),
        ("rows", True),
        ("config", 4),
        ("bit_exact", False),
        ("logits", 0),
        ("state_arrays", 0),
        ("phase", "partial"),
        ("blocks", True),
        ("batch", 1),
    ],
)
def test_invalid_sample_is_rejected(key, value):
    report = fixture()
    report["samples"][0][key] = value
    with pytest.raises(ValueError):
        profile.analyze(report)


def test_partial_duplicate_or_unwarmed_campaign_is_rejected():
    report = fixture()
    for key, value in [("status", "running"), ("parity", False), ("samples", [])]:
        with pytest.raises(ValueError):
            profile.analyze({**report, key: value})
    for index in [0, 2, 5]:
        broken = copy.deepcopy(report)
        broken["samples"][index] = copy.deepcopy(broken["samples"][3])
        with pytest.raises(ValueError):
            profile.analyze(broken)
    reordered = copy.deepcopy(report)
    reordered["samples"][32:34] = reversed(reordered["samples"][32:34])
    with pytest.raises(ValueError):
        profile.analyze(reordered)
