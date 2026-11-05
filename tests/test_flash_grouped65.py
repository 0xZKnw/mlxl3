import copy
import runpy
from pathlib import Path

import pytest

qualify = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "docs/measurements/flash-next-ssd-impl/opt-65-grouped-exl3/qualify.py"
    )
)["qualify"]


def fixture():
    records = []
    for layer in [0, 23, 47]:
        for rows in [1, 2, 4, 8]:
            for mixed in [False, True]:
                records.append(
                    {
                        "layer": layer,
                        "rows": rows,
                        "mixed": mixed,
                        "route_ids": list(range(10)) * rows,
                        "known_slots": list(range(5 if mixed else 10)) * rows,
                        "io_bytes_after_warm": 1_248_012,
                        "samples": [
                            {
                                "pair": pair,
                                "candidate": candidate,
                                "seconds": 0.98 if candidate and rows > 1 else 1.0,
                            }
                            for pair in range(4)
                            for candidate in [False, True]
                        ],
                    }
                )
    return {
        "status": "complete",
        "parity": "byte_exact_independent",
        "no_timed_io": True,
        "samples_per_cell": 8,
        "forwards_per_sample": 128,
        "records": records,
    }


def test_complete_stable_component_can_only_qualify_a_full_model_trial():
    result = qualify(fixture())
    assert result["status"] == "eligible_micro_only" and result["full_model_eligible"]
    assert not result["throughput_gain_confirmed"] and len(result["cells"]) == 24


def test_stable_regression_and_unfiltered_outlier_prevent_promotion():
    data = fixture()
    for sample in data["records"][2]["samples"]:
        if sample["candidate"]:
            sample["seconds"] = 1.02
    assert qualify(data)["status"] == "rejected_micro"
    data["records"][2]["samples"][1]["seconds"] = 2.0
    assert qualify(data)["status"] == "inconclusive_drift"


def gated_fixture():
    data = fixture()
    data["forwards_per_sample"] = 1024
    for cell in data["records"]:
        cell["shared_enabled"] = len(cell["known_slots"]) >= 16 and cell["rows"] > 1
    return data


def test_selective_gate_uses_real_decode_savings_and_measured_enablement():
    data = gated_fixture()
    # Twenty routes with four shared decodes: the exact 20% threshold.
    cell = data["records"][2]
    cell["known_slots"] = list(range(16)) + list(range(4))
    result = qualify(data, gated=True)
    assert result["full_model_eligible"] and result["cells"][2]["grouped"]
    # One shared decode in twenty: fall back, with no required speedup.
    cell["known_slots"] = list(range(19)) + [0]
    cell["shared_enabled"] = False
    for sample in cell["samples"]:
        sample["seconds"] = 1.0
    result = qualify(data, gated=True)
    assert result["full_model_eligible"] and not result["cells"][2]["grouped"]


@pytest.mark.parametrize("routes,enabled", [(15, False), (16, True)])
def test_selective_minimum_route_count(routes, enabled):
    data = gated_fixture()
    cell = data["records"][2]
    cell["known_slots"] = [0] * routes
    cell["shared_enabled"] = enabled
    assert qualify(data, gated=True)["cells"][2]["grouped"] is enabled


@pytest.mark.parametrize("bad", ["enablement", "nonbool", "short_sample"])
def test_selective_inconsistent_enablement_or_protocol_is_rejected(bad):
    data = gated_fixture()
    if bad == "enablement":
        data["records"][0]["shared_enabled"] = True
    elif bad == "nonbool":
        data["records"][0]["shared_enabled"] = 0
    else:
        data["forwards_per_sample"] = 128
    with pytest.raises(AssertionError):
        qualify(data, gated=True)


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "duplicate",
        "nan",
        "zero",
        "invalid_slot",
        "false_parity",
        "duplicate_sample",
        "nonbool",
    ],
)
def test_incomplete_invalid_or_divergent_data_is_rejected(bad):
    data = copy.deepcopy(fixture())
    if bad == "missing":
        data["records"].pop()
    elif bad == "duplicate":
        data["records"][1] = data["records"][0]
    elif bad == "nan":
        data["records"][0]["samples"][0]["seconds"] = float("nan")
    elif bad == "zero":
        data["records"][0]["samples"][0]["seconds"] = 0.0
    elif bad == "invalid_slot":
        data["records"][0]["known_slots"][0] = 53
    elif bad == "false_parity":
        data["parity"] = False
    elif bad == "duplicate_sample":
        data["records"][0]["samples"][1] = data["records"][0]["samples"][0]
    else:
        data["records"][0]["samples"][1]["candidate"] = 1
    with pytest.raises(AssertionError):
        qualify(data)
