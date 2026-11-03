import copy
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calibrate_flash_draft as calibration


def arms():
    result = []
    for prefix in range(4):
        runs = []
        for prompt in ("0000000000000000", "1111111111111111"):
            trace = [
                {
                    "drafted": 3,
                    "selected": prefix,
                    "accepted": min(prefix, i % 4),
                    "round_seconds": (prefix + 1) / 16,
                    "confidence": [0.9, 0.8, 0.7],
                }
                for i in range(16)
            ]
            runs.append({"warmup": False, "prompt_id": prompt, "trace": trace})
        result.append(
            {
                "status": "complete_calibration_arm",
                "identity": {"fixture": 1},
                "forced_prefix": prefix,
                "runs": runs,
            }
        )
    return result


def test_sts_is_identity_at_unit_temperature_and_preserves_probability_order():
    for temperature in calibration.TEMPERATURES:
        values = [calibration.calibrated_probability(p, temperature) for p in (0, 0.1, 0.5, 0.9, 1)]
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in values)
        assert values == sorted(values)
    assert calibration.calibrated_probability(0.25, 1) == pytest.approx(0.25)
    assert calibration.calibrated_probability(0.75, 1) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "p,t", [(math.nan, 1), (-1, 1), (2, 1), (0.5, 0), (0.5, math.inf), (True, 1)]
)
def test_sts_rejects_invalid_inputs(p, t):
    with pytest.raises(ValueError):
        calibration.calibrated_probability(p, t)


def test_ece_matches_hand_computed_bin_errors_and_rejects_partial_labels():
    assert calibration.prefix_ece([0.25, 0.75], [0, 1]) == 0.25
    assert calibration.prefix_ece([0.0, 1.0], [0, 1]) == 0
    for p, y in [([], []), ([0.5], []), ([math.nan], [0]), ([0.5], [2])]:
        with pytest.raises(ValueError):
            calibration.prefix_ece(p, y)


def test_profile_fits_cumulative_survival_and_keeps_all_costs_and_identities():
    inputs = arms()
    original = copy.deepcopy(inputs)
    profile = calibration.build_profile(inputs)
    assert inputs == original
    assert profile["round_seconds"] == [1 / 16, 2 / 16, 3 / 16, 4 / 16]
    assert profile["cost_observations"] == [32] * 4 and profile["calibration_blocks"] == 32
    assert profile["calibration_prompt_ids"] == ["0000000000000000", "1111111111111111"]
    assert (
        profile["identity"] == {"fixture": 1} and profile["status"] == "calibrated_greedy_profile"
    )
    assert all(0.25 <= t <= 8 for t in profile["temperatures"])
    assert profile["calibration_ece"][0]["after_ece"] <= profile["calibration_ece"][0]["before_ece"]


@pytest.mark.parametrize(
    "bad",
    [
        "missing",
        "identity",
        "incomplete",
        "empty",
        "nan",
        "accepted",
        "confidence",
        "small",
        "one_prompt",
    ],
)
def test_profile_rejects_invalid_or_incomplete_model_measurements(bad):
    inputs = arms()
    if bad == "missing":
        inputs.pop()
    elif bad == "identity":
        inputs[1]["identity"] = {"fixture": 2}
    elif bad == "incomplete":
        inputs[0]["status"] = "running"
    elif bad == "empty":
        inputs[0]["runs"][0]["trace"] = []
    elif bad == "nan":
        inputs[0]["runs"][0]["trace"][0]["round_seconds"] = math.nan
    elif bad == "accepted":
        inputs[0]["runs"][0]["trace"][0]["accepted"] = 1
    elif bad == "confidence":
        inputs[0]["runs"][0]["trace"][0]["confidence"] = [0.5]
    elif bad == "small":
        inputs[3]["runs"][0]["trace"].pop()
    else:
        for arm in inputs:
            arm["runs"][1]["prompt_id"] = "0000000000000000"
    with pytest.raises(ValueError):
        calibration.build_profile(inputs)


def test_cli_records_failure_and_refuses_to_overwrite_existing_profiles(tmp_path, monkeypatch):
    inputs = []
    for index, data in enumerate(arms()):
        path = tmp_path / f"arm{index}.json"
        path.write_text(json.dumps(data))
        inputs.append(str(path))
    output = tmp_path / "profile.json"
    monkeypatch.setattr(sys, "argv", ["calibrate", *inputs, "--output", str(output)])
    calibration.main()
    previous = output.read_bytes()
    with pytest.raises(FileExistsError):
        calibration.main()
    assert output.read_bytes() == previous
    output.unlink()
    Path(inputs[0]).write_text('{"status":"incomplete"}')
    with pytest.raises(ValueError):
        calibration.main()
    failed = json.loads(output.read_text())
    assert failed["status"] == "failed_or_interrupted" and "round_seconds" not in failed
