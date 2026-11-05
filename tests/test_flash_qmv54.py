"""Keep EXL3 controls unchanged and reject unstable or incomplete micro results."""

import itertools
import re
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/measurements/flash-next-ssd-impl/opt-54-qmv"
QUALIFY = runpy.run_path(str(DATA / "qualify-micro.py"))["qualify"]
FIXTURE = runpy.run_path(str(ROOT / "tests/test_flash_pair_kernel.py"))["micro_fixture"]


def test_reference_shader_has_identical_preprocessed_tokens():
    clang = shutil.which("clang")
    if clang is None:
        pytest.skip("clang unavailable; unchanged control shader not verified")
    for k, nt, mb, layout, reference in itertools.product(
        [2, 3, 5], [1, 2, 4], [1, 2, 3], [(0, 0), (1, 0), (1, 1)], [0, 1]
    ):
        flags = {
            "MLXL3_K3_WINDOW_DECODE": int(k == 3),
            "MLXL3_QMV_NT": nt,
            "MLXL3_QMV_MB": mb,
            "MLXL3_BATCH_ROWS": int(mb > 1),
            "EXPERT_MAP": layout[0],
            "MLXL3_SLOT_MAJOR": layout[1],
            "MLXL3_REFERENCE_ACCUMULATION": reference,
        }
        outputs = []
        for source in [
            DATA / "reference.metal",
            DATA / "candidate-source/native/shaders/_qmv_mapped_tile_kernel.metal",
        ]:
            result = subprocess.run(
                [
                    clang,
                    "-E",
                    "-P",
                    "-x",
                    "c++",
                    *[f"-D{k}={v}" for k, v in flags.items()],
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            outputs.append(re.findall(r"\w+|[^\s]", result.stdout))
        assert outputs[0] == outputs[1], flags


@pytest.mark.parametrize("case", ["eligible", "slow_b1", "drift", "incomplete", "short", "nan"])
def test_micro_requires_gain_in_every_affected_shape(case):
    data = FIXTURE()
    data["forwards_per_sample"] = 256
    for cell in data["records"]:
        for sample in cell["samples"]:
            sample["seconds"] = 0.98 if sample["candidate"] else 1.0
    if case == "slow_b1":
        for sample in data["records"][0]["samples"]:
            sample["seconds"] = 1.0
    elif case == "drift":
        data["records"][0]["samples"][0]["seconds"] = 1.1
    elif case == "incomplete":
        data["status"] = "incomplete"
    elif case == "short":
        data["forwards_per_sample"] = 16
    elif case == "nan":
        data["records"][0]["samples"][0]["seconds"] = float("nan")
    if case in ["incomplete", "short", "nan"]:
        with pytest.raises(AssertionError):
            QUALIFY(data)
    else:
        result = QUALIFY(data)
        assert (
            result["status"]
            == {
                "eligible": "eligible_micro_only",
                "slow_b1": "rejected_micro",
                "drift": "inconclusive_drift",
            }[case]
        )
        assert result["full_model_eligible"] == (case == "eligible")
        assert result["throughput_gain_confirmed"] is False
