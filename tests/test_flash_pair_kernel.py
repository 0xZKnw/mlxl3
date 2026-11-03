"""Audit the retired pair shader and qualify its fixed micro measurements."""

import itertools
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OLD = (
    ROOT
    / "docs/measurements/flash-next-ssd-impl/opt-48-paired-experts/original-source/native/shaders/_qmv_mapped_tile_kernel.metal"
)
NEW = OLD.parents[3] / "candidate-source/native/shaders/_qmv_mapped_tile_kernel.metal"


def test_disabled_pair_shader_preserves_preprocessed_tokens():
    clang = shutil.which("clang")
    if clang is None:
        pytest.skip("clang preprocessor unavailable; original shader equivalence unverified")
    checked = 0
    for k, nt, mb, layout, reference in itertools.product(
        [2, 3, 5], [1, 2, 4], [1, 2, 3], [(0, 0), (1, 0), (1, 1)], [0, 1]
    ):
        expert, slots = layout
        flags = {
            "MLXL3_K3_WINDOW_DECODE": int(k == 3),
            "MLXL3_QMV_NT": nt,
            "MLXL3_QMV_MB": mb,
            "MLXL3_BATCH_ROWS": int(mb > 1),
            "EXPERT_MAP": expert,
            "MLXL3_SLOT_MAJOR": slots,
            "MLXL3_REFERENCE_ACCUMULATION": reference,
        }
        command = [
            clang,
            "-E",
            "-P",
            "-x",
            "c++",
            *[f"-D{key}={value}" for key, value in flags.items()],
        ]
        tokens = []
        for source in [OLD, NEW]:
            result = subprocess.run(
                [*command, str(source)], capture_output=True, text=True, timeout=10, check=True
            )
            tokens.append(re.findall(r"\w+|[^\s]", result.stdout))
        assert tokens[0] == tokens[1], flags
        checked += 1
    assert checked == 162


def micro_fixture():
    return {
        "status": "complete",
        "parity": "byte_exact_independent",
        "no_timed_io": True,
        "samples_per_cell": 8,
        "forwards_per_sample": 16,
        "records": [
            {
                "layer": layer,
                "rows": rows,
                "route_ids": list(range(10)) * rows,
                "io_bytes_after_warm": 10 * 1248012,
                "samples": [
                    {
                        "pair": pair,
                        "candidate": candidate,
                        "seconds": 0.98 if candidate and rows == 2 else 1.0,
                    }
                    for pair in range(4)
                    for candidate in [False, True]
                ],
            }
            for layer in [0, 23, 47]
            for rows in [1, 2]
        ],
    }


def qualifier():
    import runpy

    return runpy.run_path(str(OLD.parents[3] / "qualify-micro.txt"))["qualify"]


@pytest.mark.parametrize("case", ["pass", "slow_pair", "slow_single", "drift"])
def test_micro_thresholds_do_not_claim_a_global_speed_gain(case):
    data = micro_fixture()
    if case != "pass":
        cell = data["records"][1 if case == "slow_pair" else 0]
        for sample in cell["samples"]:
            if sample["candidate"]:
                sample["seconds"] = {"slow_pair": 0.995, "slow_single": 1.02, "drift": 1.1}[case]
                if case == "drift":
                    break
    result = qualifier()(data)
    assert (
        result["status"]
        == {
            "pass": "eligible_micro_only",
            "slow_pair": "rejected_micro",
            "slow_single": "rejected_micro",
            "drift": "inconclusive_drift",
        }[case]
    )
    assert result["full_model_eligible"] == (case == "pass")
    assert result["throughput_gain_confirmed"] is False


@pytest.mark.parametrize(
    "case",
    [
        "incomplete",
        "parity",
        "io",
        "missing",
        "duplicate_cell",
        "missing_sample",
        "duplicate_sample",
        "nan",
        "zero",
        "bool",
        "bad_slot",
        "bad_route",
        "no_pair",
    ],
)
def test_micro_invalid_or_partial_data_never_qualifies(case):
    data = micro_fixture()
    cell = data["records"][1]
    if case == "incomplete":
        data["status"] = "incomplete"
    elif case == "parity":
        data["parity"] = "mismatch"
    elif case == "io":
        data["no_timed_io"] = False
    elif case == "missing":
        data["records"].pop()
    elif case == "duplicate_cell":
        data["records"][0] = cell
    elif case == "missing_sample":
        cell["samples"].pop()
    elif case == "duplicate_sample":
        cell["samples"][0] = cell["samples"][1]
    elif case in ["nan", "zero", "bool"]:
        cell["samples"][0]["seconds"] = {"nan": float("nan"), "zero": 0, "bool": True}[case]
    elif case == "bad_slot":
        cell["route_ids"][0] = 512
    elif case == "bad_route":
        cell["route_ids"][0] = cell["route_ids"][1]
    elif case == "no_pair":
        cell["route_ids"] = list(range(20))
    with pytest.raises(AssertionError):
        qualifier()(data)
