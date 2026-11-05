"""Disabled resident masking must preserve every existing shader configuration."""

import itertools
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_disabled_gpu_hot_shader_has_identical_preprocessed_tokens():
    clang = shutil.which("clang")
    if clang is None:
        pytest.skip("clang unavailable; control shader not verified")
    reference = ROOT / "docs/measurements/flash-next-ssd-impl/opt-56-hybrid-hot/reference.metal"
    candidate = ROOT / "native/shaders/_qmv_mapped_tile_kernel.metal"
    for k, nt, mb, layout, accumulation in itertools.product(
        [2, 3, 5], [1, 2, 4], [1, 2, 3], [(0, 0), (1, 0), (1, 1)], [0, 1]
    ):
        flags = {
            "MLXL3_K3_WINDOW_DECODE": int(k == 3),
            "MLXL3_QMV_NT": nt,
            "MLXL3_QMV_MB": mb,
            "MLXL3_BATCH_ROWS": int(mb > 1),
            "EXPERT_MAP": layout[0],
            "MLXL3_SLOT_MAJOR": layout[1],
            "MLXL3_REFERENCE_ACCUMULATION": accumulation,
        }
        outputs = []
        for source in [reference, candidate]:
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
