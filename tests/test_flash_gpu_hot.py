"""Compare the 162 historical Flash configurations with resident masking disabled."""

import itertools
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Ungrouped routes now inline the route index at the lookup. Fold this spelling
# only in the original straight-line block, where neither operand changes.
INLINE_ROUTE = re.compile(
    r"(uint route = sub / uint\(ROUTING_REPEAT\);\s*"
    r"uint projection = sub - route \* uint\(ROUTING_REPEAT\);\s*"
    r"uint source_base =\s*tile_map\[)sub / uint\(ROUTING_REPEAT\)(?=\])"
)


def shader_tokens(source):
    return re.findall(r"\w+|[^\s]", INLINE_ROUTE.sub(r"\g<1>route", source))


def test_route_comparison_requires_the_same_value_at_lookup():
    definition = (
        "uint route = sub / uint(ROUTING_REPEAT);\n"
        "uint projection = sub - route * uint(ROUTING_REPEAT);\n"
    )
    named = "uint source_base = tile_map[route];"
    inline = "uint source_base = tile_map[sub / uint(ROUTING_REPEAT)];"
    assert shader_tokens(definition + named) == shader_tokens(definition + inline)
    for mutation in ["sub += 1u;", "route = 0u;"]:
        assert shader_tokens(definition + mutation + named) != shader_tokens(
            definition + mutation + inline
        )
    assert shader_tokens(definition + named) != shader_tokens(
        definition + inline.replace("ROUTING_REPEAT", "OUTPUT_TILES")
    )


def test_disabled_gpu_hot_shader_matches_historical_tokens_after_route_alias_expansion():
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
            outputs.append(shader_tokens(result.stdout))
        assert outputs[0] == outputs[1], flags
