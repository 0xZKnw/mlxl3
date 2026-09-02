from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlxl3.checkpoint import (
    _expert_spec,
    _stack_moe_experts,
    list_exl3_modules,
    quantization_config,
)

FIXTURE = Path("models/fixtures/Qwen3.5-0.8B-EXL3-3.0bpw")


@pytest.mark.skipif(not FIXTURE.exists(), reason="local EXL3 fixture not installed")
def test_fixture_is_standard_exl3_3bpw() -> None:
    config = quantization_config(FIXTURE)
    modules = list_exl3_modules(FIXTURE)
    assert config["bits"] == 3.0
    assert len(modules) == 151
    assert "model.language_model.layers.0.mlp.gate_proj" in modules


def test_qwen35_wrapped_expert_path_maps_to_mlx_switch() -> None:
    assert _expert_spec(
        "model.language_model.layers.7.mlp.experts.19.down_proj"
    ) == (
        "ling",
        7,
        19,
        "down_proj",
        "language_model.model.layers.7.mlp.switch_mlp",
    )


def test_eager_expert_stacking_releases_serialized_sources() -> None:
    prefixes = [
        f"model.language_model.layers.0.mlp.experts.0.{projection}"
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    weights: dict[str, mx.array] = {}
    for prefix in prefixes:
        weights[f"{prefix}.trellis"] = mx.zeros((1, 1, 48), dtype=mx.uint16)
        weights[f"{prefix}.suh"] = mx.ones((16,), dtype=mx.float16)
        weights[f"{prefix}.svh"] = mx.ones((16,), dtype=mx.float16)

    replacements, consumed = _stack_moe_experts(prefixes, weights, materialize=True)

    assert replacements[0][0] == "language_model.model.layers.0.mlp.switch_mlp"
    assert consumed == set(prefixes)
    assert not weights
