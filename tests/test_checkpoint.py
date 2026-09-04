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


def test_gemma_experts_mapping():
    assert _expert_spec('model.language_model.layers.29.experts.127.down_proj') == (
        'gemma4', 29, 127, 'down_proj',
        'language_model.model.layers.29.experts.switch_glu',
    )


def test_padded_linear_matches_cropped_dense_reference():
    import numpy as np
    from mlxl3.linear import EXL3Linear
    from mlxl3.codec.trellis import pack_trellis

    rng = np.random.default_rng(44)
    codes = rng.integers(0, 8, (8, 8, 256), dtype=np.uint16)
    linear = EXL3Linear(mx.array(pack_trellis(codes, 3)),
                       mx.ones((128,), dtype=mx.float16),
                       mx.ones((128,), dtype=mx.float16),
                       bits=3, logical_shape=(96, 112))
    weight = linear.reconstruct(cache=False)[:112, :96]
    for rows in (1, 4):
        x = mx.array(rng.normal(0, 0.02, (rows, 112)).astype(np.float16))
        actual, expected = linear(x), x @ weight
        mx.eval(actual, expected)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=0.005, rtol=0.02)


def test_padded_group_preserves_output_widths(monkeypatch):
    import numpy as np
    import mlxl3.linear as linear_module
    from mlxl3.linear import EXL3Linear, EXL3LinearGroup
    from mlxl3.codec.trellis import pack_trellis
    rng = np.random.default_rng(45)
    linears = tuple(EXL3Linear(
        mx.array(pack_trellis(rng.integers(0, 8, (8, 8, 256), dtype=np.uint16), 3)),
        mx.ones((128,), dtype=mx.float16), mx.ones((128,), dtype=mx.float16),
        bits=3, logical_shape=(96, 128)) for _ in range(2))
    group = EXL3LinearGroup(linears)
    for rows in (1, 4):
        x = mx.array(rng.normal(0, 0.02, (rows, 128)).astype(np.float16))
        monkeypatch.setattr(linear_module, '_USE_PADDED_GROUPS', True)
        actual = group._evaluate(x)
        expected = tuple(layer(x) for layer in linears)
        mx.eval(actual, expected)
        for a, b in zip(actual, expected):
            assert a.shape == (rows, 96)
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=0.005, rtol=0.02)


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
