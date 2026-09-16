import runpy
from pathlib import Path

import mlx.core as mx
import numpy as np


def test_ling_expert_calibration_uses_each_experts_own_weights(tmp_path):
    from mlx_lm.models.switch_layers import SwitchGLU
    from mlx_lm.models.activations import swiglu

    adapter = runpy.run_path(str(Path(__file__).parents[1] / "scripts/quantize_ling.py"))
    switch = SwitchGLU(128, 128, 2)
    inputs = mx.ones((8, 128), dtype=mx.bfloat16) * 0.125
    switch.gate_proj.weight = mx.stack([mx.eye(128), mx.eye(128) * 2]).astype(mx.bfloat16)
    switch.up_proj.weight = mx.stack([mx.eye(128) * 3, mx.eye(128) * 4]).astype(mx.bfloat16)
    prefix = "model.layers.1.mlp"
    selected = {f"{prefix}.experts.{e}.{p}" for e in range(2)
                for p in ("gate_proj", "up_proj", "down_proj")}
    adapter["capture_experts"](tmp_path, prefix, switch, inputs, selected)
    for expert in range(2):
        base = f"{prefix}.experts.{expert}"
        expected = swiglu(inputs @ switch.gate_proj.weight[expert].T,
                          inputs @ switch.up_proj.weight[expert].T).astype(mx.float16)
        np.testing.assert_array_equal(np.load(tmp_path / (base + ".down_proj.npy")), np.array(expected))
        np.testing.assert_array_equal(np.load(tmp_path / (base + ".gate_proj.npy")),
                                      np.array(inputs.astype(mx.float16)))
    assert len({(tmp_path / (key + ".npy")).stat().st_ino for key in selected
                if key.endswith(("gate_proj", "up_proj"))}) == 1
    assert len(list(tmp_path.glob("*.npy"))) == 6
    mapping = adapter["ActivationFiles"](tmp_path)
    assert set(mapping) == selected
    for key in selected:
        np.testing.assert_array_equal(mapping[key], np.load(tmp_path / (key + ".npy")))
    assert mapping.read.cache_info().maxsize == 8


def test_measurement_groups_preserve_all_candidates(tmp_path):
    adapter = runpy.run_path(str(Path(__file__).parents[1] / "scripts/quantize_ling.py"))
    groups = []
    def measure(source, plan, modules, **kwargs):
        groups.append((list(modules), kwargs["checkpoint_path"]))
        return {"records": [{"module": key, "score": 0.01, "k": 4,
                             "candidate_bits": 4, "hessian_shrinkage": 0.0} for key in modules]}
    keys = [f"layer{i}" for i in range(70)]
    result = adapter["measure_in_groups"](measure, "source", "plan", keys,
        checkpoint_path=tmp_path / "measurements.json", score_metric="output_rel_rms", candidate_bits=[4])
    assert [key for group, _ in groups for key in group] == keys
    assert max(len(group) for group, _ in groups) == 32
    assert len({path for _, path in groups}) == 3
    assert result["module_count"] == result["candidate_count"] == 70
