"""One real Qwen MoE layer parity check without loading a second full model."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
import numpy as np

from mlxl3.checkpoint import _load_all_safetensors, _mode, _scale, _stack_moe_experts, list_exl3_modules
from mlxl3.linear import EXL3Linear
from mlxl3.moe import router_topk


def linear(weights, prefix):
    trellis = weights[f"{prefix}.trellis"]
    return EXL3Linear(trellis, _scale(weights, prefix, "suh", "su"),
        _scale(weights, prefix, "svh", "sv"), bits=trellis.shape[-1] // 16,
        mode=_mode(weights, prefix))


def main():
    binary, model = Path(sys.argv[1]), Path(sys.argv[2])
    prefix = "model.language_model.layers.0.mlp"
    weights = _load_all_safetensors(model)
    modules = [name for name in list_exl3_modules(model) if name.startswith(f"{prefix}.experts.")]
    replacements, _ = _stack_moe_experts(modules, weights)
    switch = dict(replacements)["language_model.model.layers.0.mlp.switch_mlp"]
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(1, 2048)).astype(np.float16)
    value = mx.array(x)
    gate = weights[f"{prefix}.gate.weight"].astype(mx.float16)
    probabilities = mx.softmax(value @ gate.T, axis=-1, precise=True)
    selected, scores = router_topk(probabilities, 8, normalize=True)
    routed = switch(value, selected, scores=scores)
    shared_gate = linear(weights, f"{prefix}.shared_expert.gate_proj")(value)
    shared = linear(weights, f"{prefix}.shared_expert.down_proj")(
        shared_gate * mx.sigmoid(shared_gate)
        * linear(weights, f"{prefix}.shared_expert.up_proj")(value))
    multiplier = weights[f"{prefix}.shared_expert_gate.weight"].astype(mx.float16)
    expected = routed + mx.sigmoid(value @ multiplier.T) * shared
    process = subprocess.Popen([str(binary), "codec"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    request = {"op": "mlx-qwen-moe", "path": str(model), "layer": 0,
        "cols": 2048, "value_dims": 512, "key_heads": 256, "top_k": 8,
        "x": x.view(np.uint16).ravel().tolist()}
    stdout, stderr = process.communicate(json.dumps(request) + "\n", timeout=180)
    assert process.returncode == 0, stderr
    response = json.loads(stdout)
    assert "error" not in response, response
    actual = np.asarray(response["data"], dtype=np.uint16)
    np.testing.assert_array_equal(actual, np.asarray(expected).view(np.uint16).ravel())
    print(json.dumps({"status": "passed", "model": model.name, "layer": 0,
        "comparison": "exact float16 bit patterns", "performance": "not measured"}))


if __name__ == "__main__":
    main()
