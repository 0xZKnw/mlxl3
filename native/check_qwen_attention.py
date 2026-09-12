"""Bit-exact real Qwen attention-layer check over two cached steps."""
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


def exl3(weights, prefix):
    trellis = weights[f"{prefix}.trellis"]
    return EXL3Linear(trellis, _scale(weights, prefix, "suh", "su"),
        _scale(weights, prefix, "svh", "sv"), bits=trellis.shape[-1] // 16,
        mode=_mode(weights, prefix))


def main():
    binary, model = Path(sys.argv[1]), Path(sys.argv[2])
    layer_prefix = "model.language_model.layers.3"
    attention_prefix = f"{layer_prefix}.self_attn"
    mlp_prefix = f"{layer_prefix}.mlp"
    weights = _load_all_safetensors(model)
    modules = [name for name in list_exl3_modules(model)
        if name.startswith(f"{mlp_prefix}.experts.")]
    replacements, _ = _stack_moe_experts(modules, weights)
    switch = dict(replacements)["language_model.model.layers.3.mlp.switch_mlp"]
    q_proj, k_proj, v_proj, o_proj = (exl3(weights, f"{attention_prefix}.{name}")
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
    shared_gate, shared_up, shared_down = (exl3(weights, f"{mlp_prefix}.shared_expert.{name}")
        for name in ("gate_proj", "up_proj", "down_proj"))
    input_norm = (weights[f"{layer_prefix}.input_layernorm.weight"] + 1).astype(mx.float16)
    post_norm = (weights[f"{layer_prefix}.post_attention_layernorm.weight"] + 1).astype(mx.float16)
    q_norm = (weights[f"{attention_prefix}.q_norm.weight"] + 1).astype(mx.float16)
    k_norm = (weights[f"{attention_prefix}.k_norm.weight"] + 1).astype(mx.float16)
    gate_weight = weights[f"{mlp_prefix}.gate.weight"].astype(mx.float16)
    multiplier = weights[f"{mlp_prefix}.shared_expert_gate.weight"].astype(mx.float16)
    inputs = [np.random.default_rng(2030 + i).normal(size=(1, 1, 2048)).astype(np.float16)
        for i in range(2)]
    keys = values = None
    native_keys = native_values = []
    process = subprocess.Popen([str(binary), "codec"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for offset, x_np in enumerate(inputs):
            x = mx.array(x_np)
            normalized = mx.fast.rms_norm(x, input_norm, 1e-6)
            q_gate = q_proj(normalized).reshape(1, 1, 16, 512)
            q, gate = mx.split(q_gate, 2, axis=-1)
            gate = gate.reshape(1, 1, 4096)
            q = mx.fast.rms_norm(q, q_norm, 1e-6).transpose(0, 2, 1, 3)
            key = mx.fast.rms_norm(k_proj(normalized).reshape(1, 1, 2, 256),
                k_norm, 1e-6).transpose(0, 2, 1, 3)
            value = v_proj(normalized).reshape(1, 1, 2, 256).transpose(0, 2, 1, 3)
            q = mx.fast.rope(q, 64, traditional=False, base=10_000_000.0,
                scale=1.0, offset=offset)
            key = mx.fast.rope(key, 64, traditional=False, base=10_000_000.0,
                scale=1.0, offset=offset)
            keys = key if keys is None else mx.concatenate([keys, key], axis=2)
            values = value if values is None else mx.concatenate([values, value], axis=2)
            attention = mx.fast.scaled_dot_product_attention(q, keys, values,
                scale=256 ** -0.5).transpose(0, 2, 1, 3).reshape(1, 1, 4096)
            hidden = x + o_proj(attention * mx.sigmoid(gate))
            mlp_input = mx.fast.rms_norm(hidden, post_norm, 1e-6).reshape(1, 2048)
            probabilities = mx.softmax(mlp_input @ gate_weight.T, axis=-1, precise=True)
            selected, scores = router_topk(probabilities, 8, normalize=True)
            routed = switch(mlp_input, selected, scores=scores)
            shared_gate_value = shared_gate(mlp_input)
            shared = shared_down(shared_gate_value * mx.sigmoid(shared_gate_value)
                * shared_up(mlp_input))
            expected = hidden + (routed + mx.sigmoid(mlp_input @ multiplier.T)
                * shared).reshape(1, 1, 2048)
            request = {"op": "mlx-qwen-attn-layer", "path": str(model), "layer": 3,
                "cols": 2048, "key_heads": 16, "value_heads": 2, "value_dims": 256,
                "top_k": 8, "x": x_np.view(np.uint16).ravel().tolist(),
                "keys": native_keys, "values": native_values}
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            assert "error" not in response, response
            actual = response["data"]
            np.testing.assert_array_equal(np.asarray(actual["output"], dtype=np.uint16),
                np.asarray(expected).view(np.uint16).ravel(), err_msg=f"layer output step {offset}")
            np.testing.assert_array_equal(np.asarray(actual["keys"], dtype=np.uint16),
                np.asarray(keys).view(np.uint16).ravel(), err_msg=f"keys step {offset}")
            np.testing.assert_array_equal(np.asarray(actual["values"], dtype=np.uint16),
                np.asarray(values).view(np.uint16).ravel(), err_msg=f"values step {offset}")
            native_keys, native_values = actual["keys"], actual["values"]
    finally:
        process.stdin.close()
        process.wait(timeout=10)
        stderr = process.stderr.read()
        process.stdout.close()
        process.stderr.close()
    assert process.returncode == 0, stderr
    print(json.dumps({"status": "passed", "model": model.name, "layer": 3,
        "steps": 2, "comparison": "exact output and KV cache bit patterns",
        "performance": "not measured"}))


if __name__ == "__main__":
    main()
