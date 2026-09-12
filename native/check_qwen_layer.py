"""Bit-exact real Qwen layer-0 check assembled without a full model."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
from mlx import nn
import numpy as np

from mlxl3.checkpoint import _load_all_safetensors, _mode, _scale, _stack_moe_experts, list_exl3_modules
from mlxl3.linear import EXL3Linear
from mlxl3.moe import router_topk
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_next import _precise_swiglu


def exl3(weights, prefix):
    trellis = weights[f"{prefix}.trellis"]
    return EXL3Linear(trellis, _scale(weights, prefix, "suh", "su"),
        _scale(weights, prefix, "svh", "sv"), bits=trellis.shape[-1] // 16,
        mode=_mode(weights, prefix))


def main():
    binary, model = Path(sys.argv[1]), Path(sys.argv[2])
    layer_prefix = "model.language_model.layers.0"
    attention_prefix = f"{layer_prefix}.linear_attn"
    mlp_prefix = f"{layer_prefix}.mlp"
    weights = _load_all_safetensors(model)
    modules = [name for name in list_exl3_modules(model)
        if name.startswith(f"{mlp_prefix}.experts.")]
    replacements, _ = _stack_moe_experts(modules, weights)
    switch = dict(replacements)["language_model.model.layers.0.mlp.switch_mlp"]
    qkv, z, attention_output = (exl3(weights, f"{attention_prefix}.{name}")
        for name in ("in_proj_qkv", "in_proj_z", "out_proj"))
    shared_gate, shared_up, shared_down = (exl3(weights, f"{mlp_prefix}.shared_expert.{name}")
        for name in ("gate_proj", "up_proj", "down_proj"))
    input_norm = (weights[f"{layer_prefix}.input_layernorm.weight"] + 1).astype(mx.float16)
    post_norm = (weights[f"{layer_prefix}.post_attention_layernorm.weight"] + 1).astype(mx.float16)
    conv_weight = weights[f"{attention_prefix}.conv1d.weight"].moveaxis(2, 1).astype(mx.float16)
    a_weight = weights[f"{attention_prefix}.in_proj_a.weight"].astype(mx.float16)
    b_weight = weights[f"{attention_prefix}.in_proj_b.weight"].astype(mx.float16)
    gate_weight = weights[f"{mlp_prefix}.gate.weight"].astype(mx.float16)
    multiplier = weights[f"{mlp_prefix}.shared_expert_gate.weight"].astype(mx.float16)
    norm = weights[f"{attention_prefix}.norm.weight"].astype(mx.float16)
    a_log = weights[f"{attention_prefix}.A_log"].astype(mx.float32)
    dt_bias = weights[f"{attention_prefix}.dt_bias"].astype(mx.float16)
    x_np = np.asarray(
        weights["model.language_model.embed_tokens.weight"][1].astype(mx.float16)
    ).reshape(1, 1, 2048)
    x = mx.array(x_np)
    normalized_input = mx.fast.rms_norm(x, input_norm, 1e-6)
    mixed = qkv(normalized_input)
    z_value = z(normalized_input).reshape(1, 1, 32, 128)
    a = normalized_input @ a_weight.T
    b = normalized_input @ b_weight.T
    conv_state = mx.zeros((1, 3, 8192), dtype=mx.float16)
    conv_input = mx.concatenate([conv_state, mixed], axis=1)
    conv_state = conv_input[:, -3:, :]
    conv = nn.silu(mx.conv1d(conv_input, conv_weight, groups=8192))
    q, key, value = mx.split(conv, [2048, 4096], axis=-1)
    q = (128 ** -0.5) ** 2 * mx.fast.rms_norm(q.reshape(1, 1, 16, 128), None, 1e-6)
    key = 128 ** -0.5 * mx.fast.rms_norm(key.reshape(1, 1, 16, 128), None, 1e-6)
    value = value.reshape(1, 1, 32, 128)
    attention, state = gated_delta_update(q, key, value, a, b, a_log, dt_bias, None)
    attention = mx.fast.rms_norm(attention, norm, 1e-6)
    attention = attention_output(
        _precise_swiglu(attention, z_value, attention).reshape(1, 1, 4096)
    )
    hidden = x + attention
    mlp_input = mx.fast.rms_norm(hidden, post_norm, 1e-6).reshape(1, 2048)
    probabilities = mx.softmax(mlp_input @ gate_weight.T, axis=-1, precise=True)
    selected, scores = router_topk(probabilities, 8, normalize=True)
    routed = switch(mlp_input, selected, scores=scores)
    gate = shared_gate(mlp_input)
    shared = shared_down(gate * mx.sigmoid(gate) * shared_up(mlp_input))
    expected = hidden + (routed + mx.sigmoid(mlp_input @ multiplier.T) * shared).reshape(1, 1, 2048)
    request = {"op": "mlx-qwen-layer", "path": str(model), "layer": 0,
        "cols": 2048, "key_heads": 16, "value_heads": 32, "value_dims": 128,
        "conv_length": 4, "top_k": 8, "x": x_np.view(np.uint16).ravel().tolist()}
    process = subprocess.run([str(binary), "codec"], input=json.dumps(request) + "\n",
        capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stderr
    response = json.loads(process.stdout)
    assert "error" not in response, response
    actual = response["data"]
    np.testing.assert_array_equal(np.asarray(actual["output"], dtype=np.uint16),
        np.asarray(expected).view(np.uint16).ravel())
    np.testing.assert_array_equal(np.asarray(actual["conv_state"], dtype=np.uint16),
        np.asarray(conv_state).view(np.uint16).ravel())
    np.testing.assert_array_equal(np.asarray(actual["state"], dtype=np.float32).view(np.uint32),
        np.asarray(state).astype(np.float32).ravel().view(np.uint32))
    print(json.dumps({"status": "passed", "model": model.name, "layer": 0,
        "comparison": "exact layer output and cache bit patterns", "performance": "not measured"}))


if __name__ == "__main__":
    main()
