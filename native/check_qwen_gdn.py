"""Bit-exact real Qwen Gated DeltaNet check over two recurrent steps."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import mlx.core as mx
from mlx import nn
import numpy as np

from mlxl3.checkpoint import _load_all_safetensors, _mode, _scale
from mlxl3.linear import EXL3Linear
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_next import _precise_swiglu


def exl3(weights, prefix):
    trellis = weights[f"{prefix}.trellis"]
    return EXL3Linear(trellis, _scale(weights, prefix, "suh", "su"),
        _scale(weights, prefix, "svh", "sv"), bits=trellis.shape[-1] // 16,
        mode=_mode(weights, prefix))


def main():
    binary, model = Path(sys.argv[1]), Path(sys.argv[2])
    prefix = "model.language_model.layers.0.linear_attn"
    weights = _load_all_safetensors(model)
    qkv, z, output = (exl3(weights, f"{prefix}.{name}")
        for name in ("in_proj_qkv", "in_proj_z", "out_proj"))
    a_weight = weights[f"{prefix}.in_proj_a.weight"].astype(mx.float16)
    b_weight = weights[f"{prefix}.in_proj_b.weight"].astype(mx.float16)
    conv_weight = weights[f"{prefix}.conv1d.weight"].moveaxis(2, 1).astype(mx.float16)
    norm = weights[f"{prefix}.norm.weight"].astype(mx.float16)
    a_log = weights[f"{prefix}.A_log"].astype(mx.float32)
    dt_bias = weights[f"{prefix}.dt_bias"].astype(mx.float16)
    rng = np.random.default_rng(2027)
    inputs = [rng.normal(size=(1, 1, 2048)).astype(np.float16) for _ in range(2)]
    conv_state = mx.zeros((1, 3, 8192), dtype=mx.float16)
    state = None
    process = subprocess.Popen([str(binary), "codec"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for step, x_np in enumerate(inputs):
            x = mx.array(x_np)
            mixed = qkv(x)
            z_value = z(x).reshape(1, 1, 32, 128)
            a = x @ a_weight.T
            b = x @ b_weight.T
            conv_input = mx.concatenate([conv_state, mixed], axis=1)
            conv_state = conv_input[:, -3:, :]
            conv = nn.silu(mx.conv1d(conv_input, conv_weight, groups=8192))
            q, key, value = mx.split(conv, [2048, 4096], axis=-1)
            q = q.reshape(1, 1, 16, 128)
            key = key.reshape(1, 1, 16, 128)
            value = value.reshape(1, 1, 32, 128)
            q = (128 ** -0.5) ** 2 * mx.fast.rms_norm(q, None, 1e-6)
            key = 128 ** -0.5 * mx.fast.rms_norm(key, None, 1e-6)
            out, state = gated_delta_update(q, key, value, a, b, a_log, dt_bias, state)
            normalized = mx.fast.rms_norm(out, norm, 1e-6)
            expected = output(
                _precise_swiglu(out, z_value, normalized).reshape(1, 1, 4096)
            )
            request = {"op": "mlx-qwen-gdn", "path": str(model), "layer": 0,
                "cols": 2048, "key_heads": 16, "value_heads": 32, "value_dims": 128,
                "conv_length": 4, "x": x_np.view(np.uint16).ravel().tolist(),
                "conv_state": [] if step == 0 else native_conv,
                "state": [] if step == 0 else native_state}
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            assert "error" not in response, response
            actual = response["data"]
            np.testing.assert_array_equal(np.asarray(actual["output"], dtype=np.uint16),
                np.asarray(expected).view(np.uint16).ravel(), err_msg=f"GDN output step {step}")
            np.testing.assert_array_equal(np.asarray(actual["conv_state"], dtype=np.uint16),
                np.asarray(conv_state).view(np.uint16).ravel(), err_msg=f"conv state step {step}")
            np.testing.assert_array_equal(np.asarray(actual["state"], dtype=np.float32).view(np.uint32),
                np.asarray(state).astype(np.float32).ravel().view(np.uint32),
                err_msg=f"recurrent state step {step}")
            native_conv, native_state = actual["conv_state"], actual["state"]
    finally:
        process.stdin.close()
        process.wait(timeout=10)
        stderr = process.stderr.read()
        process.stdout.close()
        process.stderr.close()
    assert process.returncode == 0, stderr
    print(json.dumps({"status": "passed", "model": model.name, "layer": 0,
        "steps": 2, "comparison": "exact output and cache bit patterns",
        "performance": "not measured"}))


if __name__ == "__main__":
    main()
