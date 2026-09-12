"""Save production Python Qwen activations after every decoder layer."""
from pathlib import Path
import sys

import mlx.core as mx
from mlx import nn
import numpy as np

from mlxl3.checkpoint import load_exl3_model
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.gated_delta import gated_delta_update


model, _, _ = load_exl3_model(Path(sys.argv[1]), lazy=False)
text = model.language_model.model
cache = model.make_cache()
hidden = text.embed_tokens(mx.array([[1]], dtype=mx.uint32))
fa_mask = create_attention_mask(hidden, cache[text.fa_idx])
ssm_mask = create_ssm_mask(hidden, cache[text.ssm_idx])
trace = {"embedding": np.asarray(hidden).view(np.uint16)}
trace_layer = int(sys.argv[3]) if len(sys.argv) > 3 else -1
for index, (layer, state) in enumerate(zip(text.layers, cache)):
    if index == trace_layer:
        assert layer.is_linear
        normalized = layer.input_layernorm(hidden)
        gdn = layer.linear_attn
        qkv = gdn.in_proj_qkv(normalized)
        z = gdn.in_proj_z(normalized).reshape(1, 1, gdn.num_v_heads, gdn.head_v_dim)
        b = gdn.in_proj_b(normalized)
        a = gdn.in_proj_a(normalized)
        conv_state = mx.zeros(
            (1, gdn.conv_kernel_size - 1, gdn.conv_dim), dtype=normalized.dtype
        )
        conv = nn.silu(gdn.conv1d(mx.concatenate([conv_state, qkv], axis=1)))
        q, key, value = [
            value.reshape(1, 1, heads, dims)
            for value, heads, dims in zip(
                mx.split(conv, [gdn.key_dim, 2 * gdn.key_dim], -1),
                (gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads),
                (gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim),
            )
        ]
        scale = key.shape[-1] ** -0.5
        q = (scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        key = scale * mx.fast.rms_norm(key, None, 1e-6)
        beta = mx.sigmoid(b)
        softplus = nn.softplus(a + gdn.dt_bias)
        decay = mx.exp(-mx.exp(gdn.A_log.astype(mx.float32)) * softplus)
        out, _ = gated_delta_update(
            q, key, value, a, b, gdn.A_log, gdn.dt_bias, None,
            ssm_mask, use_kernel=not gdn.training,
        )
        gdn_norm = mx.fast.rms_norm(out, gdn.norm.weight, gdn.norm.eps)
        gated = gdn.norm(out, z)
        attention = gdn.out_proj(gated.reshape(1, 1, -1))
        residual = hidden + attention
        post = layer.post_attention_layernorm(residual)
        mlp = layer.mlp(post)
        hidden = residual + mlp
        for name, value in zip(
            ("input_norm", "qkv", "z", "a", "b", "conv", "q", "k", "v",
             "beta", "softplus", "g", "gdn_out", "gdn_norm", "gated", "attention",
             "residual", "post_norm", "mlp", "output"),
            (normalized, qkv, z, a, b, conv, q, key, value, beta, softplus, decay,
             out, gdn_norm, gated, attention, residual, post, mlp, hidden),
        ):
            trace[f"trace_{name}"] = np.asarray(value.astype(mx.float16)).view(np.uint16)
    else:
        hidden = layer(hidden, mask=ssm_mask if layer.is_linear else fa_mask, cache=state)
    trace[f"layer_{index}"] = np.asarray(hidden).view(np.uint16)
normalized = text.norm(hidden)
trace["norm"] = np.asarray(normalized).view(np.uint16)
trace["logits"] = np.asarray(model.language_model.lm_head(normalized)).view(np.uint16)
np.savez(sys.argv[2], **trace)
print({"status": "written", "layers": len(text.layers)})
