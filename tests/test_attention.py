import mlx.core as mx
import numpy as np
import pytest

import mlxl3.attention as dispatch
from mlxl3.kernels.attention import grouped_sdpa_decode_512, matrix_sdpa_decode_512


@pytest.mark.parametrize("length", [1, 3, 8, 31, 32, 129, 1024, 4097])
@pytest.mark.parametrize("fn", [grouped_sdpa_decode_512, matrix_sdpa_decode_512])
def test_sdpa512_reference(length, fn):
    mx.random.seed(42)
    q = (mx.random.normal((1, 16, 1, 512)) * 0.1).astype(mx.float16)
    k = mx.random.normal((1, 2, length, 512)).astype(mx.float16)
    v = mx.random.normal(k.shape).astype(mx.float16)
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
    actual = fn(q, k, v)
    mx.eval(actual, expected)
    # This numeric tolerance is NOT a claim of bitwise equality.
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=0.001, rtol=0.005)


@pytest.mark.parametrize("feature_stride", [1, 2])
@pytest.mark.parametrize("fn", [grouped_sdpa_decode_512, matrix_sdpa_decode_512])
def test_strided_cache_and_batch(feature_stride, fn):
    mx.random.seed(3)
    q = (mx.random.normal((2, 1, 16, 512)).transpose(0, 2, 1, 3) * 0.1).astype(mx.float16)
    k = (
        mx.random.normal((2, 1200, 2, 512 * feature_stride))
        .astype(mx.float16)
        .transpose(0, 2, 1, 3)[:, :, :1031, ::feature_stride]
    )
    v = k * 0.25
    expected = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5)
    actual = fn(q, k, v, scale=0.5)
    mx.eval(actual, expected)
    np.testing.assert_allclose(np.array(actual), np.array(expected), atol=0.0005, rtol=0.005)


def test_geometry_validation():
    x = mx.zeros((1, 2, 1, 256), dtype=mx.float16)
    with pytest.raises(ValueError):
        matrix_sdpa_decode_512(x, x, x)


@pytest.mark.parametrize("case", ["off", "short", "prefill", "mask", "sinks", "quantized", "dtype"])
def test_dispatch_falls_back(monkeypatch, case):
    monkeypatch.setattr(dispatch, "_MODE", "off" if case == "off" else "matrix")
    q = mx.zeros(
        (1, 16, 2 if case == "prefill" else 1, 512),
        dtype=mx.float32 if case == "dtype" else mx.float16,
    )
    k = mx.zeros((1, 2, 10 if case == "short" else 2048, 512), dtype=q.dtype)
    cache = type("Quantized", (), {"bits": 4})() if case == "quantized" else None
    kwargs = {
        "cache": cache,
        "scale": 1.0,
        "mask": "causal" if case == "mask" else None,
        "sinks": mx.zeros(16) if case == "sinks" else None,
    }
    sentinel = object()
    assert dispatch._dispatch(lambda *a, **kw: sentinel, q, k, k, **kwargs) is sentinel


def test_install_is_model_scoped_and_off_is_noop(monkeypatch):
    from mlx import nn
    from mlx_lm.models import gemma4_text

    config = gemma4_text.ModelArgs(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        num_hidden_layers=1,
        num_kv_shared_layers=0,
        layer_types=["full_attention"],
    )
    model = nn.Sequential(gemma4_text.Attention(config, 0))
    reference = gemma4_text.scaled_dot_product_attention
    cls = type(model.layers[0])
    monkeypatch.setattr(dispatch, "_MODE", "off")
    assert dispatch.install_gemma_attention(model) == 0
    assert type(model.layers[0]) is cls
    monkeypatch.setattr(dispatch, "_MODE", "grouped")
    assert dispatch.install_gemma_attention(model) == 1
    assert gemma4_text.scaled_dot_product_attention is reference
    assert type(gemma4_text.Attention(config, 0)) is cls
    assert dispatch.install_gemma_attention(model) == 0
