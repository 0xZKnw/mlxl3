from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx_lm.models.cache import ArraysCache

from mlxl3.recurrent import compile_recurrent_layers


class FakeQwenRecurrentLayer(nn.Module):
    is_linear = True

    def __call__(self, x, mask=None, cache=None):
        update = x.sum(axis=-2, keepdims=True)
        first = update if cache[0] is None else cache[0] + update
        second = update * 0.5 if cache[1] is None else cache[1] - update * 0.25
        cache[0] = first
        cache[1] = second
        return x + first + second


FakeQwenRecurrentLayer.__module__ = "mlx_lm.models.qwen3_5"


def test_compiled_recurrent_layer_preserves_outputs_and_state() -> None:
    expected_layer = FakeQwenRecurrentLayer()
    actual_layer = FakeQwenRecurrentLayer()
    expected_cache = ArraysCache(2)
    actual_cache = ArraysCache(2)
    assert compile_recurrent_layers(actual_layer) == 1

    prefill = mx.arange(8, dtype=mx.float16).reshape(1, 2, 4) * 0.01
    expected = expected_layer(prefill, cache=expected_cache)
    actual = actual_layer(prefill, cache=actual_cache)
    mx.eval(expected, actual, expected_cache.state, actual_cache.state)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    token = mx.full((1, 1, 4), 0.125, dtype=mx.float16)
    for _ in range(4):
        expected = expected_layer(token, cache=expected_cache)
        actual = actual_layer(token, cache=actual_cache)
        mx.eval(expected, actual, expected_cache.state, actual_cache.state)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        for expected_state, actual_state in zip(
            expected_cache.cache, actual_cache.cache, strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(actual_state), np.asarray(expected_state)
            )
