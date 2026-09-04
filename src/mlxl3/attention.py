"""Model-scoped, opt-in d=512 Gemma decode dispatch.

The upstream attention method is preserved; only its SDPA dependency is
substituted in a private function globals dictionary. No global MLX-LM patch,
and no duplicated RoPE/cache/normalization implementation to drift upstream.
"""

import os
import types
from functools import cache

import mlx.core as mx

from mlxl3.kernels.attention import grouped_sdpa_decode_512, matrix_sdpa_decode_512

_MODE = os.environ.get("MLXL3_GEMMA_SDPA512", "off")
_MIN_LENGTH = 2048


def _dispatch(reference, queries, keys, values, cache, scale, mask, sinks=None):
    if (
        _MODE in ("grouped", "matrix")
        and not hasattr(cache, "bits")
        and mask is None
        and sinks is None
        and queries.ndim == keys.ndim == values.ndim == 4
        and queries.shape[2:] == (1, 512)
        and keys.shape == values.shape
        and keys.shape[-1] == 512
        and queries.shape[0] == keys.shape[0]
        and queries.shape[1] == keys.shape[1] * 8
        and keys.shape[2] >= _MIN_LENGTH
        and all(x.dtype == mx.float16 for x in (queries, keys, values))
    ):
        fn = matrix_sdpa_decode_512 if _MODE == "matrix" else grouped_sdpa_decode_512
        return fn(queries, keys, values, scale=scale)
    return reference(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)


@cache
def _decode_class(base):
    original = base.__call__
    reference = original.__globals__["scaled_dot_product_attention"]
    namespace = dict(original.__globals__)
    namespace["scaled_dot_product_attention"] = lambda *args, **kwargs: _dispatch(
        reference, *args, **kwargs
    )
    routed = types.FunctionType(
        original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__
    )
    routed.__kwdefaults__ = original.__kwdefaults__

    def call(self, *args, **kwargs):
        if self.training:
            return original(self, *args, **kwargs)
        return routed(self, *args, **kwargs)

    return type(f"MLXL3{base.__name__}512", (base,), {"__call__": call})


def install_gemma_attention(model):
    if _MODE not in ("off", "grouped", "matrix"):
        raise ValueError("MLXL3_GEMMA_SDPA512 must be off, grouped, or matrix")
    if _MODE == "off":
        return 0
    count = 0
    for _, module in list(model.named_modules()):
        cls = type(module)
        if (
            cls.__module__ == "mlx_lm.models.gemma4_text"
            and cls.__name__ == "Attention"
            and module.head_dim == 512
        ):
            module.__class__ = _decode_class(cls)
            count += 1
    return count
