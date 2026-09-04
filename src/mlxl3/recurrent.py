"""Lossless graph compilation for fixed-shape recurrent decode layers."""

from __future__ import annotations

import os
from functools import cache
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.cache import ArraysCache

_USE_COMPILED_RECURRENT_LAYERS = (
    os.environ.get("MLXL3_COMPILED_RECURRENT_LAYERS", "1") != "0"
)


def _compiled_recurrent_call(
    module: nn.Module,
    x: mx.array,
    mask: mx.array | None = None,
    cache: Any | None = None,
) -> mx.array:
    """Use the state-explicit graph only for initialized one-token decode."""

    if (
        _USE_COMPILED_RECURRENT_LAYERS
        and isinstance(cache, ArraysCache)
        and x.shape[-2] == 1
        and mask is None
        and all(value is not None for value in cache.cache)
    ):
        output, state = module._mlxl3_compiled_decode(x, cache.state)
        cache.state = state
        return output
    return module._mlxl3_original_call(x, mask=mask, cache=cache)


@cache
def _compiled_recurrent_class(base: type[nn.Module]) -> type[nn.Module]:
    return type(
        f"MLXL3Compiled{base.__name__}",
        (base,),
        {"__call__": _compiled_recurrent_call},
    )


def _compiled_stateless_call(module: nn.Module, x: mx.array) -> mx.array:
    if _USE_COMPILED_RECURRENT_LAYERS and x.shape[-2] == 1:
        return module._mlxl3_compiled_stateless(x)
    return module._mlxl3_stateless_original_call(x)


@cache
def _compiled_stateless_class(base: type[nn.Module]) -> type[nn.Module]:
    return type(
        f"MLXL3Compiled{base.__name__}",
        (base,),
        {"__call__": _compiled_stateless_call},
    )


def _compile_stateless_module(module: nn.Module) -> None:
    original = module.__call__
    module._mlxl3_stateless_original_call = original
    module._mlxl3_compiled_stateless = mx.compile(original)
    module.__class__ = _compiled_stateless_class(type(module))


def compile_recurrent_layers(model: nn.Module) -> int:
    """Compile Qwen recurrent decoder blocks as explicit state transitions."""

    if not _USE_COMPILED_RECURRENT_LAYERS:
        return 0
    compiled = 0
    stateless_ids: set[int] = set()
    for _, module in list(model.named_modules()):
        module_type = type(module)
        if not module_type.__module__.startswith("mlx_lm.models.qwen3_5"):
            continue
        if getattr(module, "is_linear", None) is False:
            mlp = getattr(module, "mlp", None)
            if (
                isinstance(mlp, nn.Module)
                and id(mlp) not in stateless_ids
                and not hasattr(mlp, "_mlxl3_compiled_stateless")
            ):
                _compile_stateless_module(mlp)
                stateless_ids.add(id(mlp))
                compiled += 1
            continue
        if not getattr(module, "is_linear", False) or hasattr(
            module, "_mlxl3_compiled_decode"
        ):
            continue
        original = module.__call__

        def decode(x: mx.array, state: Any, *, call=original):
            local_cache = ArraysCache(len(state[0]))
            local_cache.state = state
            output = call(x, mask=None, cache=local_cache)
            return output, local_cache.state

        module._mlxl3_original_call = original
        module._mlxl3_compiled_decode = mx.compile(decode)
        module.__class__ = _compiled_recurrent_class(module_type)
        compiled += 1
    return compiled
