"""Grouped EXL3 SwitchGLU for routed mixture-of-experts models."""

from __future__ import annotations

import os
from functools import cache
from typing import Any

import mlx.core as mx
from mlx import nn

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels.qmv import qmv_exl3_expert_mapped

_USE_FUSED_LFM_ROUTER = os.environ.get("MLXL3_FUSED_MOE_ROUTER", "1") != "0"
_USE_FUSED_ROUTER_TOPK = os.environ.get("MLXL3_FUSED_ROUTER_TOPK", "1") != "0"
_USE_FUSED_MOE_REDUCE = os.environ.get("MLXL3_FUSED_MOE_REDUCE", "1") != "0"


@cache
def _router_topk_kernel(experts: int, top_k: int, normalize: bool):
    """Single-row stable top-k adapted from mlx-swift-lm's Qwen router."""

    if experts > 1024:
        raise ValueError("fused router supports at most 1024 experts")
    if not 0 < top_k <= experts:
        raise ValueError("invalid router top-k")
    return mx.fast.metal_kernel(
        name=f"mlxl3_router_topk_e{experts}_k{top_k}_n{int(normalize)}_v1",
        input_names=["selection", "values"],
        output_names=["indices", "scores"],
        source=f"""
            constexpr uint EXPERTS = {experts}u;
            constexpr uint TOP_K = {top_k}u;
            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;

            threadgroup ulong sort_keys[EXPERTS];
            threadgroup float top_values[TOP_K];

            float value = float(selection[row * EXPERTS + tid]);
            uint bits = value == 0.0f ? 0u : as_type<uint>(value);
            uint monotonic = isnan(value)
                ? 0xffffffffu
                : bits ^ uint((int(bits) >> 31) | int(0x80000000u));
            ulong key = (ulong(monotonic) << 32) | ulong(tid);
            sort_keys[tid] = key;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int above = 0;
            for (uint other = 0u; other < EXPERTS; ++other) {{
                above += sort_keys[other] > key ? 1 : 0;
            }}
            if (above < int(TOP_K)) {{
                uint slot = TOP_K - 1u - uint(above);
                top_values[slot] = float(values[row * EXPERTS + tid]);
                indices[row * TOP_K + slot] = tid;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0u) {{
                T sum = T(0);
                for (uint slot = 0u; slot < TOP_K; ++slot) {{
                    sum = T(top_values[slot]) + sum;
                }}
                for (uint slot = 0u; slot < TOP_K; ++slot) {{
                    T score = T(top_values[slot]);
                    scores[row * TOP_K + slot] = {'score / sum' if normalize else 'score'};
                }}
            }}
        """,
        compile_options={"math_mode": "safe"},
    )


def router_topk(
    values: mx.array,
    top_k: int,
    *,
    normalize: bool,
) -> tuple[mx.array, mx.array]:
    """Select and gather one router row without MLX's full GPU sort."""

    experts = int(values.shape[-1])
    if not (
        _USE_FUSED_ROUTER_TOPK
        and values.size == experts
        and experts <= 1024
        and values.dtype in (mx.float16, mx.float32, mx.bfloat16)
    ):
        indices = mx.argpartition(values, kth=-top_k, axis=-1)[..., -top_k:]
        scores = mx.take_along_axis(values, indices, axis=-1)
        if normalize:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return indices, scores

    output_shape = (*values.shape[:-1], top_k)
    indices, scores = _router_topk_kernel(experts, top_k, normalize)(
        inputs=[values, values],
        template=[("T", values.dtype)],
        grid=(experts, 1, 1),
        threadgroup=(experts, 1, 1),
        output_shapes=[output_shape, output_shape],
        output_dtypes=[mx.uint32, values.dtype],
    )
    return indices, scores


@cache
def _topk_biased_kernel(experts: int, top_k: int):
    if experts > 32:
        raise ValueError("single-SIMD router supports at most 32 experts")
    if not 0 < top_k <= experts:
        raise ValueError("invalid router top-k")
    return mx.fast.metal_kernel(
        name=f"mlxl3_topk_biased_e{experts}_k{top_k}_v1",
        input_names=["probabilities", "bias"],
        output_names=["indices", "scores"],
        source=f"""
            constexpr uint EXPERTS = {experts}u;
            constexpr uint TOP_K = {top_k}u;
            uint row = threadgroup_position_in_grid.x;
            uint lane = thread_index_in_simdgroup;
            uint offset = row * EXPERTS;

            T probability = lane < EXPERTS ? probabilities[offset + lane] : T(0);
            float candidate = lane < EXPERTS
                ? float(probability) + float(bias[lane])
                : -INFINITY;
            uint chosen[TOP_K];
            for (uint rank = 0u; rank < TOP_K; ++rank) {{
                float best = simd_max(candidate);
                uint possible = candidate == best ? lane : 0xffffffffu;
                uint winner = simd_min(possible);
                if (lane == 0u) {{
                    chosen[rank] = winner;
                }}
                if (lane == winner) {{
                    candidate = -INFINITY;
                }}
            }}
            if (lane == 0u) {{
                // MLX argpartition returns its selected tail in ascending
                // value order for this small contiguous row.
                for (uint slot = 0u; slot < TOP_K; ++slot) {{
                    uint expert = chosen[TOP_K - 1u - slot];
                    indices[row * TOP_K + slot] = expert;
                    scores[row * TOP_K + slot] = probabilities[offset + expert];
                }}
            }}
        """,
        compile_options={"math_mode": "safe"},
    )


def topk_biased(
    probabilities: mx.array,
    bias: mx.array,
    top_k: int,
) -> tuple[mx.array, mx.array]:
    """Select biased experts while returning their original probabilities."""

    experts = int(probabilities.shape[-1])
    if bias.shape != (experts,):
        raise ValueError(f"expected expert bias {(experts,)}, got {bias.shape}")
    rows = probabilities.size // experts
    prefix = probabilities.shape[:-1]
    indices, scores = _topk_biased_kernel(experts, top_k)(
        inputs=[probabilities, bias],
        template=[("T", probabilities.dtype)],
        grid=(rows * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(*prefix, top_k), (*prefix, top_k)],
        output_dtypes=[mx.uint32, probabilities.dtype],
    )
    return indices, scores


def _fused_lfm_moe_call(module: Any, x: mx.array) -> mx.array:
    if not _USE_FUSED_LFM_ROUTER or x.size != x.shape[-1]:
        return module._mlxl3_original_call(x)

    gate_logits = module.gate(x)
    routing_weights = mx.sigmoid(gate_logits)
    indices, scores = topk_biased(routing_weights, module.expert_bias, module.top_k)
    if module.norm_topk_prob:
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-6)
    scores = (scores * module.routed_scaling_factor).astype(x.dtype)
    if _USE_FUSED_MOE_REDUCE:
        return module.switch_mlp(x, indices, scores=scores)
    output = module.switch_mlp(x, indices)
    return (output * scores[..., None]).sum(axis=-2)


def _fused_qwen_moe_call(module: Any, x: mx.array) -> mx.array:
    if (
        not _USE_FUSED_ROUTER_TOPK
        or x.size != x.shape[-1]
        or module.sharding_group is not None
    ):
        return module._mlxl3_original_call(x)

    gates = mx.softmax(module.gate(x), axis=-1, precise=True)
    indices, scores = router_topk(
        gates,
        module.top_k,
        normalize=module.norm_topk_prob,
    )
    if _USE_FUSED_MOE_REDUCE:
        routed = module.switch_mlp(x, indices, scores=scores)
    else:
        routed = module.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2)
    shared = module.shared_expert(x)
    shared = mx.sigmoid(module.shared_expert_gate(x)) * shared
    return routed + shared


def fuse_moe_routers(model: nn.Module) -> int:
    """Replace supported decode router chains with fused Metal selectors."""

    fused = 0
    for _, module in model.named_modules():
        is_lfm = (
            hasattr(module, "switch_mlp")
            and isinstance(module.switch_mlp, EXL3SwitchGLU)
            and getattr(module, "num_experts", 0) <= 32
            and hasattr(module, "expert_bias")
        )
        is_qwen = (
            hasattr(module, "switch_mlp")
            and isinstance(module.switch_mlp, EXL3SwitchGLU)
            and 0 < getattr(module, "num_experts", 0) <= 1024
            and not hasattr(module, "expert_bias")
            and hasattr(module, "shared_expert")
            and hasattr(module, "shared_expert_gate")
        )
        if not is_lfm and not is_qwen:
            continue
        module._mlxl3_original_call = module.__call__
        module.__class__ = type(
            f"MLXL3Fused{module.__class__.__name__}",
            (module.__class__,),
            {"__call__": _fused_lfm_moe_call if is_lfm else _fused_qwen_moe_call},
        )
        fused += 1
    return fused


class EXL3SwitchGLU(nn.Module):
    """Evaluate every selected expert with two mapped Metal QMV launches.

    Gate and up matrices share the first launch.  The activated intermediate
    rows feed a second mapped launch for down projection.  The interface is
    intentionally identical to MLX-LM's ``SwitchGLU`` so Bailing's router and
    weighted reduction remain upstream code.
    """

    def __init__(
        self,
        *,
        gu_trellis: mx.array,
        gu_suh: mx.array,
        gu_svh: mx.array,
        down_trellis: mx.array,
        down_suh: mx.array,
        down_svh: mx.array,
        bits: int,
        mode: CodebookMode | int,
    ):
        super().__init__()
        self.gu_trellis = gu_trellis.view(mx.uint16) if gu_trellis.dtype == mx.int16 else gu_trellis
        self.gu_suh = gu_suh
        self.gu_svh = gu_svh
        self.down_trellis = (
            down_trellis.view(mx.uint16) if down_trellis.dtype == mx.int16 else down_trellis
        )
        self.down_suh = down_suh
        self.down_svh = down_svh
        self.bits = int(bits)
        self.mode = CodebookMode(mode)

        if gu_suh.ndim != 3 or gu_suh.shape[1] != 2:
            raise ValueError(f"gu_suh must have shape (experts, 2, input), got {gu_suh.shape}")
        if gu_svh.ndim != 3 or gu_svh.shape[:2] != gu_suh.shape[:2]:
            raise ValueError(f"gu_svh must have shape (experts, 2, hidden), got {gu_svh.shape}")
        if down_suh.ndim != 2 or down_svh.ndim != 2:
            raise ValueError("down scales must be rank-2 expert matrices")
        if down_suh.shape[0] != gu_suh.shape[0] or down_svh.shape[0] != gu_suh.shape[0]:
            raise ValueError("gate/up/down expert counts differ")

        self.num_experts = int(gu_suh.shape[0])
        self.input_dims = int(gu_suh.shape[2])
        self.hidden_dims = int(gu_svh.shape[2])
        self._hidden_tiles = self.hidden_dims // 16
        self._output_tiles = self.input_dims // 16

    def __call__(
        self,
        x: mx.array,
        indices: mx.array,
        scores: mx.array | None = None,
    ) -> mx.array:
        if x.shape[:-1] != indices.shape[:-1]:
            raise ValueError(f"input/routing prefixes differ: {x.shape} and {indices.shape}")
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"expected hidden width {self.input_dims}, got {x.shape[-1]}")

        rows = x.size // self.input_dims
        top_k = indices.shape[-1]
        slots = rows * top_k
        selected = indices.reshape(-1).astype(mx.int32)
        x_rows = x.reshape(rows, self.input_dims)
        x_gu = mx.broadcast_to(
            x_rows[:, None, None, :],
            (rows, top_k, 2, self.input_dims),
        ).reshape(slots * 2, self.input_dims)

        gu = qmv_exl3_expert_mapped(
            x_gu,
            self.gu_trellis,
            self.gu_suh[selected].reshape(slots * 2, self.input_dims),
            self.gu_svh[selected].reshape(slots * 2, self.hidden_dims),
            selected,
            output_dims=self.hidden_dims,
            projections_per_route=2,
            projection_stride_tiles=self.num_experts * self._hidden_tiles,
            k=self.bits,
            mode=self.mode,
        ).reshape(slots, 2, self.hidden_dims)
        hidden = nn.silu(gu[:, 0]) * gu[:, 1]

        output = qmv_exl3_expert_mapped(
            hidden,
            self.down_trellis,
            self.down_suh[selected],
            self.down_svh[selected],
            selected,
            output_dims=self.input_dims,
            projections_per_route=1,
            projection_stride_tiles=0,
            reduce_weights=scores.reshape(rows, top_k) if scores is not None else None,
            k=self.bits,
            mode=self.mode,
        )
        if scores is not None:
            return output.reshape(*indices.shape[:-1], self.input_dims)
        return output.reshape(*indices.shape, self.input_dims)
