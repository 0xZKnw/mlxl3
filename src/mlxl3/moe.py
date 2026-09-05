"""Grouped EXL3 SwitchGLU for routed mixture-of-experts models."""

from __future__ import annotations

import os
from functools import cache
from typing import Any

import mlx.core as mx
from mlx import nn

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels.qmv import (
    _scaled_hadamard_input,
    _scaled_hadamard_output,
    _scaled_hadamard_output_reduce,
    qmm_exl3_expert_segmented,
    qmv_exl3_expert_mapped,
)
from mlxl3.kernels.routing import inverse_permutation

_USE_FUSED_LFM_ROUTER = os.environ.get("MLXL3_FUSED_MOE_ROUTER", "1") != "0"
_USE_FUSED_ROUTER_TOPK = os.environ.get("MLXL3_FUSED_ROUTER_TOPK", "1") != "0"
_USE_BITONIC_ROUTER = os.environ.get('MLXL3_BITONIC_ROUTER', '0') == '1'
_USE_GEMMA_EXPERT_REDUCE = os.environ.get('MLXL3_GEMMA_EXPERT_REDUCE', '0') == '1'
_USE_GELU_DOWN_PREP = os.environ.get('MLXL3_GELU_DOWN_PREP', '0') == '1'
_USE_PREFILL_GATHER_HADAMARD = os.environ.get('MLXL3_PREFILL_GATHER_HADAMARD', '0') == '1'
_USE_PREFILL_GATHER_SIMD = os.environ.get('MLXL3_PREFILL_GATHER_SIMD', '1') != '0'
_USE_FUSED_MOE_REDUCE = os.environ.get("MLXL3_FUSED_MOE_REDUCE", "1") != "0"
_USE_FUSED_MOE_GLU_PREP = os.environ.get("MLXL3_FUSED_MOE_GLU_PREP", "1") != "0"
_USE_SEGMENTED_MOE_PREFILL = os.environ.get("MLXL3_SEGMENTED_MOE_PREFILL", "1") != "0"
_SEGMENTED_BLOCK_ROWS = 64
_SEGMENTED_MOE_MIN_ROWS = int(os.environ.get("MLXL3_SEGMENTED_MOE_MIN_ROWS", "64"))


def _butterfly_source(source: str, target: str, round_after_four: bool = False) -> str:
    stages: list[str] = []
    for shift in range(7):
        step = 1 << shift
        stages.append(
            f"""
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float {source}_value_{shift} = {source}[tid];
            float {source}_peer_{shift} = {source}[tid ^ {step}u];
            {target}[tid] = (tid & {step}u)
                ? {source}_peer_{shift} - {source}_value_{shift}
                : {source}_value_{shift} + {source}_peer_{shift};
            """
        )
        if round_after_four and shift == 3:
            stages.append(f'{target}[tid] = float(half({target}[tid]));')
        source, target = target, source
    return "".join(stages)


@cache
def _prefill_gather_hadamard_kernel(dims):
    stages = _butterfly_source('a', 'b', round_after_four=True)
    return mx.fast.metal_kernel(
        name=f'mlxl3_prefill_gather_hadamard_d{dims}_v1',
        input_names=['x', 'scales', 'tokens', 'experts'], output_names=['out'],
        source=f"""
            uint tid = thread_position_in_threadgroup.x;
            uint block = threadgroup_position_in_grid.x;
            uint route = threadgroup_position_in_grid.y;
            uint projection = threadgroup_position_in_grid.z;
            uint column = block * 128u + tid;
            uint token = tokens[route];
            uint expert = uint(experts[route]);
            threadgroup float a[128];
            threadgroup float b[128];
            a[tid] = float(half(half(x[token * {dims}u + column]) *
                half(scales[(expert * 2u + projection) * {dims}u + column])));
            {stages}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            out[(projection * uint(tokens_shape[0]) + route) * {dims}u + column] =
                half(float(half(b[tid])) * 0.08838834764831845f);
        """, compile_options={'math_mode': 'safe'},
    )


def _prefill_gather_hadamard(x, scales, tokens, experts):
    dims = x.shape[-1]
    if dims % 128 or scales.shape[1:] != (2, dims) or tokens.shape != experts.shape:
        raise ValueError('invalid routed prefill Hadamard shape')
    return _prefill_gather_hadamard_kernel(dims)(
        inputs=[x, scales, tokens, experts], grid=(dims, tokens.size, 2),
        threadgroup=(128, 1, 1), output_shapes=[(2, tokens.size, dims)],
        output_dtypes=[mx.float16],
    )[0]


@cache
def _prefill_gather_simd_kernel(dims: int):
    # Gather + multiply + MLX's radix-16/radix-8 Hadamard in one SIMD group.
    # No routed activation/scales buffers, threadgroup scratch or barriers.
    # Keep the two FP16 intermediate roundings and final scale unchanged.
    return mx.fast.metal_kernel(
        name=f'mlxl3_prefill_gather_simd_d{dims}_v1',
        input_names=['x', 'scales', 'tokens', 'experts'], output_names=['out'],
        source=f'''
            uint tid = thread_position_in_threadgroup.x;
            uint block = threadgroup_position_in_grid.x;
            uint route = threadgroup_position_in_grid.y;
            uint projection = threadgroup_position_in_grid.z;
            uint token = tokens[route];
            uint expert = uint(experts[route]);
            uint base = block * 128u + tid * 4u;
            float v[4];
            for (uint r=0; r<4; ++r) {{
                uint col = base + r;
                v[r] = float(half(half(x[token * {dims}u + col]) *
                    half(scales[(expert * 2u + projection) * {dims}u + col])));
            }}
            for (uint h=1; h<4; h*=2) {{
                for (uint i=0; i<2; ++i) {{
                    uint k=i & (h-1); uint j=((i-k)<<1)+k;
                    float a=v[j], b=v[j+h]; v[j]=a+b; v[j+h]=a-b;
                }}
            }}
            for (uint h=1; h<4; h*=2) {{
                for (uint r=0; r<4; ++r) {{
                    float peer=simd_shuffle_xor(v[r], h);
                    v[r]=(tid & h) ? peer-v[r] : v[r]+peer;
                }}
            }}
            for (uint r=0; r<4; ++r) v[r]=float(half(v[r]));
            for (uint h=4; h<32; h*=2) {{
                for (uint r=0; r<4; ++r) {{
                    float peer=simd_shuffle_xor(v[r], h);
                    v[r]=(tid & h) ? peer-v[r] : v[r]+peer;
                }}
            }}
            for (uint r=0; r<4; ++r) {{
                out[(projection * uint(tokens_shape[0]) + route) * {dims}u + base + r] =
                    half(float(half(v[r])) * 0.08838834764831845f);
            }}
        ''', compile_options={'math_mode': 'safe'},
    )


def _prefill_gather_simd(x, scales, tokens, experts):
    dims = x.shape[-1]
    if dims % 128 or scales.shape[1:] != (2, dims) or tokens.shape != experts.shape:
        raise ValueError('invalid routed SIMD prefill Hadamard shape')
    return _prefill_gather_simd_kernel(dims)(
        inputs=[x, scales, tokens, experts], grid=(dims // 4, tokens.size, 2),
        threadgroup=(32, 1, 1), output_shapes=[(2, tokens.size, dims)],
        output_dtypes=[mx.float16],
    )[0]


@cache
def _fused_glu_down_prepare_kernel(hidden_dims: int, activation: str = 'silu', logical_hidden: int | None = None):
    """Fuse two output rotations, SwiGLU and the down input rotation."""

    if hidden_dims % 128:
        raise ValueError("fused MoE hidden width must be 128-aligned")
    gate_stages = _butterfly_source("gate_a", "gate_b", activation == 'gelu')
    up_stages = _butterfly_source("up_a", "up_b", activation == 'gelu')
    down_stages = _butterfly_source("down_a", "down_b", activation == 'gelu')
    if activation not in ('silu', 'gelu'):
        raise ValueError('unsupported GLU activation')
    if logical_hidden is not None and not 0 < logical_hidden <= hidden_dims:
        raise ValueError('invalid logical GLU width')
    activate = ('gate / (1.0f + exp(-gate))' if activation == 'silu' else
                '0.5f * gate * (1.0f + tanh(0.7978845608028654f * (gate + 0.044715f * gate * gate * gate)))')
    gate_finish = ('float(half(float(half(gate_b[tid])) * HAD_SCALE))' if activation == 'gelu'
                   else 'gate_b[tid] * HAD_SCALE')
    up_finish = ('float(half(float(half(up_b[tid])) * HAD_SCALE))' if activation == 'gelu'
                 else 'up_b[tid] * HAD_SCALE')
    hidden_finish = f'float(half(({activate}) * up))' if activation == 'silu' else f'float(half(half({activate}) * half(up)))'
    down_finish = 'float(half(down_b[tid]))' if activation == 'gelu' else 'down_b[tid]'
    # Seven stages leave the result in the alternate buffer.
    return mx.fast.metal_kernel(
        name=f"mlxl3_moe_glu_down_prep_h{hidden_dims}_{activation}_l{logical_hidden or hidden_dims}_v2",
        input_names=["ygu", "gu_svh", "down_suh", "selected"],
        output_names=["xhat"],
        source=f"""
            constexpr uint HIDDEN = {hidden_dims}u;
            constexpr float HAD_SCALE = 0.08838834764831845f;
            uint tid = thread_position_in_threadgroup.x;
            uint group = threadgroup_position_in_grid.x;
            uint blocks = HIDDEN / 128u;
            uint slot = group / blocks;
            uint block = group - slot * blocks;
            uint column = block * 128u + tid;
            uint expert = uint(selected[slot]);

            threadgroup float gate_a[128];
            threadgroup float gate_b[128];
            threadgroup float up_a[128];
            threadgroup float up_b[128];
            threadgroup float down_a[128];
            threadgroup float down_b[128];

            gate_a[tid] = float(half(ygu[(slot * 2u) * HIDDEN + column]));
            up_a[tid] = float(half(ygu[(slot * 2u + 1u) * HIDDEN + column]));
            {gate_stages}
            {up_stages}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float gate = float(half(
                {gate_finish}
                * float(gu_svh[(expert * 2u) * HIDDEN + column])
            ));
            float up = float(half(
                {up_finish}
                * float(gu_svh[(expert * 2u + 1u) * HIDDEN + column])
            ));
            float hidden = {hidden_finish};
            if (column >= {logical_hidden or hidden_dims}u) hidden = 0.0f;
            down_a[tid] = float(half(
                hidden * float(down_suh[expert * HIDDEN + column])
            ));
            {down_stages}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            xhat[slot * HIDDEN + column] = half({down_finish} * HAD_SCALE);
        """,
        compile_options={"math_mode": "safe"},
    )


def _fused_glu_down_prepare(
    ygu: mx.array,
    gu_svh: mx.array,
    down_suh: mx.array,
    selected: mx.array,
    activation: str = 'silu',
    logical_hidden: int | None = None,
) -> mx.array:
    slots = int(selected.size)
    hidden_dims = int(ygu.shape[-1])
    return _fused_glu_down_prepare_kernel(hidden_dims, activation, logical_hidden)(
        inputs=[ygu, gu_svh, down_suh, selected],
        grid=(slots * (hidden_dims // 128) * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(slots, hidden_dims)],
        output_dtypes=[mx.float16],
    )[0]


@cache
def _segmented_block_table(experts: int, max_blocks: int):
    """Build a device-side expert/run table without synchronizing routing."""

    @mx.compile
    def build(sorted_experts: mx.array) -> tuple[mx.array, mx.array]:
        counts = mx.zeros((experts,), dtype=mx.uint32).at[sorted_experts].add(1)
        segment_start = mx.cumsum(counts) - counts
        blocks_per_expert = (
            counts + (_SEGMENTED_BLOCK_ROWS - 1)
        ) // _SEGMENTED_BLOCK_ROWS
        block_end = mx.cumsum(blocks_per_expert)
        block_count = block_end[experts - 1 : experts]
        marks = mx.zeros((max_blocks + 1,), dtype=mx.uint32).at[block_end].add(1)
        block_expert = mx.minimum(mx.cumsum(marks)[:max_blocks], experts - 1)
        local_block = mx.arange(max_blocks, dtype=mx.uint32) - (
            block_end - blocks_per_expert
        )[block_expert]
        row_begin = (
            segment_start[block_expert] + local_block * _SEGMENTED_BLOCK_ROWS
        )
        row_count = mx.minimum(
            counts[block_expert] - local_block * _SEGMENTED_BLOCK_ROWS,
            _SEGMENTED_BLOCK_ROWS,
        )
        table = mx.stack([block_expert, row_begin, row_count]).astype(mx.uint32)
        return table, block_count.astype(mx.uint32)

    return build


@cache
def _router_topk_kernel(experts: int, top_k: int, normalize: bool, bitonic: bool = False):
    """Single-row stable top-k adapted from mlx-swift-lm's Qwen router."""

    if experts > 1024:
        raise ValueError("fused router supports at most 1024 experts")
    if not 0 < top_k <= experts:
        raise ValueError("invalid router top-k")
    if bitonic and experts & (experts - 1):
        raise ValueError('bitonic router requires a power-of-two expert count')
    ranking = """
            int above = 0;
            for (uint other = 0u; other < EXPERTS; ++other) {
                above += sort_keys[other] > key ? 1 : 0;
            }
            if (above < int(TOP_K)) {
                uint slot = TOP_K - 1u - uint(above);
                top_values[slot] = float(values[row * EXPERTS + tid]);
                indices[row * TOP_K + slot] = tid;
            }
    """
    if bitonic:
        ranking = """
            for (uint width = 2u; width <= EXPERTS; width <<= 1u) {
                for (uint step = width >> 1u; step; step >>= 1u) {
                    ulong own = sort_keys[tid];
                    ulong peer = sort_keys[tid ^ step];
                    bool take_min = ((tid & width) == 0u) == ((tid & step) == 0u);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    sort_keys[tid] = take_min ? min(own, peer) : max(own, peer);
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            if (tid < TOP_K) {
                uint expert = uint(sort_keys[EXPERTS - TOP_K + tid]);
                indices[row * TOP_K + tid] = expert;
                top_values[tid] = float(values[row * EXPERTS + expert]);
            }
        """
    return mx.fast.metal_kernel(
        name=f"mlxl3_router_topk_e{experts}_k{top_k}_n{int(normalize)}_b{int(bitonic)}_v2",
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

            {ranking}
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
    indices, scores = _router_topk_kernel(experts, top_k, normalize, _USE_BITONIC_ROUTER and experts & (experts - 1) == 0)(
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


def _fused_gemma_experts_call(module, x, top_k_indices, top_k_weights):
    if x.size != x.shape[-1]:
        return module._mlxl3_original_call(x, top_k_indices, top_k_weights)
    return module.switch_glu(x, top_k_indices, scores=top_k_weights)


def fuse_moe_routers(model: nn.Module) -> int:
    """Replace supported decode router chains with fused Metal selectors."""

    fused = 0
    for _, module in model.named_modules():
        if (_USE_GEMMA_EXPERT_REDUCE
                and type(module).__module__ == 'mlx_lm.models.gemma4_text'
                and type(module).__name__ == 'Experts'
                and isinstance(module.switch_glu, EXL3SwitchGLU)):
            module._mlxl3_original_call = module.__call__
            module.__class__ = type('MLXL3FusedGemmaExperts', (type(module),),
                                    {'__call__': _fused_gemma_experts_call})
            fused += 1
            continue
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
        activation: str = 'silu',
        logical_hidden_dims: int | None = None,
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
        if activation not in ('silu', 'gelu'):
            raise ValueError('unsupported EXL3 expert activation')
        self.activation = activation
        self.logical_hidden_dims = logical_hidden_dims

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

    def _segmented_prefill(
        self,
        x: mx.array,
        indices: mx.array,
        scores: mx.array | None,
    ) -> mx.array:
        """Sort routes and reuse decoded expert tiles across token blocks."""

        rows = x.size // self.input_dims
        top_k = indices.shape[-1]
        slots = rows * top_k
        selected = indices.reshape(-1).astype(mx.int32)
        order = mx.argsort(selected)
        inverse = inverse_permutation(order)
        sorted_experts = selected[order]
        token_rows = (mx.arange(slots, dtype=mx.uint32) // top_k)[order]

        padded_tokens = mx.concatenate(
            [token_rows, mx.zeros((_SEGMENTED_BLOCK_ROWS,), dtype=mx.uint32)]
        )
        padded_experts = mx.concatenate(
            [
                sorted_experts,
                mx.zeros((_SEGMENTED_BLOCK_ROWS,), dtype=mx.int32),
            ]
        )
        max_blocks = slots // _SEGMENTED_BLOCK_ROWS + self.num_experts + 1
        block_table, block_count = _segmented_block_table(
            self.num_experts, max_blocks
        )(sorted_experts)

        # Promote on SwiGLU only: the full post-tool validation used Qwen.
        # GeGLU/Gemma retains its existing path until model-level validation.
        use_simd_gather = _USE_PREFILL_GATHER_SIMD and self.activation == 'silu'
        if use_simd_gather or _USE_PREFILL_GATHER_HADAMARD:
            prepare = _prefill_gather_simd if use_simd_gather else _prefill_gather_hadamard
            prepared = prepare(
                x.reshape(rows, self.input_dims), self.gu_suh, padded_tokens, padded_experts)
            gate_x, up_x = prepared[0], prepared[1]
        else:
            routed_x = x.reshape(rows, self.input_dims)[padded_tokens]
            gate_x = _scaled_hadamard_input(routed_x, self.gu_suh[padded_experts, 0])
            up_x = _scaled_hadamard_input(routed_x, self.gu_suh[padded_experts, 1])
        gate_raw = qmm_exl3_expert_segmented(
            gate_x,
            self.gu_trellis,
            block_table,
            block_count,
            rows=slots,
            tiles_per_expert=self._hidden_tiles,
            output_dims=self.hidden_dims,
            expert_tile_base=0,
            k=self.bits,
            mode=self.mode,
        )
        up_raw = qmm_exl3_expert_segmented(
            up_x,
            self.gu_trellis,
            block_table,
            block_count,
            rows=slots,
            tiles_per_expert=self._hidden_tiles,
            output_dims=self.hidden_dims,
            expert_tile_base=self.num_experts * self._hidden_tiles,
            k=self.bits,
            mode=self.mode,
        )

        use_fused_glu = self._use_fused_prepare()
        if use_fused_glu:
            gate_up = mx.stack([gate_raw, up_raw], axis=1).reshape(
                slots * 2, self.hidden_dims
            )
            down_x = _fused_glu_down_prepare(
                gate_up,
                self.gu_svh,
                self.down_suh,
                sorted_experts,
                self.activation,
                self.logical_hidden_dims,
            )
        else:
            gate = _scaled_hadamard_output(
                gate_raw,
                self.gu_svh[sorted_experts, 0],
            )
            up = _scaled_hadamard_output(
                up_raw,
                self.gu_svh[sorted_experts, 1],
            )
            hidden = (nn.gelu_approx(gate) if self.activation == 'gelu' else nn.silu(gate)) * up
            if self.logical_hidden_dims is not None:
                hidden = mx.where(mx.arange(self.hidden_dims) < self.logical_hidden_dims, hidden, 0)
            down_x = _scaled_hadamard_input(
                hidden,
                self.down_suh[sorted_experts],
            )

        down_x = mx.concatenate(
            [
                down_x,
                mx.zeros(
                    (_SEGMENTED_BLOCK_ROWS, self.hidden_dims),
                    dtype=down_x.dtype,
                ),
            ]
        )
        down_raw = qmm_exl3_expert_segmented(
            down_x,
            self.down_trellis,
            block_table,
            block_count,
            rows=slots,
            tiles_per_expert=self._output_tiles,
            output_dims=self.input_dims,
            expert_tile_base=0,
            k=self.bits,
            mode=self.mode,
        )
        output = _scaled_hadamard_output(
            down_raw,
            self.down_svh[sorted_experts],
        )[inverse]
        output = output.reshape(rows, top_k, self.input_dims)
        if scores is not None:
            output = (output * scores.reshape(rows, top_k, 1)).sum(axis=1)
            return output.reshape(*indices.shape[:-1], self.input_dims)
        return output.reshape(*indices.shape, self.input_dims)

    def _use_fused_prepare(self):
        return self.hidden_dims % 128 == 0 and (
            (_USE_FUSED_MOE_GLU_PREP and self.activation == 'silu' and self.logical_hidden_dims is None)
            or (_USE_GELU_DOWN_PREP and self.activation == 'gelu')
        )

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
        if (
            _USE_SEGMENTED_MOE_PREFILL
            and rows >= _SEGMENTED_MOE_MIN_ROWS
            and self.input_dims % 64 == 0
            and self.hidden_dims % 64 == 0
            and self.bits != 7
        ):
            return self._segmented_prefill(x, indices, scores)
        slots = rows * top_k
        selected = indices.reshape(-1).astype(mx.int32)
        x_rows = x.reshape(rows, self.input_dims)
        x_gu = mx.broadcast_to(
            x_rows[:, None, None, :],
            (rows, top_k, 2, self.input_dims),
        ).reshape(slots * 2, self.input_dims)

        use_fused_glu = self._use_fused_prepare()
        gu = qmv_exl3_expert_mapped(
            x_gu,
            self.gu_trellis,
            self.gu_suh[selected].reshape(slots * 2, self.input_dims),
            None if use_fused_glu else self.gu_svh[selected].reshape(
                slots * 2, self.hidden_dims
            ),
            selected,
            output_dims=self.hidden_dims,
            projections_per_route=2,
            projection_stride_tiles=self.num_experts * self._hidden_tiles,
            k=self.bits,
            mode=self.mode,
            return_raw=use_fused_glu,
        )
        if use_fused_glu:
            hidden = _fused_glu_down_prepare(
                gu.reshape(slots * 2, self.hidden_dims),
                self.gu_svh,
                self.down_suh,
                selected,
                self.activation,
                self.logical_hidden_dims,
            )
        else:
            gu = gu.reshape(slots, 2, self.hidden_dims)
            hidden = (nn.gelu_approx(gu[:, 0]) if self.activation == 'gelu' else nn.silu(gu[:, 0])) * gu[:, 1]
            if self.logical_hidden_dims is not None:
                hidden = mx.where(mx.arange(self.hidden_dims) < self.logical_hidden_dims, hidden, 0)

        output = qmv_exl3_expert_mapped(
            hidden,
            self.down_trellis,
            None if use_fused_glu else self.down_suh[selected],
            None if use_fused_glu else self.down_svh[selected],
            selected,
            output_dims=self.input_dims,
            projections_per_route=1,
            projection_stride_tiles=0,
            reduce_weights=(
                scores.reshape(rows, top_k)
                if scores is not None and not use_fused_glu
                else None
            ),
            k=self.bits,
            mode=self.mode,
            input_pretransformed=use_fused_glu,
            return_raw=use_fused_glu,
        )
        if use_fused_glu:
            scales = self.down_svh[selected]
            if scores is not None:
                output = _scaled_hadamard_output_reduce(
                    output,
                    scales,
                    scores.reshape(rows, top_k),
                ).astype(mx.result_type(x, scores))
            else:
                output = _scaled_hadamard_output(output, scales).astype(x.dtype)
        if scores is not None:
            return output.reshape(*indices.shape[:-1], self.input_dims)
        return output.reshape(*indices.shape, self.input_dims)
