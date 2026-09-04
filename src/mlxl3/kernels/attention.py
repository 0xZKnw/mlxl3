"""Experimental full-context d=512 decode with GQA-aware matrix kernels.

No KV quantization, sparse selection, or speculative generation. Score and
probability precision match the fallback, but matrix reduction order can differ.
The slower online-softmax prototype was discarded, not routed into the model.
"""

from functools import cache, partial

import mlx.core as mx


@cache
def _grouped_decode_graph(batch, heads, kv_heads):
    # Only KV length changes during decode. A shape-specialized mx.compile
    # would retrace this function for every generated token.
    @partial(mx.compile, shapeless=True)
    def run(q, k, v, scale):
        qs = (q * scale).reshape(batch, kv_heads, heads // kv_heads, 512)
        scores = qs @ k.swapaxes(-1, -2)
        probs = mx.softmax(scores, axis=-1, precise=True)
        return (probs @ v).reshape(batch, heads, 1, 512)

    return run


def grouped_sdpa_decode_512(q, k, v, scale=1.0):
    """Expose GQA heads as matrix rows, not independent GEMV batches.

    Preserves the fallback's FP16 score/probability materialization. Unlike the
    tiled experimental kernel this does not use block-local softmax reductions.
    Matrix kernel reduction order can still differ from the original GEMVs.
    """
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.shape != k.shape
        or q.shape[0] != k.shape[0]
        or k.shape[1] == 0
        or q.shape[1] % k.shape[1]
        or q.shape[2:] != (1, 512)
        or k.shape[-1] != 512
        or k.shape[2] < 1
        or any(x.dtype != mx.float16 for x in (q, k, v))
    ):
        raise ValueError("grouped sdpa512 requires FP16 d=512 GQA decode")
    return _grouped_decode_graph(q.shape[0], q.shape[1], k.shape[1])(
        q, k, v, mx.array(scale, dtype=q.dtype)
    )


@cache
def _gqa_qk_kernel():
    return mx.fast.metal_kernel(
        name="mlxl3_gqa8_qk512_v1",
        input_names=["q", "k"],
        output_names=["scores"],
        ensure_row_contiguous=False,
        header="#include <metal_simdgroup_matrix>\n",
        compile_options={"math_mode": "safe"},
        source=r"""
            uint tid=thread_position_in_threadgroup.x;
            uint lane=thread_index_in_simdgroup;
            uint sg=simdgroup_index_in_threadgroup;
            uint block=threadgroup_position_in_grid.x;
            uint bk=threadgroup_position_in_grid.y;
            uint kvh=bk % uint(k_shape[1]);
            uint batch=bk / uint(k_shape[1]);
            uint length=uint(k_shape[2]);
            uint start=block*32u+sg*8u;
            size_t kb=batch*k_strides[0]+kvh*k_strides[1];
            size_t qb=batch*q_strides[0]+kvh*8u*q_strides[1];
            if (q_strides[3] != 1 || k_strides[3] != 1) {
                // Rare strided feature views: correct scalar fallback within
                // this kernel, no full KV copy to repair strides each token.
                for (uint i=lane;i<64u;i+=32u) {
                    uint h=i/8u,t=start+i%8u;
                    if (t<length) {
                        float sum=0;
                        for (uint d=0;d<512u;++d)
                            sum=fma(float(q[qb+h*q_strides[1]+d*q_strides[3]]),float(k[kb+t*k_strides[2]+d*k_strides[3]]),sum);
                        scores[(bk*8u+h)*length+t]=half(sum);
                    }
                }
                return;
            }
            threadgroup half tail[8*512];
            threadgroup float dots[4*64];
            bool last_block=block==length/32u;
            if (last_block && length%8u) {
                uint tail_start=length/8u*8u;
                for (uint i=tid;i<8u*512u;i+=128u) {
                    uint row=i/512u, d=i%512u;
                    tail[i]=tail_start+row<length ? k[kb+(tail_start+row)*k_strides[2]+d] : half(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.0f);
            if (start<length) {
                for (uint d=0;d<512u;d+=8u) {
                    simdgroup_half8x8 qm,km;
                    simdgroup_load(qm,q+qb+d,q_strides[1]);
                    if (start+8u<=length)
                        simdgroup_load(km,k+kb+start*k_strides[2]+d,k_strides[2],ulong2(0),true);
                    else
                        simdgroup_load(km,tail+d,512,ulong2(0),true);
                    simdgroup_multiply_accumulate(acc,qm,km,acc);
                }
                simdgroup_store(acc,dots+sg*64u,8);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (start<length) {
                for (uint i=lane;i<64u;i+=32u) {
                    uint h=i/8u,t=start+i%8u;
                    if (t<length) scores[(bk*8u+h)*length+t]=half(dots[sg*64u+i]);
                }
            }
        """,
    )


def matrix_sdpa_decode_512(q, k, v, *, scale=1.0):
    """Custom Metal QK for eight query heads sharing each KV head."""
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.shape != k.shape
        or q.shape[0] != k.shape[0]
        or q.shape[1] != k.shape[1] * 8
        or q.shape[2:] != (1, 512)
        or k.shape[-1] != 512
        or k.shape[2] < 1
        or any(x.dtype != mx.float16 for x in (q, k, v))
    ):
        raise ValueError("matrix sdpa512 requires FP16 d=512 GQA=8 decode")
    # Keep strided/reserved KV storage; the kernel reads its explicit strides.
    qs = q * mx.array(scale, dtype=mx.float16)
    batch, heads, _, dim = q.shape
    length = k.shape[2]
    kv_heads = k.shape[1]
    scores = _gqa_qk_kernel()(
        inputs=[qs, k],
        grid=(((length + 31) // 32) * 128, batch * kv_heads, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(batch, kv_heads, 8, length)],
        output_dtypes=[mx.float16],
    )[0]
    probs = mx.softmax(scores, axis=-1, precise=True)
    return (probs @ v).reshape(batch, heads, 1, dim)
