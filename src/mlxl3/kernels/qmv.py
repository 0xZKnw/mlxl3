"""EXL3 Metal QMV/QMM kernels.

Kernel algorithms incorporate adaptations of ExLlamaV3 and PonyExl3. See the
repository's ``THIRD_PARTY_NOTICES.md`` for attribution and license details.
"""

from __future__ import annotations

import math
import os
import platform
from functools import cache

import mlx.core as mx

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels.common import (
    forward_permutation_header,
    permutation_header,
    specialized_codebook_header,
)
from mlxl3.tuning import tensor_tile

_HADAMARD_SCALE = 1.0 / math.sqrt(128.0)
_USE_SCALED_HADAMARD = os.environ.get('MLXL3_SCALED_HADAMARD', '0') == '1'
_USE_QMV_OUTPUT_FUSION = os.environ.get('MLXL3_QMV_OUTPUT_FUSION', '0') == '1'


@cache
def _scaled_hadamard_kernel(input_scale: bool):
    # Match MLX radix-16 then radix-8, including both FP16 roundings.
    return mx.fast.metal_kernel(
        name=f'mlxl3_scaled_hadamard_{int(input_scale)}_simd_v2',
        input_names=['x', 'scales'], output_names=['y'],
        source='''
            uint tid = thread_position_in_threadgroup.x;
            uint base = threadgroup_position_in_grid.x * 128u;
            float v[4];
            for (uint r=0; r<4; ++r) {
                uint index=base+tid*4+r;
                half value=half(x[index % X_SIZE]);
                if (INPUT_SCALE) value=half(value * half(scales[index % SCALE_SIZE]));
                v[r]=float(value);
            }
            for (uint h=1; h<4; h*=2) {
                for (uint i=0; i<2; ++i) {
                    uint k=i & (h-1); uint j=((i-k)<<1)+k;
                    float a=v[j], b=v[j+h]; v[j]=a+b; v[j+h]=a-b;
                }
            }
            for (uint h=1; h<4; h*=2) {
                for (uint r=0; r<4; ++r) {
                    float peer=simd_shuffle_xor(v[r], h);
                    v[r]=(tid & h) ? peer-v[r] : v[r]+peer;
                }
            }
            for (uint r=0; r<4; ++r) v[r]=float(half(v[r]));
            for (uint h=4; h<32; h*=2) {
                for (uint r=0; r<4; ++r) {
                    float peer=simd_shuffle_xor(v[r], h);
                    v[r]=(tid & h) ? peer-v[r] : v[r]+peer;
                }
            }
            for (uint r=0; r<4; ++r) {
                uint index=base+tid*4+r;
                half result=half(float(half(v[r])) * 0.08838834764831845f);
                if (!INPUT_SCALE) result=half(result * half(scales[index % SCALE_SIZE]));
                y[index]=result;
            }
        ''', compile_options={'math_mode':'safe'},
    )


@mx.compile
def _run_scaled_hadamard(x, scale, input_scale):
    shape = mx.broadcast_arrays(x, scale)[0].shape
    size = math.prod(shape)
    return _scaled_hadamard_kernel(input_scale)(
        inputs=[x, scale], template=[('INPUT_SCALE', input_scale), ('SCALE_SIZE', scale.size), ('X_SIZE', x.size)],
        grid=(size//128*32, 1, 1), threadgroup=(32,1,1),
        output_shapes=[shape], output_dtypes=[mx.float16])[0]
_USE_K3_WINDOW_DECODE = os.environ.get("MLXL3_K3_WINDOW_DECODE", "1") != "0"
_USE_TENSOR_QMM = os.environ.get("MLXL3_TENSOR_QMM", "1") != "0"
_USE_TENSOR_SEGMENTED_QMM = (
    os.environ.get("MLXL3_TENSOR_SEGMENTED_QMM", "1") != "0"
)
_SEGMENTED_TENSOR_ROWS = int(os.environ.get('MLXL3_SEGMENTED_TENSOR_ROWS', '32'))
_USE_SEGMENTED_BUCKETS = os.environ.get('MLXL3_SEGMENTED_BUCKETS', '0') == '1'
_USE_SEGMENTED_LOCALITY = os.environ.get('MLXL3_SEGMENTED_LOCALITY', '0') == '1'
if _SEGMENTED_TENSOR_ROWS not in (8, 16, 32):
    raise ValueError('MLXL3_SEGMENTED_TENSOR_ROWS must be 8, 16 or 32')


@cache
def _qmv_inner_kernel(k: int, mode: int, input_dims: int, output_dims: int):
    return mx.fast.metal_kernel(
        name=f"mlxl3_qmv_inner_k{k}_cb{mode}_{input_dims}x{output_dims}",
        input_names=["xhat", "trellis"],
        output_names=["yhat"],
        header=specialized_codebook_header(mode) + permutation_header(),
        source=r"""
            uint tid = thread_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint n = threadgroup_position_in_grid.x;
            threadgroup float partials[4];
            float acc = 0.0f;

            for (uint row = tid; row < uint(INPUT_DIMS); row += 128u) {
                uint tile_k = row >> 4;
                uint tile_n = n >> 4;
                uint local_rm = ((row & 15u) << 4) | (n & 15u);
                uint t = uint(mlxl3_perm_inv[local_rm]);
                uint packed_per_tile = 16u * uint(K);
                uint words_per_tile = 8u * uint(K);
                uint tile_base =
                    (tile_k * uint(TILES_N) + tile_n) * packed_per_tile;

                uint b0 = t * uint(K) + uint(K) - 16u + 256u * uint(K);
                uint b1 = b0 + 16u;
                uint i0 = b0 >> 5;
                uint i1 = (b1 - 1u) >> 5;
                uint shift = ((i1 + 1u) << 5) - b1;
                uint wi0 = i0 % words_per_tile;
                uint wi1 = i1 % words_per_tile;
                uint a = uint(trellis[tile_base + 2u * wi0]) |
                         (uint(trellis[tile_base + 2u * wi0 + 1u]) << 16);
                uint b = uint(trellis[tile_base + 2u * wi1]) |
                         (uint(trellis[tile_base + 2u * wi1 + 1u]) << 16);
                ulong merged = (ulong(a) << 32) | ulong(b);
                uint word = uint((merged >> shift) & 0xfffful);
                float weight = float(mlxl3_decode_codeword(word, CB));
                acc = fma(float(xhat[row]), weight, acc);
            }

            acc = simd_sum(acc);
            if (lane == 0u) {
                partials[simd] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = tid < 4u ? partials[tid] : 0.0f;
            total = simd_sum(total);
            if (tid == 0u) {
                yhat[n] = total;
            }
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _qmv_tile_kernel(
    k: int,
    mode: int,
    input_dims: int,
    output_dims: int,
    tiles_per_group: int = 1,
    simdgroups: int = 4,
    fuse_output: bool = False,
):
    """Configurable-simdgroup CUDA-style kernel over adjacent output tiles."""

    if tiles_per_group not in (1, 2, 4, 8):
        raise ValueError(f"unsupported QMV output tile group {tiles_per_group}")
    if fuse_output and tiles_per_group != 8:
        raise ValueError('fused output needs a whole 128-element Hadamard block')
    if simdgroups not in (2, 4, 8) or simdgroups * tiles_per_group > 64:
        raise ValueError(
            f"unsupported QMV shape: {simdgroups} simdgroups x {tiles_per_group} tiles"
        )

    return mx.fast.metal_kernel(
        name=(
            f"mlxl3_qmv_tile_k{k}_cb{mode}_{input_dims}x{output_dims}"
            f"_nt{tiles_per_group}_sg{simdgroups}_f{int(fuse_output)}_v6"
        ),
        input_names=["xhat", "trellis", "svh"],
        output_names=["yhat"],
        header=(
            f"#define MLXL3_QMV_NT {tiles_per_group}u\n"
            f"#define MLXL3_QMV_SG {simdgroups}u\n"
            f"#define MLXL3_K_BITS {k}u\n"
            f"#define MLXL3_FUSE_OUTPUT {int(fuse_output)}\n"
            + specialized_codebook_header(mode)
        ),
        source=r"""
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint tile_n = threadgroup_position_in_grid.x * MLXL3_QMV_NT;
            uint split = threadgroup_position_in_grid.z;
            threadgroup float partials[MLXL3_QMV_SG][MLXL3_QMV_NT * 16u];
            threadgroup float ha[128];
            threadgroup float hb[128];

            uint word0[2];
            uint word1[2];
            uint shifts[2];
            for (uint group = 0u; group < 2u; ++group) {
                uint first = lane * 8u + group * 4u;
                int end = int(first + 4u) * K + 256 * K;
                int last_word = (end - 1) / 32;
                word1[group] = uint(last_word) % uint(PACKED_U32);
                word0[group] = uint(last_word - 1) % uint(PACKED_U32);
                shifts[group] = uint((last_word + 1) * 32 - end);
            }

            float acc[MLXL3_QMV_NT][8];
            for (uint output_tile = 0u; output_tile < MLXL3_QMV_NT; ++output_tile) {
                for (uint j = 0u; j < 8u; ++j) {
                    acc[output_tile][j] = 0.0f;
                }
            }
            uint tiles_per_split =
                (uint(TILES_K) + uint(N_SPLITS) - 1u) / uint(N_SPLITS);
            uint tile_begin = split * tiles_per_split;
            uint tile_end = min(tile_begin + tiles_per_split, uint(TILES_K));
            for (
                uint tile_k = tile_begin + simd;
                tile_k < tile_end;
                tile_k += MLXL3_QMV_SG
            ) {
                const device half2* x_pairs =
                    reinterpret_cast<const device half2*>(xhat + tile_k * 16u);
                half2 x_pair = x_pairs[lane & 7u];
                half2 x_pair0 = simd_shuffle(x_pair, ushort(lane & 3u));
                half2 x_pair1 = simd_shuffle(x_pair, ushort((lane & 3u) + 4u));
                float x_values[4];
                x_values[0] = float(x_pair0.x);
                x_values[1] = float(x_pair0.y);
                x_values[2] = float(x_pair1.x);
                x_values[3] = float(x_pair1.y);
                for (
                    uint output_tile = 0u;
                    output_tile < MLXL3_QMV_NT;
                    ++output_tile
                ) {
                    const device uint* words = trellis +
                        (tile_k * uint(TILES_N) + tile_n + output_tile)
                        * uint(PACKED_U32);
                    for (uint group = 0u; group < 2u; ++group) {
                        ulong merged = (ulong(words[word0[group]]) << 32) |
                                       ulong(words[word1[group]]);
                        uint shift = shifts[group];
                        uint window = uint(merged >> shift);
                        uint cw3 = window & 0xffffu;
                        uint cw2 = (window >> MLXL3_K_BITS) & 0xffffu;
                        uint cw1 = (window >> (2u * MLXL3_K_BITS)) & 0xffffu;
                        uint cw0 = MLXL3_K_BITS <= 5u
                            ? (window >> (3u * MLXL3_K_BITS)) & 0xffffu
                            : uint(merged >> (shift + 3u * MLXL3_K_BITS)) & 0xffffu;
                        uint j = group * 4u;
                        acc[output_tile][j] = fma(
                            x_values[0],
                            float(mlxl3_decode_codeword(cw0, CB)),
                            acc[output_tile][j]
                        );
                        acc[output_tile][j + 1u] = fma(
                            x_values[1],
                            float(mlxl3_decode_codeword(cw1, CB)),
                            acc[output_tile][j + 1u]
                        );
                        acc[output_tile][j + 2u] = fma(
                            x_values[2],
                            float(mlxl3_decode_codeword(cw2, CB)),
                            acc[output_tile][j + 2u]
                        );
                        acc[output_tile][j + 3u] = fma(
                            x_values[3],
                            float(mlxl3_decode_codeword(cw3, CB)),
                            acc[output_tile][j + 3u]
                        );
                    }
                }
            }

            for (uint output_tile = 0u; output_tile < MLXL3_QMV_NT; ++output_tile) {
                // The EXL3 permutation assigns each four-lane quadgroup one
                // output column and gives every lane four rows for columns
                // c and c + 8. Reduce those rows in registers instead of
                // materializing all 256 products in threadgroup memory.
                float column0 = quad_sum(
                    acc[output_tile][0] + acc[output_tile][1]
                ) + quad_sum(
                    acc[output_tile][2] + acc[output_tile][3]
                );
                float column1 = quad_sum(
                    acc[output_tile][4] + acc[output_tile][5]
                ) + quad_sum(
                    acc[output_tile][6] + acc[output_tile][7]
                );
                if ((lane & 3u) == 0u) {
                    uint column = lane >> 2u;
                    partials[simd][output_tile * 16u + column] = column0;
                    partials[simd][output_tile * 16u + column + 8u] = column1;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 16u * MLXL3_QMV_NT) {
                uint output_tile = tid >> 4u;
                uint column = tid & 15u;
                float sum = 0.0f;
                for (uint group = 0u; group < MLXL3_QMV_SG; ++group) {
                    sum += partials[group][output_tile * 16u + column];
                }
                if (MLXL3_FUSE_OUTPUT) ha[tid] = float(half(sum));
                else yhat[
                    split * uint(OUTPUT_DIMS)
                    + (tile_n + output_tile) * 16u + column
                ] = sum;
            }
            if (MLXL3_FUSE_OUTPUT) {
                for (uint shift = 0u; shift < 7u; ++shift) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < 128u) {
                        uint h = 1u << shift;
                        float own = ha[tid], peer = ha[tid ^ h];
                        float value = (tid & h) ? peer - own : own + peer;
                        hb[tid] = shift == 3u ? float(half(value)) : value;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < 128u) ha[tid] = hb[tid];
                }
                if (tid < 128u) {
                    uint index = tile_n * 16u + tid;
                    half rotated = half(float(half(ha[tid])) * 0.08838834764831845f);
                    yhat[index] = half(rotated * half(svh[index]));
                }
            }
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _qmv_mapped_tile_kernel(
    k: int,
    mode: int,
    input_dims: int,
    source_tiles: int,
    tiles_per_group: int = 1,
    simdgroups: int = 4,
    k3_window_decode: bool = False,
):
    """Tile-cooperative QMV over an indirect list of expert output tiles."""

    if tiles_per_group not in (1, 2, 4):
        raise ValueError(f"unsupported mapped QMV output tile group {tiles_per_group}")
    if simdgroups not in (2, 4, 8) or simdgroups * tiles_per_group > 32:
        raise ValueError(
            f"unsupported mapped QMV shape: {simdgroups} simdgroups x "
            f"{tiles_per_group} tiles"
        )

    return mx.fast.metal_kernel(
        name=(
            f"mlxl3_qmv_mapped_k{k}_cb{mode}_{input_dims}x{source_tiles}"
            f"_nt{tiles_per_group}_sg{simdgroups}_k3w{int(k3_window_decode)}_v12"
        ),
        input_names=["xhat", "trellis", "tile_map", "tile_sub"],
        output_names=["yhat"],
        header=(
            f"#define MLXL3_QMV_NT {tiles_per_group}u\n"
            f"#define MLXL3_QMV_SG {simdgroups}u\n"
            f"#define MLXL3_K_BITS {k}u\n"
            f"#define MLXL3_K3_WINDOW_DECODE {int(k3_window_decode)}\n"
            + specialized_codebook_header(mode)
        ),
        source=r"""
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint tile_group = threadgroup_position_in_grid.x;
            uint local_tile = tile_group * MLXL3_QMV_NT;
            uint split = threadgroup_position_in_grid.z;
            threadgroup float partials[MLXL3_QMV_SG][MLXL3_QMV_NT * 16u];

#if MLXL3_K3_WINDOW_DECODE
            int k3_bit_end = int((lane * 8u + 257u) * 3u + 21u);
            int k3_word_end = (k3_bit_end - 1) / 32;
            uint k3_word1 = uint(k3_word_end) % uint(PACKED_U32);
            uint k3_word0 = uint((k3_bit_end - 21 - 16) / 32) % uint(PACKED_U32);
            uint k3_shift = uint((k3_word_end + 1) * 32 - k3_bit_end);
#else
            uint word0[2];
            uint word1[2];
            uint shifts[2];
            for (uint group = 0u; group < 2u; ++group) {
                uint first = lane * 8u + group * 4u;
                int end = int(first + 4u) * K + 256 * K;
                int last_word = (end - 1) / 32;
                word1[group] = uint(last_word) % uint(PACKED_U32);
                word0[group] = uint(last_word - 1) % uint(PACKED_U32);
                shifts[group] = uint((last_word + 1) * 32 - end);
            }
#endif

            float acc[MLXL3_QMV_NT][8];
            for (uint output_tile = 0u; output_tile < MLXL3_QMV_NT; ++output_tile) {
                for (uint j = 0u; j < 8u; ++j) {
                    acc[output_tile][j] = 0.0f;
                }
            }
            uint tiles_per_split =
                (uint(TILES_K) + uint(N_SPLITS) - 1u) / uint(N_SPLITS);
            uint tile_begin = split * tiles_per_split;
            uint tile_end = min(tile_begin + tiles_per_split, uint(TILES_K));
            uint sub;
            uint tile_ns[MLXL3_QMV_NT];
            if (EXPERT_MAP) {
                uint groups_per_row = uint(OUTPUT_TILES) / MLXL3_QMV_NT;
                sub = tile_group / groups_per_row;
                uint group_in_row = tile_group - sub * groups_per_row;
                uint route = sub / uint(ROUTING_REPEAT);
                uint projection = sub - route * uint(ROUTING_REPEAT);
                uint source_base =
                    tile_map[route] * uint(OUTPUT_TILES)
                    + projection * uint(PROJECTION_STRIDE_TILES)
                    + group_in_row * MLXL3_QMV_NT;
                for (
                    uint output_tile = 0u;
                    output_tile < MLXL3_QMV_NT;
                    ++output_tile
                ) {
                    tile_ns[output_tile] = source_base + output_tile;
                }
            } else {
                sub = tile_sub[local_tile];
                for (
                    uint output_tile = 0u;
                    output_tile < MLXL3_QMV_NT;
                    ++output_tile
                ) {
                    uint tile_offset = local_tile + output_tile;
                    tile_ns[output_tile] =
                        IDENTITY_MAP ? tile_offset : tile_map[tile_offset];
                }
            }
            for (
                uint tile_k = tile_begin + simd;
                tile_k < tile_end;
                tile_k += MLXL3_QMV_SG
            ) {
                // Output rows are 128-aligned, so both tiles in a pair use
                // the same transformed activation row.
                const device half2* x_pairs = reinterpret_cast<const device half2*>(
                    xhat + sub * uint(INPUT_DIMS) + tile_k * 16u
                );
                half2 x_pair = x_pairs[lane & 7u];
                half2 x_pair0 = simd_shuffle(x_pair, ushort(lane & 3u));
                half2 x_pair1 = simd_shuffle(x_pair, ushort((lane & 3u) + 4u));
                float x_values[4];
                x_values[0] = float(x_pair0.x);
                x_values[1] = float(x_pair0.y);
                x_values[2] = float(x_pair1.x);
                x_values[3] = float(x_pair1.y);
                for (
                    uint output_tile = 0u;
                    output_tile < MLXL3_QMV_NT;
                    ++output_tile
                ) {
                    const device uint* words = trellis +
                        (tile_k * uint(TILES_N) + tile_ns[output_tile])
                        * uint(PACKED_U32);
#if MLXL3_K3_WINDOW_DECODE
                    ulong merged = (ulong(words[k3_word0]) << 32)
                        | ulong(words[k3_word1]);
                    uint codewords[8];
                    uint window7 = uint(merged >> k3_shift);
                    codewords[7] = window7;
                    codewords[6] = window7 >> 3u;
                    codewords[5] = window7 >> 6u;
                    codewords[4] = window7 >> 9u;
                    uint window3 = uint(merged >> (k3_shift + 12u));
                    codewords[3] = window3;
                    codewords[2] = window3 >> 3u;
                    codewords[1] = window3 >> 6u;
                    codewords[0] = window3 >> 9u;
                    for (uint j = 0u; j < 8u; ++j) {
                        acc[output_tile][j] = fma(
                            x_values[j & 3u],
                            float(mlxl3_decode_codeword(codewords[j], CB)),
                            acc[output_tile][j]
                        );
                    }
#else
                    for (uint group = 0u; group < 2u; ++group) {
                        ulong merged = (ulong(words[word0[group]]) << 32) |
                                       ulong(words[word1[group]]);
                        uint shift = shifts[group];
                        uint window = uint(merged >> shift);
                        uint cw3 = window & 0xffffu;
                        uint cw2 = (window >> MLXL3_K_BITS) & 0xffffu;
                        uint cw1 = (window >> (2u * MLXL3_K_BITS)) & 0xffffu;
                        uint cw0 = MLXL3_K_BITS <= 5u
                            ? (window >> (3u * MLXL3_K_BITS)) & 0xffffu
                            : uint(merged >> (shift + 3u * MLXL3_K_BITS)) & 0xffffu;
                        uint j = group * 4u;
                        acc[output_tile][j] = fma(
                            x_values[0],
                            float(mlxl3_decode_codeword(cw0, CB)),
                            acc[output_tile][j]
                        );
                        acc[output_tile][j + 1u] = fma(
                            x_values[1],
                            float(mlxl3_decode_codeword(cw1, CB)),
                            acc[output_tile][j + 1u]
                        );
                        acc[output_tile][j + 2u] = fma(
                            x_values[2],
                            float(mlxl3_decode_codeword(cw2, CB)),
                            acc[output_tile][j + 2u]
                        );
                        acc[output_tile][j + 3u] = fma(
                            x_values[3],
                            float(mlxl3_decode_codeword(cw3, CB)),
                            acc[output_tile][j + 3u]
                        );
                    }
#endif
                }
            }

            for (uint output_tile = 0u; output_tile < MLXL3_QMV_NT; ++output_tile) {
                float column0 = quad_sum(
                    acc[output_tile][0] + acc[output_tile][1]
                ) + quad_sum(
                    acc[output_tile][2] + acc[output_tile][3]
                );
                float column1 = quad_sum(
                    acc[output_tile][4] + acc[output_tile][5]
                ) + quad_sum(
                    acc[output_tile][6] + acc[output_tile][7]
                );
                if ((lane & 3u) == 0u) {
                    uint column = lane >> 2u;
                    partials[simd][output_tile * 16u + column] = column0;
                    partials[simd][output_tile * 16u + column + 8u] = column1;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 16u * MLXL3_QMV_NT) {
                uint output_tile = tid >> 4u;
                uint column = tid & 15u;
                float sum = 0.0f;
                for (uint group = 0u; group < MLXL3_QMV_SG; ++group) {
                    sum += partials[group][output_tile * 16u + column];
                }
                yhat[
                    split * uint(LOCAL_OUTPUT_DIMS)
                    + (local_tile + output_tile) * 16u + column
                ] = sum;
            }
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _qmm_tile_kernel(k: int, mode: int, input_dims: int, output_dims: int, mt: int):
    """Decode each EXL3 tile once for a 2/4/8-row activation group."""

    if mt not in (2, 4, 8):
        raise ValueError(f"unsupported QMM row tile {mt}")
    vector_width = 2 if mt == 2 else 4
    vectors = mt // vector_width
    vector_type = f"float{vector_width}"
    half_vector_type = f"half{vector_width}"
    return mx.fast.metal_kernel(
        name=f"mlxl3_qmm_tile_k{k}_cb{mode}_{input_dims}x{output_dims}_m{mt}_v1",
        input_names=["xhat", "trellis"],
        output_names=["yhat"],
        header=specialized_codebook_header(mode) + forward_permutation_header(),
        source=f"""
            #define MT {mt}u
            #define VW {vector_width}u
            #define NV {vectors}u

            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint input_phase = simd >> 1u;
            uint tile_half = simd & 1u;
            uint tile_n = threadgroup_position_in_grid.x;
            uint row_group = threadgroup_position_in_grid.y;
            uint split = threadgroup_position_in_grid.z;
            uint row_begin = row_group * MT;
            threadgroup float partials[2][256];

            uint positions[4];
            uint rows[4];
            for (uint j = 0u; j < 4u; ++j) {{
                uint code = tile_half * 128u + lane * 4u + j;
                positions[j] = uint(mlxl3_perm[code]);
                rows[j] = positions[j] >> 4u;
            }}

            uint code0 = tile_half * 128u + lane * 4u;
            int end = int(code0 + 4u) * K + 256 * K;
            int last_word = (end - 1) / 32;
            uint word1 = uint(last_word) % uint(PACKED_U32);
            uint word0 = uint(last_word - 1) % uint(PACKED_U32);
            uint shift = uint((last_word + 1) * 32 - end);

            {vector_type} acc[4][NV];
            for (uint j = 0u; j < 4u; ++j) {{
                for (uint q = 0u; q < NV; ++q) {{
                    acc[j][q] = {vector_type}(0.0f);
                }}
            }}

            uint tiles_per_split =
                (uint(TILES_K) + uint(N_SPLITS) - 1u) / uint(N_SPLITS);
            uint tile_begin = split * tiles_per_split;
            uint tile_end = min(tile_begin + tiles_per_split, uint(TILES_K));
            const device half* group_x = xhat + row_begin;
            for (uint tile_k = tile_begin + input_phase; tile_k < tile_end; tile_k += 2u) {{
                const device uint* words =
                    trellis + (tile_k * uint(TILES_N) + tile_n) * uint(PACKED_U32);
                ulong merged = (ulong(words[word0]) << 32) | ulong(words[word1]);
                uint codewords[4];
                codewords[3] = uint(merged >> shift) & 0xffffu;
                codewords[2] = uint(merged >> (shift + uint(K))) & 0xffffu;
                codewords[1] = uint(merged >> (shift + 2u * uint(K))) & 0xffffu;
                codewords[0] = uint(merged >> (shift + 3u * uint(K))) & 0xffffu;

                for (uint j = 0u; j < 4u; ++j) {{
                    float weight = mlxl3_decode_codeword(codewords[j], CB);
                    const device {half_vector_type}* values =
                        (const device {half_vector_type}*)(
                            group_x + (tile_k * 16u + rows[j]) * uint(PADDED_ROWS)
                        );
                    for (uint q = 0u; q < NV; ++q) {{
                        acc[j][q] = fma(
                            {vector_type}(values[q]), {vector_type}(weight), acc[j][q]
                        );
                    }}
                }}
            }}

            for (uint mm = 0u; mm < MT; ++mm) {{
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint j = 0u; j < 4u; ++j) {{
                    partials[input_phase][positions[j]] = acc[j][mm / VW][mm % VW];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid < 16u && row_begin + mm < uint(ROWS)) {{
                    float sum = 0.0f;
                    for (uint row = 0u; row < 16u; ++row) {{
                        uint position = row * 16u + tid;
                        sum += partials[0][position] + partials[1][position];
                    }}
                    uint output_index =
                        ((row_begin + mm) * uint(N_SPLITS) + split) *
                        uint(OUTPUT_DIMS) + tile_n * 16u + tid;
                    yhat[output_index] = sum;
                }}
            }}
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _qmm_matrix_kernel(
    k: int,
    mode: int,
    input_dims: int,
    output_dims: int,
    block_rows: int,
):
    """32/64x64 EXL3 GEMM tile using Metal simdgroup matrix instructions."""

    if block_rows not in (32, 64):
        raise ValueError(f"unsupported matrix QMM row block {block_rows}")
    row_fragments = block_rows // 16
    simd_rows = block_rows // 2
    matrix_header = (
        "#include <metal_simdgroup_matrix>\n"
        f"#define MLXL3_BM {block_rows}u\n"
        f"#define MLXL3_ROW_FRAGMENTS {row_fragments}u\n"
        f"#define MLXL3_SIMD_ROWS {simd_rows}u\n"
    )
    return _qmm_matrix_kernel_impl(
        k, mode, input_dims, output_dims, block_rows, matrix_header
    )


@cache
def _qmm_tensor_kernel(
    k: int,
    mode: int,
    input_dims: int,
    output_dims: int,
    block_rows: int,
    block_columns: int = 32,
    block_depth: int = 32,
    weight_tiles_n: int | None = None,
    weight_tile_offset: int = 0,
):
    """EXL3 QMM using M5 TensorOps with on-chip cooperative dequantization."""

    if block_rows not in (8, 16, 32):
        raise ValueError(f"unsupported TensorOp QMM row block {block_rows}")
    if block_columns not in (16, 32, 64):
        raise ValueError(f"unsupported TensorOp QMM column block {block_columns}")
    if block_depth not in (16, 32, 64):
        raise ValueError(f"unsupported TensorOp QMM depth block {block_depth}")
    if input_dims % block_depth or output_dims % block_columns:
        raise ValueError("TensorOp QMM requires exactly tiled matrix dimensions")
    return mx.fast.metal_kernel(
        name=(
            f"mlxl3_qmm_tensor_k{k}_cb{mode}_{input_dims}x{output_dims}"
            f"_m{block_rows}_n{block_columns}_d{block_depth}_s{weight_tiles_n or output_dims // 16}_o{weight_tile_offset}_v2"
        ),
        input_names=["xhat", "trellis"],
        output_names=["yhat"],
        header=(
            "#include <metal_tensor>\n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            "using namespace metal;\n"
            "using namespace mpp;\n"
            + specialized_codebook_header(mode)
            + permutation_header()
        ),
        source=f"""
            constexpr uint BM = {block_rows}u;
            constexpr uint BN = {block_columns}u;
            constexpr uint BK = {block_depth}u;
            constexpr uint K_BITS = {k}u;
            constexpr uint PACKED_U32 = {k * 8}u;
            constexpr uint INPUT_DIMS = {input_dims}u;
            constexpr uint OUTPUT_DIMS = {output_dims}u;
            constexpr uint TILES_N = {weight_tiles_n or output_dims // 16}u;

            uint column_block = threadgroup_position_in_grid.x;
            uint row_block = threadgroup_position_in_grid.y;
            device half* activation = const_cast<device half*>(xhat);

            constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
                BM,
                BN,
                BK,
                false,
                false,
                false,
                tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
            );
            tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
            auto right = operation.get_right_input_cooperative_tensor<
                half, half, float
            >();
            auto first_left = tensor(
                activation + ulong(row_block * BM) * INPUT_DIMS,
                dextents<int, 2>{{int(BK), int(BM)}},
                array<int, 2>{{1, int(INPUT_DIMS)}}
            );
            auto accumulator = operation.get_destination_cooperative_tensor<
                decltype(first_left), decltype(right), float
            >();
            for (ushort index = 0; index < accumulator.get_capacity(); ++index) {{
                if (accumulator.is_valid_element(index)) {{
                    accumulator[index] = 0.0f;
                }}
            }}

            for (uint depth = 0u; depth < INPUT_DIMS; depth += BK) {{
                for (ushort index = 0; index < right.get_capacity(); ++index) {{
                    if (!right.is_valid_element(index)) {{
                        continue;
                    }}
                    auto coordinate = right.get_multidimensional_index(index);
                    uint output_column = column_block * BN + uint(coordinate[0]);
                    uint input_row = depth + uint(coordinate[1]);
                    uint tile_k = input_row >> 4u;
                    uint tile_n = (output_column >> 4u) + {weight_tile_offset}u;
                    uint local = (input_row & 15u) * 16u + (output_column & 15u);
                    uint source = uint(mlxl3_perm_inv[local]);
                    uint position = source >> 1u;
                    int begin = int(position * 2u * K_BITS + K_BITS)
                        - 16 + int(256u * K_BITS);
                    int end = begin + int(K_BITS) + 16;
                    uint word0 = uint(begin / 32) % PACKED_U32;
                    uint word1 = uint((end - 1) / 32) % PACKED_U32;
                    uint shift = uint(((end - 1) / 32 + 1) * 32 - end)
                        + ((source & 1u) ? 0u : K_BITS);
                    const device uint* words = trellis
                        + ulong(tile_k * TILES_N + tile_n) * PACKED_U32;
                    ulong merged = (ulong(words[word0]) << 32) | ulong(words[word1]);
                    uint codeword = uint(merged >> shift) & 0xffffu;
                    right[index] = half(mlxl3_decode_codeword(codeword, 0));
                }}
                auto left = tensor(
                    activation + ulong(row_block * BM) * INPUT_DIMS + depth,
                    dextents<int, 2>{{int(BK), int(BM)}},
                    array<int, 2>{{1, int(INPUT_DIMS)}}
                );
                operation.run(left, right, accumulator);
            }}

            for (ushort index = 0; index < accumulator.get_capacity(); ++index) {{
                if (!accumulator.is_valid_element(index)) {{
                    continue;
                }}
                auto coordinate = accumulator.get_multidimensional_index(index);
                uint output_row = row_block * BM + uint(coordinate[1]);
                uint output_column = column_block * BN + uint(coordinate[0]);
                yhat[ulong(output_row) * OUTPUT_DIMS + output_column] =
                    half(accumulator[index]);
            }}
        """,
        compile_options={"math_mode": "safe"},
    )


def _qmm_matrix_kernel_impl(
    k: int,
    mode: int,
    input_dims: int,
    output_dims: int,
    block_rows: int,
    matrix_header: str,
):
    return mx.fast.metal_kernel(
        name=(
            f"mlxl3_qmm_matrix_k{k}_cb{mode}_{input_dims}x{output_dims}"
            f"_m{block_rows}_v2"
        ),
        input_names=["xhat", "trellis"],
        output_names=["yhat"],
        header=(
            matrix_header
            + specialized_codebook_header(mode)
            + permutation_header()
        ),
        source=r"""
            #define BM MLXL3_BM
            #define BN 64u
            #define WS 72u

            uint col_block = threadgroup_position_in_grid.x;
            uint row_block = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint simd_row = simd >> 1u;
            uint simd_col = simd & 1u;

            uint decoded_col = tid >> 3u;
            uint row_pair = tid & 7u;
            uint word0[2];
            uint word1[2];
            uint shifts[2];
            for (uint j = 0u; j < 2u; ++j) {
                uint source = uint(
                    mlxl3_perm_inv[(2u * row_pair + j) * 16u + decoded_col]
                );
                uint pair = source >> 1u;
                int bit0 = int(pair * 2u * uint(K) + uint(K))
                         - 16 + 256 * K;
                int bit2 = bit0 + K + 16;
                word0[j] = uint(bit0 / 32) % uint(PACKED_U32);
                word1[j] = uint((bit2 - 1) / 32) % uint(PACKED_U32);
                shifts[j] = uint(((bit2 - 1) / 32 + 1) * 32 - bit2)
                          + ((source & 1u) ? 0u : uint(K));
            }

            threadgroup half weights[32u * WS];
            threadgroup float scratch[4u * 64u];
            simdgroup_float8x8 accum[MLXL3_ROW_FRAGMENTS][4];
            for (uint row = 0u; row < MLXL3_ROW_FRAGMENTS; ++row) {
                for (uint col = 0u; col < 4u; ++col) {
                    accum[row][col] =
                        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }

            const device half* x_block =
                xhat + (row_block * BM + simd_row * MLXL3_SIMD_ROWS)
                    * uint(INPUT_DIMS);
            for (uint input_stage = 0u; input_stage < uint(TILES_K) / 2u; ++input_stage) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint input_half = 0u; input_half < 2u; ++input_half) {
                    uint tile_k = input_stage * 2u + input_half;
                    for (uint output_tile = 0u; output_tile < 4u; ++output_tile) {
                        uint tile_n = col_block * 4u + output_tile;
                        const device uint* words = trellis +
                            (tile_k * uint(TILES_N) + tile_n) * uint(PACKED_U32);
                        for (uint j = 0u; j < 2u; ++j) {
                            ulong merged =
                                (ulong(words[word0[j]]) << 32) | ulong(words[word1[j]]);
                            uint codeword = uint(merged >> shifts[j]) & 0xffffu;
                            weights[
                                (input_half * 16u + 2u * row_pair + j) * WS
                                + output_tile * 16u + decoded_col
                            ] = half(mlxl3_decode_codeword(codeword, CB));
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                for (uint depth = 0u; depth < 4u; ++depth) {
                    simdgroup_half8x8 weight0;
                    simdgroup_half8x8 weight1;
                    simdgroup_half8x8 weight2;
                    simdgroup_half8x8 weight3;
                    const threadgroup half* weight_row =
                        &weights[depth * 8u * WS + simd_col * 32u];
                    simdgroup_load(weight0, weight_row, WS);
                    simdgroup_load(weight1, weight_row + 8u, WS);
                    simdgroup_load(weight2, weight_row + 16u, WS);
                    simdgroup_load(weight3, weight_row + 24u, WS);
                    for (uint row = 0u; row < MLXL3_ROW_FRAGMENTS; ++row) {
                        simdgroup_half8x8 activation;
                        simdgroup_load(
                            activation,
                            x_block + row * 8u * uint(INPUT_DIMS)
                                + input_stage * 32u + depth * 8u,
                            uint(INPUT_DIMS)
                        );
                        simdgroup_multiply_accumulate(
                            accum[row][0], activation, weight0, accum[row][0]
                        );
                        simdgroup_multiply_accumulate(
                            accum[row][1], activation, weight1, accum[row][1]
                        );
                        simdgroup_multiply_accumulate(
                            accum[row][2], activation, weight2, accum[row][2]
                        );
                        simdgroup_multiply_accumulate(
                            accum[row][3], activation, weight3, accum[row][3]
                        );
                    }
                }
            }

            uint output_col = col_block * BN + simd_col * 32u;
            uint output_row = row_block * BM + simd_row * MLXL3_SIMD_ROWS;
            for (uint row = 0u; row < MLXL3_ROW_FRAGMENTS; ++row) {
                for (uint col = 0u; col < 4u; ++col) {
                    simdgroup_store(accum[row][col], &scratch[simd * 64u], 8u);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint index = lane; index < 64u; index += 32u) {
                        uint local_row = index >> 3u;
                        uint local_col = index & 7u;
                        yhat[
                            (output_row + row * 8u + local_row) * uint(OUTPUT_DIMS)
                            + output_col + col * 8u + local_col
                        ] = half(scratch[simd * 64u + index]);
                    }
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        """,
        compile_options={"math_mode": "safe"},
    )


def _hadamard_right(x: mx.array) -> mx.array:
    if x.shape[-1] % 128:
        raise ValueError(f"Hadamard dimension must be divisible by 128, got {x.shape[-1]}")
    shape = x.shape
    blocks = x.astype(mx.float16).reshape(*shape[:-1], shape[-1] // 128, 128)
    return mx.hadamard_transform(blocks, scale=_HADAMARD_SCALE).reshape(shape)


def _scaled_hadamard_input(x: mx.array, scale: mx.array) -> mx.array:
    if _USE_SCALED_HADAMARD:
        return _run_scaled_hadamard(x, scale, True)
    return _reference_scaled_hadamard_input(x, scale)


@mx.compile
def _reference_scaled_hadamard_input(x: mx.array, scale: mx.array) -> mx.array:
    """Fuse input scaling, fp16 rounding, and block Hadamards."""

    values = x.astype(mx.float16) * scale.astype(mx.float16)
    shape = values.shape
    blocks = values.reshape(*shape[:-1], shape[-1] // 128, 128)
    return mx.hadamard_transform(blocks, scale=_HADAMARD_SCALE).reshape(shape)


def _scaled_hadamard_output(x: mx.array, scale: mx.array) -> mx.array:
    if _USE_SCALED_HADAMARD:
        return _run_scaled_hadamard(x, scale, False)
    return _reference_scaled_hadamard_output(x, scale)


@mx.compile
def _reference_scaled_hadamard_output(x: mx.array, scale: mx.array) -> mx.array:
    """Fuse fp16 block Hadamards with their output scaling."""

    values = x.astype(mx.float16)
    shape = values.shape
    blocks = values.reshape(*shape[:-1], shape[-1] // 128, 128)
    transformed = mx.hadamard_transform(blocks, scale=_HADAMARD_SCALE).reshape(shape)
    return transformed * scale.astype(mx.float16)


@mx.compile
def _scaled_hadamard_output_reduce(
    x: mx.array,
    scale: mx.array,
    weights: mx.array,
) -> mx.array:
    """Finish routed expert rows and reduce them without an intermediate."""

    values = x.astype(mx.float16)
    shape = values.shape
    blocks = values.reshape(*shape[:-1], shape[-1] // 128, 128)
    transformed = mx.hadamard_transform(blocks, scale=_HADAMARD_SCALE).reshape(shape)
    finished = transformed * scale.astype(mx.float16)
    return (finished.reshape(weights.shape[0], weights.shape[1], shape[-1])
            * weights[..., None]).sum(axis=1)


def _split_count(input_tiles: int, output_tiles: int) -> int:
    # A wide projection already exposes enough independent output tiles to
    # occupy the GPU. Splitting it further only adds a partial-output buffer
    # and a reduction. Tall/narrow projections still need split-K to create
    # enough threadgroups, with shorter chunks for fewer output tiles.
    # Narrow outputs do not expose enough independent threadgroups. Halving
    # their per-split K target raised decode throughput on both 2048->1024 and
    # 3584->1024 projections while preserving wide-projection behavior.
    target_input_tiles = min(max(output_tiles, 32), 256)
    if output_tiles <= 64:
        target_input_tiles = max(output_tiles // 2, 32)
    splits = 1
    while input_tiles // (splits * 2) >= target_input_tiles:
        splits *= 2
    return splits


def _qmv_tiles_per_group(
    input_tiles: int,
    output_tiles: int,
    k: int,
    mode: CodebookMode,
) -> int:
    """Reuse each activation slice across adjacent output tiles."""

    del input_tiles, k, mode
    # Four tiles amortize activation routing and launch work best for ordinary
    # transformer projections. Very wide vocabulary heads already expose ample
    # parallelism and retain better occupancy with the smaller register tile.
    if output_tiles >= 1024:
        return 2 if output_tiles % 2 == 0 else 1
    if output_tiles % 4 == 0:
        return 4
    return 2 if output_tiles % 2 == 0 else 1


@cache
def _is_m5_gpu() -> bool:
    return str(mx.device_info().get("architecture", "")) == "applegpu_g17g"


@cache
def _tensor_ops_available() -> bool:
    """Return whether this host exposes the cooperative TensorOps used below."""

    if not _is_m5_gpu():
        return False
    try:
        version = tuple(int(part) for part in platform.mac_ver()[0].split(".")[:2])
    except ValueError:
        return False
    return version >= (26, 3)


def _qmv_simdgroups(
    input_tiles: int,
    output_tiles: int,
    k: int,
    mode: CodebookMode,
    mapped_rows: int,
) -> int:
    """Choose cooperative depth from end-to-end measurements on this GPU."""

    del input_tiles, output_tiles, mode
    # Microbenchmarks disagree slightly on narrow shapes, but full decode is
    # consistently faster with eight groups on dense projections. K=2 routed
    # experts need fewer lanes per output tile and retain better occupancy with
    # four groups; K>=3 still benefits from the extra cooperative depth.
    if _is_m5_gpu() and mapped_rows and k == 2:
        return 4
    return 8 if _is_m5_gpu() else 4


def qmv_exl3(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
    k: int,
    mode: CodebookMode | int = CodebookMode.DEFAULT,
) -> mx.array:
    """Compute one row ``x @ W`` directly from serialized EXL3 weights.

    This first implementation fuses trellis decode and the inner QMV. The two
    128-point Hadamard transforms remain ordinary MLX operations, providing a
    clear correctness baseline for the fully fused kernel.
    """

    cb = CodebookMode(mode)
    if trellis.dtype == mx.int16:
        trellis = trellis.view(mx.uint16)
    if trellis.dtype != mx.uint16 or trellis.ndim != 3 or trellis.shape[-1] != 16 * k:
        raise ValueError(f"invalid EXL3 trellis shape/dtype: {trellis.shape}/{trellis.dtype}")
    input_dims = trellis.shape[0] * 16
    output_dims = trellis.shape[1] * 16
    if x.shape[-1] != input_dims or x.size != input_dims:
        raise ValueError(f"QMV expects one row with {input_dims} inputs, got {x.shape}")

    original_shape = x.shape[:-1]
    xhat = _scaled_hadamard_input(x.reshape(1, input_dims), suh)[0]
    if k == 7:
        yhat = _qmv_inner_kernel(k, int(cb), input_dims, output_dims)(
            inputs=[xhat, trellis],
            template=[
                ("K", k),
                ("CB", int(cb)),
                ("INPUT_DIMS", input_dims),
                ("TILES_N", trellis.shape[1]),
            ],
            grid=(output_dims * 128, 1, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(output_dims,)],
            output_dtypes=[mx.float32],
        )[0]
    else:
        splits = _split_count(trellis.shape[0], trellis.shape[1])
        tiles_per_group = _qmv_tiles_per_group(
            trellis.shape[0], trellis.shape[1], k, cb
        )
        simdgroups = _qmv_simdgroups(
            trellis.shape[0], trellis.shape[1], k, cb, 0
        )
        if trellis.shape[1] % tiles_per_group:
            tiles_per_group = 1
        fused_output = _USE_QMV_OUTPUT_FUSION and splits == 1 and output_dims % 128 == 0
        if fused_output:
            tiles_per_group = 8
        partials = _qmv_tile_kernel(
            k, int(cb), input_dims, output_dims, tiles_per_group, simdgroups, fused_output
        )(
            inputs=[xhat, trellis.reshape(-1).view(mx.uint32), svh],
            template=[
                ("K", k),
                ("CB", int(cb)),
                ("PACKED_U32", k * 8),
                ("TILES_K", trellis.shape[0]),
                ("TILES_N", trellis.shape[1]),
                ("N_SPLITS", splits),
                ("OUTPUT_DIMS", output_dims),
            ],
            grid=(
                (trellis.shape[1] // tiles_per_group) * simdgroups * 32,
                1,
                splits,
            ),
            threadgroup=(simdgroups * 32, 1, 1),
            output_shapes=[(splits, output_dims)],
            output_dtypes=[mx.float32],
        )[0]
        if fused_output:
            return partials[0].astype(x.dtype).reshape(*original_shape, output_dims)
        yhat = partials[0] if splits == 1 else partials.sum(axis=0)
    output = _scaled_hadamard_output(yhat.reshape(1, output_dims), svh)[0]
    return output.astype(x.dtype).reshape(*original_shape, output_dims)


def qmv_exl3_mapped(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
    tile_map: mx.array,
    *,
    output_dims: int,
    k: int,
    mode: CodebookMode | int = CodebookMode.DEFAULT,
) -> mx.array:
    """Run equally shaped rows against indirectly selected expert matrices.

    ``trellis`` stacks expert matrices along its output-tile dimension.
    ``tile_map`` lists the source tiles for each output row, row-major.  This
    lets one Metal launch evaluate every selected expert instead of issuing a
    Python/Metal dispatch per projection.
    """

    cb = CodebookMode(mode)
    if k == 7:
        raise NotImplementedError("mapped K=7 requires the three-word decode path")
    if trellis.dtype == mx.int16:
        trellis = trellis.view(mx.uint16)
    if trellis.dtype != mx.uint16 or trellis.ndim != 3 or trellis.shape[-1] != 16 * k:
        raise ValueError(f"invalid EXL3 trellis shape/dtype: {trellis.shape}/{trellis.dtype}")
    if x.ndim != 2:
        raise ValueError(f"mapped QMV expects a row matrix, got {x.shape}")
    rows, input_dims = x.shape
    if input_dims != trellis.shape[0] * 16:
        raise ValueError(
            f"mapped QMV input width {input_dims} != trellis width {trellis.shape[0] * 16}"
        )
    if output_dims % 16:
        raise ValueError(f"output_dims must be a multiple of 16, got {output_dims}")
    output_tiles = output_dims // 16
    if tile_map.size != rows * output_tiles:
        raise ValueError(f"tile_map has {tile_map.size} entries, expected {rows * output_tiles}")
    if suh.shape != x.shape or svh.shape != (rows, output_dims):
        raise ValueError(
            f"scale shapes must be {x.shape} and {(rows, output_dims)}, "
            f"got {suh.shape} and {svh.shape}"
        )

    xhat = _scaled_hadamard_input(x, suh)
    tile_sub = mx.repeat(mx.arange(rows, dtype=mx.uint32), output_tiles)
    splits = _split_count(trellis.shape[0], tile_map.size)
    tiles_per_group = _qmv_tiles_per_group(
        trellis.shape[0], tile_map.size, k, cb
    )
    simdgroups = _qmv_simdgroups(
        trellis.shape[0], tile_map.size, k, cb, rows
    )
    if tile_map.size % tiles_per_group:
        tiles_per_group = 1
    partials = _qmv_mapped_tile_kernel(
        k, int(cb), input_dims, trellis.shape[1], tiles_per_group, simdgroups
    )(
        inputs=[
            xhat.reshape(-1),
            trellis.reshape(-1).view(mx.uint32),
            tile_map.astype(mx.uint32),
            tile_sub,
        ],
        template=[
            ("K", k),
            ("CB", int(cb)),
            ("PACKED_U32", k * 8),
            ("INPUT_DIMS", input_dims),
            ("TILES_K", trellis.shape[0]),
            ("TILES_N", trellis.shape[1]),
            ("N_SPLITS", splits),
            ("LOCAL_OUTPUT_DIMS", tile_map.size * 16),
            ("IDENTITY_MAP", 0),
            ("EXPERT_MAP", 0),
            ("OUTPUT_TILES", output_tiles),
            ("ROUTING_REPEAT", 1),
            ("PROJECTION_STRIDE_TILES", 0),
        ],
        grid=(
            (tile_map.size // tiles_per_group) * simdgroups * 32,
            1,
            splits,
        ),
        threadgroup=(simdgroups * 32, 1, 1),
        output_shapes=[(splits, rows, output_dims)],
        output_dtypes=[mx.float32],
    )[0]
    yhat = partials[0] if splits == 1 else partials.sum(axis=0)
    return _scaled_hadamard_output(yhat, svh).astype(x.dtype)


def qmv_exl3_expert_mapped(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array | None,
    svh: mx.array | None,
    selected: mx.array,
    *,
    output_dims: int,
    projections_per_route: int,
    projection_stride_tiles: int,
    reduce_weights: mx.array | None = None,
    k: int,
    mode: CodebookMode | int = CodebookMode.DEFAULT,
    input_pretransformed: bool = False,
    return_raw: bool = False,
) -> mx.array:
    """Run contiguous expert projections from compact routing indices."""

    cb = CodebookMode(mode)
    if k == 7:
        raise NotImplementedError("expert-mapped K=7 requires the three-word decode path")
    if trellis.dtype == mx.int16:
        trellis = trellis.view(mx.uint16)
    if trellis.dtype != mx.uint16 or trellis.ndim != 3 or trellis.shape[-1] != 16 * k:
        raise ValueError(f"invalid EXL3 trellis shape/dtype: {trellis.shape}/{trellis.dtype}")
    if x.ndim != 2:
        raise ValueError(f"expert-mapped QMV expects a row matrix, got {x.shape}")
    rows, input_dims = x.shape
    if input_dims != trellis.shape[0] * 16:
        raise ValueError(
            f"expert-mapped QMV input width {input_dims} != "
            f"trellis width {trellis.shape[0] * 16}"
        )
    if output_dims % 128:
        raise ValueError(f"expert output_dims must be a multiple of 128, got {output_dims}")
    if projections_per_route < 1:
        raise ValueError("projections_per_route must be positive")
    if selected.size * projections_per_route != rows:
        raise ValueError(
            f"{selected.size} routes x {projections_per_route} projections != {rows} rows"
        )
    if not input_pretransformed and (suh is None or suh.shape != x.shape):
        raise ValueError(
            f"input scales must be {x.shape}, got {None if suh is None else suh.shape}"
        )
    if not return_raw and (svh is None or svh.shape != (rows, output_dims)):
        raise ValueError(
            f"output scales must be {(rows, output_dims)}, "
            f"got {None if svh is None else svh.shape}"
        )
    if return_raw and reduce_weights is not None:
        raise ValueError("raw expert QMV cannot reduce before the output transform")
    if reduce_weights is not None and (
        reduce_weights.ndim != 2 or reduce_weights.size != selected.size
    ):
        raise ValueError(
            "reduce_weights must be a row/top-k matrix with "
            f"{selected.size} entries, got {reduce_weights.shape}"
        )

    output_tiles = output_dims // 16
    local_tiles = rows * output_tiles
    xhat = x if input_pretransformed else _scaled_hadamard_input(x, suh)
    routes = selected.reshape(-1).astype(mx.uint32)
    splits = _split_count(trellis.shape[0], local_tiles)
    tiles_per_group = _qmv_tiles_per_group(
        trellis.shape[0], local_tiles, k, cb
    )
    simdgroups = _qmv_simdgroups(
        trellis.shape[0], local_tiles, k, cb, rows
    )
    if output_tiles % tiles_per_group:
        tiles_per_group = 1
    partials = _qmv_mapped_tile_kernel(
        k,
        int(cb),
        input_dims,
        trellis.shape[1],
        tiles_per_group,
        simdgroups,
        _USE_K3_WINDOW_DECODE and k == 3,
    )(
        inputs=[
            xhat.reshape(-1),
            trellis.reshape(-1).view(mx.uint32),
            routes,
            routes,
        ],
        template=[
            ("K", k),
            ("CB", int(cb)),
            ("PACKED_U32", k * 8),
            ("INPUT_DIMS", input_dims),
            ("TILES_K", trellis.shape[0]),
            ("TILES_N", trellis.shape[1]),
            ("N_SPLITS", splits),
            ("LOCAL_OUTPUT_DIMS", local_tiles * 16),
            ("IDENTITY_MAP", 0),
            ("EXPERT_MAP", 1),
            ("OUTPUT_TILES", output_tiles),
            ("ROUTING_REPEAT", projections_per_route),
            ("PROJECTION_STRIDE_TILES", projection_stride_tiles),
        ],
        grid=(
            (local_tiles // tiles_per_group) * simdgroups * 32,
            1,
            splits,
        ),
        threadgroup=(simdgroups * 32, 1, 1),
        output_shapes=[(splits, rows, output_dims)],
        output_dtypes=[mx.float32],
    )[0]
    yhat = partials[0] if splits == 1 else partials.sum(axis=0)
    if return_raw:
        return yhat
    if reduce_weights is not None:
        if projections_per_route != 1:
            raise ValueError("expert reduction requires one projection per route")
        return _scaled_hadamard_output_reduce(
            yhat,
            svh,
            reduce_weights,
        ).astype(mx.result_type(x, reduce_weights))
    return _scaled_hadamard_output(yhat, svh).astype(x.dtype)


def qmv_exl3_grouped(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
    tile_sub: mx.array,
    *,
    output_dims: tuple[int, ...],
    k: int,
    mode: CodebookMode | int = CodebookMode.DEFAULT,
) -> tuple[mx.array, ...]:
    """Evaluate compatible ragged projections of the same activation together."""

    if len(output_dims) < 2:
        raise ValueError("grouped QMV requires at least two projections")
    if any(width % 128 for width in output_dims):
        raise ValueError(f"grouped QMV outputs must be 128-aligned, got {output_dims}")
    if trellis.dtype == mx.int16:
        trellis = trellis.view(mx.uint16)
    if trellis.dtype != mx.uint16 or trellis.ndim != 3 or trellis.shape[-1] != 16 * k:
        raise ValueError(f"invalid EXL3 trellis shape/dtype: {trellis.shape}/{trellis.dtype}")
    input_dims = trellis.shape[0] * 16
    total_output_dims = sum(output_dims)
    if trellis.shape[1] * 16 != total_output_dims:
        raise ValueError(
            f"grouped trellis has {trellis.shape[1] * 16} outputs, "
            f"expected {total_output_dims}"
        )
    if x.size != input_dims:
        raise ValueError(f"grouped QMV expects one row of width {input_dims}, got {x.shape}")
    if suh.shape != (len(output_dims), input_dims):
        raise ValueError(
            f"grouped input scales must be {(len(output_dims), input_dims)}, got {suh.shape}"
        )
    if svh.shape != (total_output_dims,):
        raise ValueError(f"grouped output scales must be {(total_output_dims,)}, got {svh.shape}")
    if tile_sub.shape != (trellis.shape[1],):
        raise ValueError(f"grouped tile map must be {(trellis.shape[1],)}, got {tile_sub.shape}")
    if k == 7:
        raise NotImplementedError("grouped K=7 requires the three-word decode path")

    cb = CodebookMode(mode)
    original_shape = x.shape[:-1]
    xhat = _scaled_hadamard_input(x.reshape(1, input_dims), suh)
    total_tiles = trellis.shape[1]
    identity_map = mx.zeros((1,), dtype=mx.uint32)
    splits = _split_count(trellis.shape[0], total_tiles)
    tiles_per_group = _qmv_tiles_per_group(
        trellis.shape[0], total_tiles, k, cb
    )
    simdgroups = _qmv_simdgroups(
        trellis.shape[0], total_tiles, k, cb, len(output_dims)
    )
    if total_tiles % tiles_per_group:
        tiles_per_group = 1
    partials = _qmv_mapped_tile_kernel(
        k, int(cb), input_dims, total_tiles, tiles_per_group, simdgroups
    )(
        inputs=[
            xhat.reshape(-1),
            trellis.reshape(-1).view(mx.uint32),
            identity_map,
            tile_sub.astype(mx.uint32),
        ],
        template=[
            ("K", k),
            ("CB", int(cb)),
            ("PACKED_U32", k * 8),
            ("INPUT_DIMS", input_dims),
            ("TILES_K", trellis.shape[0]),
            ("TILES_N", total_tiles),
            ("N_SPLITS", splits),
            ("LOCAL_OUTPUT_DIMS", total_output_dims),
            ("IDENTITY_MAP", 1),
            ("EXPERT_MAP", 0),
            ("OUTPUT_TILES", total_tiles),
            ("ROUTING_REPEAT", 1),
            ("PROJECTION_STRIDE_TILES", 0),
        ],
        grid=(
            (total_tiles // tiles_per_group) * simdgroups * 32,
            1,
            splits,
        ),
        threadgroup=(simdgroups * 32, 1, 1),
        output_shapes=[(splits, total_output_dims)],
        output_dtypes=[mx.float32],
    )[0]
    yhat = partials[0] if splits == 1 else partials.sum(axis=0)
    output = _scaled_hadamard_output(
        yhat.reshape(1, total_output_dims), svh
    )[0].astype(x.dtype)
    boundaries = []
    cursor = 0
    for width in output_dims[:-1]:
        cursor += width
        boundaries.append(cursor)
    return tuple(
        part.reshape(*original_shape, width)
        for part, width in zip(mx.split(output, boundaries), output_dims, strict=True)
    )


@cache
def _segmented_expert_qmm_kernel(k: int, mode: int):
    """Decode each routed expert tile once per 64-row Metal GEMM block."""

    packed_u32 = k * 256 // 32
    return mx.fast.metal_kernel(
        name=f"mlxl3_expert_qmm_segmented_k{k}_cb{mode}_v1",
        input_names=["xhat", "trellis", "block_table", "block_count", "dims"],
        output_names=["output"],
        header=(
            "#include <metal_simdgroup_matrix>\n"
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            + specialized_codebook_header(mode)
            + permutation_header()
        ),
        source=f"""
            constexpr uint PACKED_U32 = {packed_u32}u;
            constexpr uint K_BITS = {k}u;
            constexpr uint BM = 64u;
            constexpr uint BN = 64u;
            constexpr uint WEIGHT_STRIDE = 72u;

            uint block = threadgroup_position_in_grid.y;
            if (block >= block_count[0]) {{
                return;
            }}
            uint column_block = threadgroup_position_in_grid.x;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint simd_row = simd >> 1u;
            uint simd_column = simd & 1u;

            uint input_tiles = dims[0];
            uint tiles_per_expert = dims[1];
            uint source_tiles = dims[2];
            uint tile_base = dims[3];
            uint output_dims = dims[4];
            uint max_blocks = dims[5];
            uint input_dims = input_tiles * 16u;

            uint expert = block_table[block];
            uint row_begin = block_table[max_blocks + block];
            uint block_rows = block_table[2u * max_blocks + block];
            uint output_tile =
                tile_base + expert * tiles_per_expert + column_block * 4u;

            uint decoded_column = tid >> 3u;
            uint row_pair = tid & 7u;
            uint word0[2];
            uint word1[2];
            uint shifts[2];
            for (uint pair = 0u; pair < 2u; ++pair) {{
                uint source = mlxl3_perm_inv[
                    (2u * row_pair + pair) * 16u + decoded_column
                ];
                uint position = source >> 1u;
                int begin = int(position) * 2 * int(K_BITS)
                    + int(K_BITS) - 16 + 256 * int(K_BITS);
                int end = begin + int(K_BITS) + 16;
                word0[pair] = uint(begin / 32) % PACKED_U32;
                word1[pair] = uint((end - 1) / 32) % PACKED_U32;
                shifts[pair] = uint(((end - 1) / 32 + 1) * 32 - end)
                    + ((source & 1u) ? 0u : K_BITS);
            }}

            threadgroup half weights[32u * WEIGHT_STRIDE];
            threadgroup float scratch[4u * 64u];
            simdgroup_float8x8 accum[4][4];
            for (uint row = 0u; row < 4u; ++row) {{
                for (uint column = 0u; column < 4u; ++column) {{
                    accum[row][column] =
                        make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }}
            }}

            const device half* input_row = xhat
                + ulong(row_begin + simd_row * 32u) * input_dims;
            uint stages = input_tiles >> 1u;
            for (uint stage = 0u; stage < stages; ++stage) {{
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint inner_tile = 0u; inner_tile < 2u; ++inner_tile) {{
                    uint input_tile = stage * 2u + inner_tile;
                    for (uint local_tile = 0u; local_tile < 4u; ++local_tile) {{
                        const device uint* words = trellis
                            + ulong(input_tile * source_tiles
                                + output_tile + local_tile) * PACKED_U32;
                        for (uint pair = 0u; pair < 2u; ++pair) {{
                            ulong merged =
                                (ulong(words[word0[pair]]) << 32)
                                | ulong(words[word1[pair]]);
                            uint codeword = uint(merged >> shifts[pair]) & 0xffffu;
                            weights[
                                (inner_tile * 16u + 2u * row_pair + pair)
                                    * WEIGHT_STRIDE
                                + local_tile * 16u + decoded_column
                            ] = half(mlxl3_decode_codeword(codeword, 0));
                        }}
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k_fragment = 0u; k_fragment < 4u; ++k_fragment) {{
                    simdgroup_half8x8 weight0;
                    simdgroup_half8x8 weight1;
                    simdgroup_half8x8 weight2;
                    simdgroup_half8x8 weight3;
                    const threadgroup half* weight_row =
                        &weights[k_fragment * 8u * WEIGHT_STRIDE
                            + simd_column * 32u];
                    simdgroup_load(weight0, weight_row, WEIGHT_STRIDE);
                    simdgroup_load(weight1, weight_row + 8u, WEIGHT_STRIDE);
                    simdgroup_load(weight2, weight_row + 16u, WEIGHT_STRIDE);
                    simdgroup_load(weight3, weight_row + 24u, WEIGHT_STRIDE);
                    for (uint row = 0u; row < 4u; ++row) {{
                        simdgroup_half8x8 input_fragment;
                        simdgroup_load(
                            input_fragment,
                            input_row + ulong(row * 8u) * input_dims
                                + stage * 32u + k_fragment * 8u,
                            input_dims
                        );
                        simdgroup_multiply_accumulate(
                            accum[row][0], input_fragment, weight0, accum[row][0]
                        );
                        simdgroup_multiply_accumulate(
                            accum[row][1], input_fragment, weight1, accum[row][1]
                        );
                        simdgroup_multiply_accumulate(
                            accum[row][2], input_fragment, weight2, accum[row][2]
                        );
                        simdgroup_multiply_accumulate(
                            accum[row][3], input_fragment, weight3, accum[row][3]
                        );
                    }}
                }}
            }}

            uint output_column = column_block * BN + simd_column * 32u;
            uint output_row = simd_row * 32u;
            for (uint row = 0u; row < 4u; ++row) {{
                if (output_row + row * 8u >= block_rows) {{
                    break;
                }}
                for (uint column = 0u; column < 4u; ++column) {{
                    simdgroup_store(accum[row][column], &scratch[simd * 64u], 8u);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    for (uint index = lane; index < 64u; index += 32u) {{
                        uint local_row = index >> 3u;
                        uint local_column = index & 7u;
                        uint global_row = output_row + row * 8u + local_row;
                        if (global_row < block_rows) {{
                            output[
                                ulong(row_begin + global_row) * output_dims
                                + output_column + column * 8u + local_column
                            ] = half(scratch[simd * 64u + index]);
                        }}
                    }}
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }}
            }}
        """,
        compile_options={"math_mode": "safe"},
    )


def _make_segmented_tensor_kernel(*, bucketed, block_rows, **kwargs):
    if bucketed:
        # One GPU dispatch. The device-side route table selects a descriptor;
        # no host readback, second output buffer or merge/reduction is needed.
        source = kwargs['source']
        marker = 'uint local_row_begin = row_subblock * BM;'
        prefix, body = source.split(marker)
        prefix = prefix.replace(f'constexpr uint BM = {block_rows}u;', '')
        body = marker + body
        kwargs['source'] = prefix + (
            'if (segment_rows <= 8u) { constexpr uint BM = 8u;' + body + '}'
            'else if (segment_rows <= 16u) { constexpr uint BM = 16u;' + body + '}'
            'else { constexpr uint BM = 32u;' + body + '}'
        )
    return mx.fast.metal_kernel(**kwargs)


@cache
def _segmented_expert_qmm_tensor_kernel(k: int, mode: int, block_rows: int = 32, bucketed: bool = False, locality: bool = False):
    """M5 TensorOp QMM over sorted expert rows and EXL3 trellises."""

    packed_u32 = k * 256 // 32
    subblocks = 8 if bucketed else 64 // block_rows
    # Interleave row subblocks of the same expert/column tile. The work and
    # per-output accumulation order are identical; only dispatch coordinates
    # change so a serialized weight tile can stay hot across its consumers.
    row_coordinate = f'threadgroup_position_in_grid.x % {subblocks}u' if locality else 'threadgroup_position_in_grid.z'
    column_coordinate = f'threadgroup_position_in_grid.x / {subblocks}u' if locality else 'threadgroup_position_in_grid.x'
    return _make_segmented_tensor_kernel(
        bucketed=bucketed, block_rows=block_rows,
        name=f"mlxl3_expert_qmm_tensor_segmented_k{k}_cb{mode}_bm{block_rows}_b{int(bucketed)}_l{int(locality)}_v4",
        input_names=["xhat", "trellis", "block_table", "block_count", "dims"],
        output_names=["output"],
        header=(
            "#include <metal_tensor>\n"
            "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
            "using namespace metal;\n"
            "using namespace mpp;\n"
            + specialized_codebook_header(mode)
            + permutation_header()
        ),
        source=f"""
            constexpr uint BM = {block_rows}u;
            constexpr uint BN = 32u;
            constexpr uint BK = 16u;
            constexpr uint PACKED_U32 = {packed_u32}u;
            constexpr uint K_BITS = {k}u;

            uint block = threadgroup_position_in_grid.y;
            if (block >= block_count[0]) {{
                return;
            }}
            uint row_subblock = {row_coordinate};
            uint column_block = {column_coordinate};
            uint input_tiles = dims[0];
            uint tiles_per_expert = dims[1];
            uint source_tiles = dims[2];
            uint tile_base = dims[3];
            uint output_dims = dims[4];
            uint max_blocks = dims[5];
            uint input_dims = input_tiles * 16u;

            uint expert = block_table[block];
            uint segment_begin = block_table[max_blocks + block];
            uint segment_rows = block_table[2u * max_blocks + block];
            uint local_row_begin = row_subblock * BM;
            if (local_row_begin >= segment_rows) {{
                return;
            }}
            uint row_begin = segment_begin + local_row_begin;
            uint output_tile = tile_base + expert * tiles_per_expert
                + column_block * (BN / 16u);
            device half* activation = const_cast<device half*>(xhat);

            constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
                BM,
                BN,
                BK,
                false,
                false,
                false,
                tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
            );
            tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
            auto right = operation.get_right_input_cooperative_tensor<
                half, half, float
            >();
            auto first_left = tensor(
                activation + ulong(row_begin) * input_dims,
                dextents<int, 2>{{int(BK), int(BM)}},
                array<int, 2>{{1, int(input_dims)}}
            );
            auto accumulator = operation.get_destination_cooperative_tensor<
                decltype(first_left), decltype(right), float
            >();
            for (ushort index = 0; index < accumulator.get_capacity(); ++index) {{
                if (accumulator.is_valid_element(index)) {{
                    accumulator[index] = 0.0f;
                }}
            }}

            for (uint depth = 0u; depth < input_dims; depth += BK) {{
                for (ushort index = 0; index < right.get_capacity(); ++index) {{
                    if (!right.is_valid_element(index)) {{
                        continue;
                    }}
                    auto coordinate = right.get_multidimensional_index(index);
                    uint local_column = uint(coordinate[0]);
                    uint input_row = depth + uint(coordinate[1]);
                    uint tile_k = input_row >> 4u;
                    uint tile_n = output_tile + (local_column >> 4u);
                    uint local = (input_row & 15u) * 16u + (local_column & 15u);
                    uint source = uint(mlxl3_perm_inv[local]);
                    uint position = source >> 1u;
                    int begin = int(position * 2u * K_BITS + K_BITS)
                        - 16 + int(256u * K_BITS);
                    int end = begin + int(K_BITS) + 16;
                    uint word0 = uint(begin / 32) % PACKED_U32;
                    uint word1 = uint((end - 1) / 32) % PACKED_U32;
                    uint shift = uint(((end - 1) / 32 + 1) * 32 - end)
                        + ((source & 1u) ? 0u : K_BITS);
                    const device uint* words = trellis
                        + ulong(tile_k * source_tiles + tile_n) * PACKED_U32;
                    ulong merged = (ulong(words[word0]) << 32) | ulong(words[word1]);
                    uint codeword = uint(merged >> shift) & 0xffffu;
                    right[index] = half(mlxl3_decode_codeword(codeword, 0));
                }}
                auto left = tensor(
                    activation + ulong(row_begin) * input_dims + depth,
                    dextents<int, 2>{{int(BK), int(BM)}},
                    array<int, 2>{{1, int(input_dims)}}
                );
                operation.run(left, right, accumulator);
            }}

            for (ushort index = 0; index < accumulator.get_capacity(); ++index) {{
                if (!accumulator.is_valid_element(index)) {{
                    continue;
                }}
                auto coordinate = accumulator.get_multidimensional_index(index);
                uint local_row = uint(coordinate[1]);
                if (local_row_begin + local_row >= segment_rows) {{
                    continue;
                }}
                uint output_row = row_begin + local_row;
                uint output_column = column_block * BN + uint(coordinate[0]);
                output[ulong(output_row) * output_dims + output_column] =
                    half(accumulator[index]);
            }}
        """,
        compile_options={"math_mode": "safe"},
    )


def qmm_exl3_expert_segmented(
    xhat_sorted: mx.array,
    trellis: mx.array,
    block_table: mx.array,
    block_count: mx.array,
    *,
    rows: int,
    tiles_per_expert: int,
    output_dims: int,
    expert_tile_base: int,
    k: int,
    mode: CodebookMode | int,
) -> mx.array:
    """Multiply expert-sorted rows without materializing decoded weights."""

    if rows < 1 or xhat_sorted.shape[0] < rows + 64:
        raise ValueError("segmented expert QMM input needs 64 padded rows")
    if output_dims % 64 or trellis.shape[0] % 2:
        raise ValueError("segmented expert QMM requires 64-aligned matrices")
    cb = CodebookMode(mode)
    max_blocks = int(block_table.shape[1])
    dims = mx.array(
        [
            trellis.shape[0],
            tiles_per_expert,
            trellis.shape[1],
            expert_tile_base,
            output_dims,
            max_blocks,
        ],
        dtype=mx.uint32,
    )
    kernel = (
        _segmented_expert_qmm_tensor_kernel(k, int(cb), _SEGMENTED_TENSOR_ROWS, _USE_SEGMENTED_BUCKETS, _USE_SEGMENTED_LOCALITY)
        if _USE_TENSOR_SEGMENTED_QMM and _tensor_ops_available()
        else _segmented_expert_qmm_kernel(k, int(cb))
    )
    grid = (
        ((output_dims // 32) * 32, max_blocks, 8 if _USE_SEGMENTED_BUCKETS else 64 // _SEGMENTED_TENSOR_ROWS)
        if _USE_TENSOR_SEGMENTED_QMM and _tensor_ops_available()
        else ((output_dims // 64) * 128, max_blocks, 1)
    )
    if _USE_SEGMENTED_LOCALITY and _USE_TENSOR_SEGMENTED_QMM and _tensor_ops_available():
        grid = (grid[0] * grid[2], grid[1], 1)
    threadgroup = (
        (32, 1, 1)
        if _USE_TENSOR_SEGMENTED_QMM and _tensor_ops_available()
        else (128, 1, 1)
    )
    return kernel(
        inputs=[
            xhat_sorted.reshape(-1),
            trellis.reshape(-1).view(mx.uint32),
            block_table.reshape(-1),
            block_count,
            dims,
        ],
        template=[("T", mx.float16)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(rows, output_dims)],
        output_dtypes=[mx.float16],
    )[0]


def qmm_exl3_view(x, trellis, suh, svh, k, mode, *, tile_offset, output_dims):
    """Read one projection from grouped storage without compacting its weights.

    The TensorOps math and dispatch shape match the ordinary QMM. Other
    backends retain the established contiguous path until separately ported.
    """
    if tile_offset < 0 or output_dims <= 0 or output_dims % 128:
        raise ValueError('invalid EXL3 projection view')
    if tile_offset + output_dims // 16 > trellis.shape[1]:
        raise ValueError('EXL3 projection view exceeds parent storage')
    input_dims = trellis.shape[0] * 16
    rows = x.size // input_dims
    if x.shape[-1] != input_dims:
        raise ValueError('EXL3 projection input width mismatch')
    if rows < 24 or not (_USE_TENSOR_QMM and _tensor_ops_available()):
        return qmm_exl3(x, trellis[:, tile_offset:tile_offset + output_dims // 16], suh, svh, k, mode)
    bm, bn, bk = tensor_tile(rows, input_dims, output_dims, k, mode)
    padded_rows = ((rows + bm - 1) // bm) * bm
    xhat = _scaled_hadamard_input(x.reshape(rows, input_dims), suh)
    if padded_rows != rows:
        xhat = mx.pad(xhat, [(0, padded_rows - rows), (0, 0)])
    raw = _qmm_tensor_kernel(
        k, int(mode), input_dims, output_dims, bm, bn, bk,
        trellis.shape[1], tile_offset,
    )(
        inputs=[mx.contiguous(xhat), trellis.reshape(-1).view(mx.uint32)],
        grid=((output_dims // bn) * 32, padded_rows // bm, 1),
        threadgroup=(32, 1, 1), output_shapes=[(padded_rows, output_dims)],
        output_dtypes=[mx.float16],
    )[0][:rows]
    return _scaled_hadamard_output(raw, svh).astype(x.dtype).reshape(*x.shape[:-1], output_dims)


def qmm_exl3(
    x: mx.array,
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
    k: int,
    mode: CodebookMode | int = CodebookMode.DEFAULT,
) -> mx.array:
    """Evaluate a small token batch without reconstructing a dense weight.

    Rows are processed in groups of 2, 4, or 8. Each simdgroup decodes its
    assigned weights once and applies them to a vector of activation rows,
    amortizing trellis decode across the batch.
    """

    if x.ndim < 2:
        raise ValueError(f"QMM expects one or more rows, got {x.shape}")
    input_dims = trellis.shape[0] * 16
    output_dims = trellis.shape[1] * 16
    if x.shape[-1] != input_dims:
        raise ValueError(f"QMM expects width {input_dims}, got {x.shape[-1]}")
    rows = x.size // input_dims
    flat = x.reshape(rows, input_dims)
    cb = CodebookMode(mode)
    if trellis.dtype == mx.int16:
        trellis = trellis.view(mx.uint16)
    if rows >= 24:
        if _USE_TENSOR_QMM and _tensor_ops_available():
            block_rows, block_columns, block_depth = tensor_tile(rows, input_dims, output_dims, k, cb)
            padded_rows = ((rows + block_rows - 1) // block_rows) * block_rows
            xhat = _scaled_hadamard_input(flat, suh)
            if rows < padded_rows:
                xhat = mx.pad(xhat, [(0, padded_rows - rows), (0, 0)])
            yhat = _qmm_tensor_kernel(
                k,
                int(cb),
                input_dims,
                output_dims,
                block_rows,
                block_columns,
                block_depth,
            )(
                inputs=[mx.contiguous(xhat), trellis.reshape(-1).view(mx.uint32)],
                grid=((output_dims // block_columns) * 32, padded_rows // block_rows, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(padded_rows, output_dims)],
                output_dtypes=[mx.float16],
            )[0][:rows]
            output = _scaled_hadamard_output(yhat, svh).astype(x.dtype)
            return output.reshape(*x.shape[:-1], output_dims)
        block_rows = 64 if rows >= 48 else 32
        padded_rows = ((rows + block_rows - 1) // block_rows) * block_rows
        xhat = _scaled_hadamard_input(flat, suh)
        if rows < padded_rows:
            xhat = mx.pad(xhat, [(0, padded_rows - rows), (0, 0)])
        yhat = _qmm_matrix_kernel(k, int(cb), input_dims, output_dims, block_rows)(
            inputs=[mx.contiguous(xhat), trellis.reshape(-1).view(mx.uint32)],
            template=[
                ("K", k),
                ("CB", int(cb)),
                ("PACKED_U32", k * 8),
                ("INPUT_DIMS", input_dims),
                ("OUTPUT_DIMS", output_dims),
                ("TILES_K", trellis.shape[0]),
                ("TILES_N", trellis.shape[1]),
            ],
            grid=((output_dims // 64) * 128, padded_rows // block_rows, 1),
            threadgroup=(128, 1, 1),
            output_shapes=[(padded_rows, output_dims)],
            output_dtypes=[mx.float16],
        )[0][:rows]
        output = _scaled_hadamard_output(yhat, svh).astype(x.dtype)
        return output.reshape(*x.shape[:-1], output_dims)

    # The small-row CUDA-style tile path extracts four adjacent K-bit
    # codewords from a two-word window, which is insufficient for K=7 at a
    # handful of bit offsets. Matrix/TensorOps prefill above decodes pairs and
    # supports K=7 natively; retain the exact scalar kernel only for tiny
    # batches where building a matrix tile would be wasteful.
    if k == 7:
        output = mx.stack(
            [qmv_exl3(row, trellis, suh, svh, k, mode) for row in flat], axis=0
        )
        return output.reshape(*x.shape[:-1], output_dims)

    mt = 2 if rows <= 2 else (4 if rows <= 4 else 8)
    row_groups = (rows + mt - 1) // mt
    padded_rows = row_groups * mt
    xhat = _scaled_hadamard_input(flat, suh)
    x_transposed = xhat.transpose(1, 0)
    if rows < padded_rows:
        x_transposed = mx.pad(x_transposed, [(0, 0), (0, padded_rows - rows)])
    x_transposed = mx.contiguous(x_transposed).reshape(-1)

    output_tiles = trellis.shape[1]
    splits = 1
    while output_tiles * row_groups * splits < 8192 and trellis.shape[0] // (splits * 2) >= 64:
        splits *= 2
    partials = _qmm_tile_kernel(k, int(cb), input_dims, output_dims, mt)(
        inputs=[x_transposed, trellis.reshape(-1).view(mx.uint32)],
        template=[
            ("K", k),
            ("CB", int(cb)),
            ("PACKED_U32", k * 8),
            ("TILES_K", trellis.shape[0]),
            ("TILES_N", output_tiles),
            ("N_SPLITS", splits),
            ("ROWS", rows),
            ("PADDED_ROWS", padded_rows),
            ("OUTPUT_DIMS", output_dims),
        ],
        grid=(output_tiles * 128, row_groups, splits),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, splits, output_dims)],
        output_dtypes=[mx.float32],
    )[0]
    yhat = partials[:, 0] if splits == 1 else partials.sum(axis=1)
    output = _scaled_hadamard_output(yhat, svh).astype(x.dtype)
    return output.reshape(*x.shape[:-1], output_dims)
