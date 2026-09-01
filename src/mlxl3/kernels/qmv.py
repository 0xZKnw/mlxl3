"""Correctness-first EXL3 Metal QMV for autoregressive decoding."""

from __future__ import annotations

import math
from functools import cache

import mlx.core as mx

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels.common import (
    forward_permutation_header,
    permutation_header,
    specialized_codebook_header,
)

_HADAMARD_SCALE = 1.0 / math.sqrt(128.0)


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
def _qmv_tile_kernel(k: int, mode: int, input_dims: int, output_dims: int):
    """Four-simdgroup CUDA-style 16-column tile kernel."""

    return mx.fast.metal_kernel(
        name=f"mlxl3_qmv_tile_k{k}_cb{mode}_{input_dims}x{output_dims}_v1",
        input_names=["xhat", "trellis"],
        output_names=["yhat"],
        header=specialized_codebook_header(mode) + forward_permutation_header(),
        source=r"""
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint tile_n = threadgroup_position_in_grid.x;
            uint split = threadgroup_position_in_grid.z;
            threadgroup float partials[4][256];

            uint positions[8];
            uint rows[8];
            for (uint j = 0u; j < 8u; ++j) {
                positions[j] = uint(mlxl3_perm[lane * 8u + j]);
                rows[j] = positions[j] >> 4;
            }

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

            float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            uint tiles_per_split =
                (uint(TILES_K) + uint(N_SPLITS) - 1u) / uint(N_SPLITS);
            uint tile_begin = split * tiles_per_split;
            uint tile_end = min(tile_begin + tiles_per_split, uint(TILES_K));
            for (uint tile_k = tile_begin + simd; tile_k < tile_end; tile_k += 4u) {
                const device uint* words =
                    trellis + (tile_k * uint(TILES_N) + tile_n) * uint(PACKED_U32);
                float x_lane = float(xhat[tile_k * 16u + (lane & 15u)]);
                for (uint group = 0u; group < 2u; ++group) {
                    ulong merged = (ulong(words[word0[group]]) << 32) |
                                   ulong(words[word1[group]]);
                    uint shift = shifts[group];
                    uint cw3 = uint(merged >> shift) & 0xffffu;
                    uint cw2 = uint(merged >> (shift + uint(K))) & 0xffffu;
                    uint cw1 = uint(merged >> (shift + 2u * uint(K))) & 0xffffu;
                    uint cw0 = uint(merged >> (shift + 3u * uint(K))) & 0xffffu;
                    uint j = group * 4u;
                    acc[j] = fma(
                        simd_shuffle(x_lane, ushort(rows[j])),
                        float(mlxl3_decode_codeword(cw0, CB)), acc[j]
                    );
                    acc[j + 1u] = fma(
                        simd_shuffle(x_lane, ushort(rows[j + 1u])),
                        float(mlxl3_decode_codeword(cw1, CB)), acc[j + 1u]
                    );
                    acc[j + 2u] = fma(
                        simd_shuffle(x_lane, ushort(rows[j + 2u])),
                        float(mlxl3_decode_codeword(cw2, CB)), acc[j + 2u]
                    );
                    acc[j + 3u] = fma(
                        simd_shuffle(x_lane, ushort(rows[j + 3u])),
                        float(mlxl3_decode_codeword(cw3, CB)), acc[j + 3u]
                    );
                }
            }

            for (uint j = 0u; j < 8u; ++j) {
                partials[simd][positions[j]] = acc[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 16u) {
                float sum = 0.0f;
                for (uint row = 0u; row < 16u; ++row) {
                    uint position = row * 16u + tid;
                    for (uint group = 0u; group < 4u; ++group) {
                        sum += partials[group][position];
                    }
                }
                yhat[split * uint(OUTPUT_DIMS) + tile_n * 16u + tid] = sum;
            }
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _qmv_mapped_tile_kernel(k: int, mode: int, input_dims: int, source_tiles: int):
    """Tile-cooperative QMV over an indirect list of expert output tiles."""

    return mx.fast.metal_kernel(
        name=f"mlxl3_qmv_mapped_k{k}_cb{mode}_{input_dims}x{source_tiles}_v1",
        input_names=["xhat", "trellis", "tile_map", "tile_sub"],
        output_names=["yhat"],
        header=specialized_codebook_header(mode) + forward_permutation_header(),
        source=r"""
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint local_tile = threadgroup_position_in_grid.x;
            uint tile_n = tile_map[local_tile];
            uint sub = tile_sub[local_tile];
            uint split = threadgroup_position_in_grid.z;
            threadgroup float partials[4][256];

            uint positions[8];
            uint rows[8];
            for (uint j = 0u; j < 8u; ++j) {
                positions[j] = uint(mlxl3_perm[lane * 8u + j]);
                rows[j] = positions[j] >> 4;
            }

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

            float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            uint tiles_per_split =
                (uint(TILES_K) + uint(N_SPLITS) - 1u) / uint(N_SPLITS);
            uint tile_begin = split * tiles_per_split;
            uint tile_end = min(tile_begin + tiles_per_split, uint(TILES_K));
            for (uint tile_k = tile_begin + simd; tile_k < tile_end; tile_k += 4u) {
                const device uint* words =
                    trellis + (tile_k * uint(TILES_N) + tile_n) * uint(PACKED_U32);
                float x_lane = float(
                    xhat[sub * uint(INPUT_DIMS) + tile_k * 16u + (lane & 15u)]
                );
                for (uint group = 0u; group < 2u; ++group) {
                    ulong merged = (ulong(words[word0[group]]) << 32) |
                                   ulong(words[word1[group]]);
                    uint shift = shifts[group];
                    uint cw3 = uint(merged >> shift) & 0xffffu;
                    uint cw2 = uint(merged >> (shift + uint(K))) & 0xffffu;
                    uint cw1 = uint(merged >> (shift + 2u * uint(K))) & 0xffffu;
                    uint cw0 = uint(merged >> (shift + 3u * uint(K))) & 0xffffu;
                    uint j = group * 4u;
                    acc[j] = fma(
                        simd_shuffle(x_lane, ushort(rows[j])),
                        float(mlxl3_decode_codeword(cw0, CB)), acc[j]
                    );
                    acc[j + 1u] = fma(
                        simd_shuffle(x_lane, ushort(rows[j + 1u])),
                        float(mlxl3_decode_codeword(cw1, CB)), acc[j + 1u]
                    );
                    acc[j + 2u] = fma(
                        simd_shuffle(x_lane, ushort(rows[j + 2u])),
                        float(mlxl3_decode_codeword(cw2, CB)), acc[j + 2u]
                    );
                    acc[j + 3u] = fma(
                        simd_shuffle(x_lane, ushort(rows[j + 3u])),
                        float(mlxl3_decode_codeword(cw3, CB)), acc[j + 3u]
                    );
                }
            }

            for (uint j = 0u; j < 8u; ++j) {
                partials[simd][positions[j]] = acc[j];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 16u) {
                float sum = 0.0f;
                for (uint row = 0u; row < 16u; ++row) {
                    uint position = row * 16u + tid;
                    for (uint group = 0u; group < 4u; ++group) {
                        sum += partials[group][position];
                    }
                }
                yhat[split * uint(LOCAL_OUTPUT_DIMS) + local_tile * 16u + tid] = sum;
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


def _hadamard_right(x: mx.array) -> mx.array:
    if x.shape[-1] % 128:
        raise ValueError(f"Hadamard dimension must be divisible by 128, got {x.shape[-1]}")
    shape = x.shape
    blocks = x.astype(mx.float16).reshape(*shape[:-1], shape[-1] // 128, 128)
    return mx.hadamard_transform(blocks, scale=_HADAMARD_SCALE).reshape(shape)


def _split_count(input_tiles: int, output_tiles: int) -> int:
    splits = 1
    while output_tiles * splits < 8192 and input_tiles // (splits * 2) >= 32:
        splits *= 2
    return splits


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
    xhat = _hadamard_right(x.reshape(1, input_dims).astype(mx.float32) * suh)[0]
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
        partials = _qmv_tile_kernel(k, int(cb), input_dims, output_dims)(
            inputs=[xhat, trellis.reshape(-1).view(mx.uint32)],
            template=[
                ("K", k),
                ("CB", int(cb)),
                ("PACKED_U32", k * 8),
                ("TILES_K", trellis.shape[0]),
                ("TILES_N", trellis.shape[1]),
                ("N_SPLITS", splits),
                ("OUTPUT_DIMS", output_dims),
            ],
            grid=(trellis.shape[1] * 128, 1, splits),
            threadgroup=(128, 1, 1),
            output_shapes=[(splits, output_dims)],
            output_dtypes=[mx.float32],
        )[0]
        yhat = partials[0] if splits == 1 else partials.sum(axis=0)
    output = _hadamard_right(yhat.reshape(1, output_dims))[0] * svh
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

    xhat = _hadamard_right(x.astype(mx.float32) * suh)
    tile_sub = mx.repeat(mx.arange(rows, dtype=mx.uint32), output_tiles)
    splits = _split_count(trellis.shape[0], tile_map.size)
    partials = _qmv_mapped_tile_kernel(k, int(cb), input_dims, trellis.shape[1])(
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
        ],
        grid=(tile_map.size * 128, 1, splits),
        threadgroup=(128, 1, 1),
        output_shapes=[(splits, rows, output_dims)],
        output_dtypes=[mx.float32],
    )[0]
    yhat = partials[0] if splits == 1 else partials.sum(axis=0)
    return (_hadamard_right(yhat) * svh).astype(x.dtype)


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
    if k == 7:
        output = mx.stack(
            [qmv_exl3(row, trellis, suh, svh, k, mode) for row in flat], axis=0
        )
        return output.reshape(*x.shape[:-1], output_dims)

    cb = CodebookMode(mode)
    if trellis.dtype == mx.int16:
        trellis = trellis.view(mx.uint16)
    mt = 2 if rows <= 2 else (4 if rows <= 4 else 8)
    row_groups = (rows + mt - 1) // mt
    padded_rows = row_groups * mt
    xhat = _hadamard_right(flat.astype(mx.float32) * suh)
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
    output = (_hadamard_right(yhat) * svh).astype(x.dtype)
    return output.reshape(*x.shape[:-1], output_dims)
