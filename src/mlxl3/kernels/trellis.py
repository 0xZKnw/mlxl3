"""JIT-compiled Metal kernels for EXL3 packing and fused unpack/decode."""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels.common import CODEBOOK_HEADER


@cache
def _decode_kernel(k: int, mode: int):
    # K and codebook are included in the name as well as template parameters.
    # This avoids collisions in MLX's in-process Metal library cache.
    return mx.fast.metal_kernel(
        name=f"mlxl3_decode_trellis_k{k}_cb{mode}",
        input_names=["packed"],
        output_names=["codewords", "values"],
        header=CODEBOOK_HEADER,
        source=r"""
            uint elem = thread_position_in_grid.x;
            uint tile = elem >> 8;
            uint t = elem & 255u;
            uint words_per_tile = 8u * uint(K);
            uint packed_per_tile = 16u * uint(K);
            uint tile_base = tile * packed_per_tile;

            uint b0 = t * uint(K) + uint(K) - 16u + 256u * uint(K);
            uint b1 = b0 + 16u;
            uint i0 = b0 >> 5;
            uint i1 = (b1 - 1u) >> 5;
            uint shift = ((i1 + 1u) << 5) - b1;

            uint wi0 = i0 % words_per_tile;
            uint wi1 = i1 % words_per_tile;
            uint a = uint(packed[tile_base + 2u * wi0]) |
                     (uint(packed[tile_base + 2u * wi0 + 1u]) << 16);
            uint b = uint(packed[tile_base + 2u * wi1]) |
                     (uint(packed[tile_base + 2u * wi1 + 1u]) << 16);
            ulong merged = (ulong(a) << 32) | ulong(b);
            ushort word = ushort((merged >> shift) & 0xfffful);
            codewords[elem] = word;
            values[elem] = mlxl3_decode_codeword(uint(word), CB);
        """,
        compile_options={"math_mode": "safe"},
    )


@cache
def _pack_kernel(k: int):
    return mx.fast.metal_kernel(
        name=f"mlxl3_pack_trellis_k{k}",
        input_names=["encoded"],
        output_names=["packed"],
        source=r"""
            uint elem = thread_position_in_grid.x;
            uint packed_per_tile = 16u * uint(K);
            uint tile = elem / packed_per_tile;
            uint physical = elem - tile * packed_per_tile;
            uint logical = physical ^ 1u;
            uint span = logical / uint(K);
            uint word_idx = logical - span * uint(K);
            uint encoded_base = tile * 256u + span * 16u;
            ushort word = 0;
            for (uint bit_idx = 0; bit_idx < 16u; ++bit_idx) {
                uint stream_bit = word_idx * 16u + bit_idx;
                uint value_idx = stream_bit / uint(K);
                uint value_bit = uint(K) - 1u - (stream_bit % uint(K));
                ushort bit = (encoded[encoded_base + value_idx] >> value_bit) & 1u;
                word = ushort((word << 1) | bit);
            }
            packed[elem] = word;
        """,
        compile_options={"math_mode": "safe"},
    )


def _validate_k(k: int) -> None:
    if not 1 <= k <= 8:
        raise ValueError(f"K must be in [1, 8], got {k}")


def pack_trellis_metal(encoded: mx.array, k: int) -> mx.array:
    """Pack low-K trellis decisions on the GPU."""

    _validate_k(k)
    if encoded.dtype != mx.uint16 or encoded.ndim == 0 or encoded.shape[-1] != 256:
        raise ValueError("encoded must be a uint16 MLX array with last dimension 256")
    output_shape = (*encoded.shape[:-1], 16 * k)
    total = encoded.size // 256 * 16 * k
    return _pack_kernel(k)(
        inputs=[encoded],
        template=[("K", k)],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[mx.uint16],
    )[0]


def decode_trellis_metal(
    packed: mx.array,
    k: int,
    mode: CodebookMode | int,
) -> tuple[mx.array, mx.array]:
    """Fused unpack + procedural-codebook decode on the GPU."""

    _validate_k(k)
    cb = CodebookMode(mode)
    if packed.dtype != mx.uint16 or packed.ndim == 0 or packed.shape[-1] != 16 * k:
        raise ValueError(f"packed must be a uint16 MLX array with last dimension {16 * k}")
    output_shape = (*packed.shape[:-1], 256)
    total = packed.size // (16 * k) * 256
    codewords, values = _decode_kernel(k, int(cb))(
        inputs=[packed],
        template=[("K", k), ("CB", int(cb))],
        grid=(total, 1, 1),
        threadgroup=(min(256, total), 1, 1),
        output_shapes=[output_shape, output_shape],
        output_dtypes=[mx.uint16, mx.float16],
    )
    return codewords, values
