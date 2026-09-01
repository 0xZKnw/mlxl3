"""Metal reconstruction kernels for EXL3 matrices."""

from __future__ import annotations

from functools import cache

import mlx.core as mx

from mlxl3.codec.codebook import CodebookMode
from mlxl3.codec.reconstruct import hadamard_128
from mlxl3.kernels.common import CODEBOOK_HEADER, permutation_header


@cache
def _inner_kernel(k: int, mode: int):
    return mx.fast.metal_kernel(
        name=f"mlxl3_reconstruct_inner_k{k}_cb{mode}",
        input_names=["trellis"],
        output_names=["weights"],
        header=CODEBOOK_HEADER + permutation_header(),
        source=r"""
            uint elem = thread_position_in_grid.x;
            uint row = elem / uint(N);
            uint col = elem - row * uint(N);
            uint tile_k = row >> 4;
            uint tile_n = col >> 4;
            uint local_rm = ((row & 15u) << 4) | (col & 15u);
            uint t = uint(mlxl3_perm_inv[local_rm]);

            uint packed_per_tile = 16u * uint(K);
            uint words_per_tile = 8u * uint(K);
            uint tile_base = (tile_k * uint(TILES_N) + tile_n) * packed_per_tile;
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
            weights[elem] = mlxl3_decode_codeword(word, CB);
        """,
        compile_options={"math_mode": "safe"},
    )


def reconstruct_inner_metal(
    trellis: mx.array,
    k: int,
    mode: CodebookMode | int = CodebookMode.DEFAULT,
) -> mx.array:
    """Decode ``W_hat`` directly into a row-major fp16 MLX array."""

    cb = CodebookMode(mode)
    if not 1 <= k <= 8:
        raise ValueError(f"K must be in [1, 8], got {k}")
    if trellis.dtype == mx.int16:
        trellis = trellis.view(mx.uint16)
    if trellis.dtype != mx.uint16 or trellis.ndim != 3 or trellis.shape[-1] != 16 * k:
        raise ValueError(f"expected uint16 [tiles_k, tiles_n, {16 * k}], got {trellis}")
    tiles_k, tiles_n, _ = trellis.shape
    rows, cols = tiles_k * 16, tiles_n * 16
    return _inner_kernel(k, int(cb))(
        inputs=[trellis],
        template=[("K", k), ("CB", int(cb)), ("N", cols), ("TILES_N", tiles_n)],
        grid=(rows * cols, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(rows, cols)],
        output_dtypes=[mx.float16],
    )[0]


def reconstruct_public_weights_mlx(
    trellis: mx.array,
    suh: mx.array,
    svh: mx.array,
    k: int,
    mode: CodebookMode | int = CodebookMode.DEFAULT,
) -> mx.array:
    """Reference MLX reconstruction in the original basis.

    The trellis decode runs as a custom Metal kernel. Hadamard transforms use
    MLX matmul for now; the optimized path will fuse them into QMV/GEMM.
    """

    weights = reconstruct_inner_metal(trellis, k, mode).astype(mx.float32)
    rows, cols = weights.shape
    if rows % 128 or cols % 128:
        raise ValueError(f"EXL3 public reconstruction needs 128-aligned dims, got {weights.shape}")
    had = mx.array(hadamard_128())
    weights = mx.concatenate(
        [had @ weights[row : row + 128] for row in range(0, rows, 128)], axis=0
    )
    weights = weights * suh.astype(mx.float32)[:, None]
    weights = mx.concatenate(
        [weights[:, col : col + 128] @ had for col in range(0, cols, 128)], axis=1
    )
    return (weights * svh.astype(mx.float32)[None, :]).astype(mx.float16)
