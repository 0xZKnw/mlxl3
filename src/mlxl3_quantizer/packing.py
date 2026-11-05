"""Bit-exact EXL3 packing without intermediate MLX masks, stacks or reductions."""

from functools import lru_cache
from operator import index
from pathlib import Path


@lru_cache(maxsize=1)
def _kernel():
    import mlx.core as mx

    return mx.fast.metal_kernel(
        name="mlxl3_pack_trellis",
        input_names=["encoded", "word_count"],
        output_names=["packed"],
        header=Path(__file__).with_name("pack_trellis.metal").read_text(),
        source="""
            uint word = thread_position_in_grid.x;
            if (word < word_count) packed[word] = mlxl3_pack_word(encoded, K, word);
        """,
    )


def pack_trellis_mlx(encoded, k: int):
    """Pack low K bits of (..., 256) uint16 states, preserving EXL3 SWAP16."""
    import mlx.core as mx

    k = index(k)
    if not 1 <= k <= 8:
        raise ValueError(f"K must be in [1, 8], got {k}")
    arr = mx.array(encoded, dtype=mx.uint16)
    if arr.ndim < 1 or arr.shape[-1] != 256:
        raise ValueError(f"expected encoded last dim 256, got {arr.shape}")
    if arr.size >= 1 << 32:
        raise ValueError("Metal packing requires fewer than 2**32 input values")
    shape = (*arr.shape[:-1], 16 * k)
    words = arr.size // 256 * 16 * k
    if words == 0:
        return mx.zeros(shape, dtype=mx.uint16)
    return _kernel()(
        inputs=[arr, mx.array(words, dtype=mx.uint32)],
        template=[("K", k)],
        grid=(words, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.uint16],
    )[0]
