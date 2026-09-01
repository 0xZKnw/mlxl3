"""EXL3 tensor-core tile permutation.

The encoder stores each 16x16 tile in the lane order consumed by CUDA tensor
cores. ``permutation_inverse()[row_major_index]`` returns the corresponding
encoded trellis index.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def permutation() -> npt.NDArray[np.int32]:
    out = np.empty(256, dtype=np.int32)
    for lane in range(32):
        r0 = (lane % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = lane // 4
        cols = (c0, c0 + 8)
        base = lane * 8
        for col_group, col in enumerate(cols):
            for row_offset, row in enumerate(rows):
                out[base + col_group * 4 + row_offset] = row * 16 + col
    return out


def permutation_inverse() -> npt.NDArray[np.int32]:
    direct = permutation()
    inverse = np.empty_like(direct)
    inverse[direct] = np.arange(256, dtype=np.int32)
    return inverse


def kernel_tile_to_row_major(values: npt.ArrayLike) -> npt.NDArray:
    tile = np.asarray(values)
    if tile.shape != (256,):
        raise ValueError(f"expected shape (256,), got {tile.shape}")
    return tile[permutation_inverse()].reshape(16, 16)
