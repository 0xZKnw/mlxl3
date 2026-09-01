"""Readable CPU reconstruction of public EXL3 weights."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .codebook import CodebookMode, decode_codewords
from .perm import kernel_tile_to_row_major
from .trellis import unpack_trellis_tile


def hadamard_128() -> npt.NDArray[np.float32]:
    """Return the normalized Sylvester Hadamard matrix used by EXL3."""

    matrix = np.ones((1, 1), dtype=np.float32)
    while matrix.shape[0] < 128:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix * np.float32(1.0 / math.sqrt(128.0))


def reconstruct_inner(
    trellis: npt.ArrayLike,
    k: int,
    mode: CodebookMode | int = CodebookMode.DEFAULT,
) -> npt.NDArray[np.float16]:
    """Decode the tensor-core/Hadamard-basis matrix ``W_hat``."""

    packed = np.asarray(trellis, dtype=np.uint16)
    if packed.ndim != 3 or packed.shape[-1] != 16 * k:
        raise ValueError(f"expected [tiles_k, tiles_n, {16 * k}], got {packed.shape}")
    tiles_k, tiles_n, _ = packed.shape
    output = np.empty((tiles_k * 16, tiles_n * 16), dtype=np.float16)
    for tile_k in range(tiles_k):
        for tile_n in range(tiles_n):
            states = unpack_trellis_tile(packed[tile_k, tile_n], k)
            decoded = decode_codewords(states, mode)
            output[
                tile_k * 16 : (tile_k + 1) * 16,
                tile_n * 16 : (tile_n + 1) * 16,
            ] = kernel_tile_to_row_major(decoded)
    return output


def _expand_scales(values: npt.ArrayLike, expected: int) -> npt.NDArray[np.float16]:
    scales = np.asarray(values)
    if scales.dtype.kind in "iu" and scales.size * 16 == expected:
        words = scales.astype(np.uint16, copy=False).reshape(-1, 1)
        masks = (np.uint16(1) << np.arange(16, dtype=np.uint16)).reshape(1, 16)
        return np.where(words & masks, -1.0, 1.0).astype(np.float16).reshape(-1)
    if scales.size != expected:
        raise ValueError(f"expected {expected} scales, got {scales.size}")
    return scales.astype(np.float16, copy=False).reshape(-1)


def reconstruct_public_weights(
    trellis: npt.ArrayLike,
    suh: npt.ArrayLike,
    svh: npt.ArrayLike,
    k: int,
    mode: CodebookMode | int = CodebookMode.DEFAULT,
) -> npt.NDArray[np.float16]:
    """Reconstruct ``diag(suh) H W_hat H diag(svh)`` in the public basis."""

    inner = reconstruct_inner(trellis, k, mode).astype(np.float32)
    rows, cols = inner.shape
    if rows % 128 or cols % 128:
        raise ValueError(f"EXL3 public reconstruction needs 128-aligned dims, got {inner.shape}")
    had = hadamard_128()
    for row in range(0, rows, 128):
        inner[row : row + 128] = had @ inner[row : row + 128]
    row_scales = _expand_scales(suh, rows).astype(np.float32)
    inner *= row_scales[:, None]
    for col in range(0, cols, 128):
        inner[:, col : col + 128] = inner[:, col : col + 128] @ had
    col_scales = _expand_scales(svh, cols).astype(np.float32)
    inner *= col_scales[None, :]
    return inner.astype(np.float16)
