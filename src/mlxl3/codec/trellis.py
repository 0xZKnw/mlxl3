"""CPU oracle for EXL3 trellis packing and tail-biting unpacking.

The layout follows ``exllamav3_ext/quant/pack.cu`` exactly: a 16x16 weight tile
is split into sixteen 16-state spans, each stored MSB first, and adjacent
uint16 words are swapped so CUDA/Metal can load the stream as native uint32.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate_k(k: int) -> None:
    if not 1 <= k <= 8:
        raise ValueError(f"K must be in [1, 8], got {k}")


def pack_trellis_tile(encoded: npt.ArrayLike, k: int) -> npt.NDArray[np.uint16]:
    """Pack the low ``k`` bits of 256 encoded states into one EXL3 tile."""

    _validate_k(k)
    values = np.asarray(encoded, dtype=np.uint16)
    if values.shape != (256,):
        raise ValueError(f"expected shape (256,), got {values.shape}")

    logical = np.zeros(16 * k, dtype=np.uint16)
    mask = (1 << k) - 1
    for span in range(16):
        base = span * 16
        for word_idx in range(k):
            word = 0
            for bit_idx in range(16):
                stream_bit = word_idx * 16 + bit_idx
                value_idx = stream_bit // k
                value_bit = k - 1 - (stream_bit % k)
                bit = (int(values[base + value_idx]) & mask) >> value_bit
                word = (word << 1) | (bit & 1)
            logical[span * k + word_idx] = word

    # SWAP16 on each native uint32, as performed by the CUDA pack kernel.
    return logical.reshape(-1, 2)[:, ::-1].reshape(-1).copy()


def unpack_trellis_tile(packed: npt.ArrayLike, k: int) -> npt.NDArray[np.uint16]:
    """Return the 256 full 16-bit sliding trellis states for one tile."""

    _validate_k(k)
    stream = np.asarray(packed, dtype=np.uint16)
    expected = 16 * k
    if stream.shape != (expected,):
        raise ValueError(f"expected shape ({expected},), got {stream.shape}")

    native_words = stream.copy().view(np.uint32)
    words_per_tile = 8 * k
    decoded = np.empty(256, dtype=np.uint16)
    for t in range(256):
        b0 = t * k + k - 16 + 256 * k
        b1 = b0 + 16
        i0 = b0 // 32
        i1 = (b1 - 1) // 32
        shift = (i1 + 1) * 32 - b1
        a = int(native_words[i0 % words_per_tile])
        b = int(native_words[i1 % words_per_tile])
        merged = (a << 32) | b
        decoded[t] = (merged >> shift) & 0xFFFF
    return decoded


def pack_trellis(encoded: npt.ArrayLike, k: int) -> npt.NDArray[np.uint16]:
    """Pack an array whose last dimension is 256."""

    _validate_k(k)
    values = np.asarray(encoded, dtype=np.uint16)
    if values.ndim == 0 or values.shape[-1] != 256:
        raise ValueError(f"expected last dimension 256, got {values.shape}")
    flat = values.reshape(-1, 256)
    out = np.empty((flat.shape[0], 16 * k), dtype=np.uint16)
    for i, tile in enumerate(flat):
        out[i] = pack_trellis_tile(tile, k)
    return out.reshape(*values.shape[:-1], 16 * k)


def unpack_trellis(packed: npt.ArrayLike, k: int) -> npt.NDArray[np.uint16]:
    """Unpack an array whose last dimension is ``16 * k``."""

    _validate_k(k)
    stream = np.asarray(packed, dtype=np.uint16)
    if stream.ndim == 0 or stream.shape[-1] != 16 * k:
        raise ValueError(f"expected last dimension {16 * k}, got {stream.shape}")
    flat = stream.reshape(-1, 16 * k)
    out = np.empty((flat.shape[0], 256), dtype=np.uint16)
    for i, tile in enumerate(flat):
        out[i] = unpack_trellis_tile(tile, k)
    return out.reshape(*stream.shape[:-1], 256)
