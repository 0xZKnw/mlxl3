"""Bit-exact CPU oracle for the procedural EXL3 codebooks.

This is a direct, readable translation of
``exllamav3_ext/quant/codebook.cuh``. Intermediate integer operations wrap to
32 bits and arithmetic values are rounded to fp16 at the same points as the
CUDA implementation.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import numpy.typing as npt


class CodebookMode(IntEnum):
    DEFAULT = 0
    MCG = 1
    MUL1 = 2


_FP16_INV = np.array([0x1EEE], dtype=np.uint16).view(np.float16)[0]
_FP16_BIAS = np.array([0xC931], dtype=np.uint16).view(np.float16)[0]


def _half_from_bits(bits: int) -> np.float16:
    return np.array([bits & 0xFFFF], dtype=np.uint16).view(np.float16)[0]


def _half2_add(bits: int) -> np.float16:
    lo = _half_from_bits(bits)
    hi = _half_from_bits(bits >> 16)
    return np.add(lo, hi, dtype=np.float16)


def decode_codeword(codeword: int, mode: CodebookMode | int) -> np.float16:
    """Decode one 16-bit trellis state to the fp16 EXL3 codebook value."""

    cb = CodebookMode(mode)
    x = int(codeword) & 0xFFFF

    if cb is CodebookMode.DEFAULT:
        x = (x * 89_226_354 + 64_248_484) & 0xFFFFFFFF
        x = 0x3B603B60 ^ (x & 0x8FFF8FFF)
        return _half2_add(x)

    if cb is CodebookMode.MCG:
        x = (x * 0xCBAC1FED) & 0xFFFFFFFF
        x = 0x3B603B60 ^ (x & 0x8FFF8FFF)
        return _half2_add(x)

    if cb is CodebookMode.MUL1:
        x = (x * 0x83DCD12D) & 0xFFFFFFFF
        byte_sum = 0x6400 + sum((x >> shift) & 0xFF for shift in (0, 8, 16, 24))
        h = _half_from_bits(byte_sum)
        # CUDA __hfma rounds only the final result to half precision.
        return np.float16(float(h) * float(_FP16_INV) + float(_FP16_BIAS))

    raise AssertionError("unreachable")


def decode_codewords(
    codewords: npt.ArrayLike,
    mode: CodebookMode | int,
) -> npt.NDArray[np.float16]:
    """Decode an array while preserving its shape."""

    words = np.asarray(codewords, dtype=np.uint16)
    flat = words.reshape(-1)
    out = np.empty(flat.shape, dtype=np.float16)
    for i, word in enumerate(flat):
        out[i] = decode_codeword(int(word), mode)
    return out.reshape(words.shape)
