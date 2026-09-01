from __future__ import annotations

import numpy as np
import pytest

from mlxl3.codec.codebook import CodebookMode, decode_codewords
from mlxl3.codec.trellis import pack_trellis, pack_trellis_tile, unpack_trellis


@pytest.mark.parametrize("k", range(1, 9))
def test_pack_layout_and_tail_biting_roundtrip(k: int) -> None:
    rng = np.random.default_rng(0xE1_3 + k)
    encoded = rng.integers(0, 1 << k, size=(2, 3, 256), dtype=np.uint16)
    packed = pack_trellis(encoded, k)
    decoded = unpack_trellis(packed, k)

    assert packed.shape == (2, 3, 16 * k)
    np.testing.assert_array_equal(decoded & ((1 << k) - 1), encoded)

    # Independently replay two complete tail-biting rings. The second pass is
    # the fixed point represented by the cyclic EXL3 bitstream.
    expected = np.empty_like(decoded)
    state = np.zeros(encoded.shape[:-1], dtype=np.uint32)
    for t in list(range(256)) * 2:
        state = ((state << k) | encoded[..., t]) & 0xFFFF
        expected[..., t] = state
    np.testing.assert_array_equal(decoded, expected)


def test_pack_uses_cuda_uint16_pair_swap() -> None:
    encoded = np.zeros(256, dtype=np.uint16)
    encoded[:16] = np.arange(16, dtype=np.uint16)
    packed = pack_trellis_tile(encoded, 4)
    # Logical MSB-first words are 0x0123, 0x4567, 0x89ab, 0xcdef;
    # CUDA's SWAP16 stores each adjacent pair in reverse order.
    np.testing.assert_array_equal(packed[:4], [0x4567, 0x0123, 0xCDEF, 0x89AB])


@pytest.mark.parametrize("mode", list(CodebookMode))
def test_codebook_is_finite_for_all_states(mode: CodebookMode) -> None:
    words = np.arange(65536, dtype=np.uint16)
    values = decode_codewords(words, mode)
    assert values.dtype == np.float16
    assert np.isfinite(values).all()
