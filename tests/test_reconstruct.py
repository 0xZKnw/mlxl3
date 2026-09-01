from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxl3.codec.codebook import CodebookMode
from mlxl3.codec.reconstruct import reconstruct_inner, reconstruct_public_weights
from mlxl3.codec.trellis import pack_trellis
from mlxl3.kernels.reconstruct import reconstruct_inner_metal, reconstruct_public_weights_mlx


@pytest.mark.parametrize("k", (2, 3, 4))
@pytest.mark.parametrize("mode", list(CodebookMode))
def test_inner_reconstruction_is_bit_exact(k: int, mode: CodebookMode) -> None:
    rng = np.random.default_rng(12000 + 10 * k + int(mode))
    encoded = rng.integers(0, 1 << k, size=(8, 8, 256), dtype=np.uint16)
    trellis = pack_trellis(encoded, k)
    expected = reconstruct_inner(trellis, k, mode)
    actual = reconstruct_inner_metal(mx.array(trellis), k, mode)
    mx.eval(actual)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint16), expected.view(np.uint16))


def test_public_reconstruction_matches_cpu_oracle() -> None:
    rng = np.random.default_rng(31337)
    encoded = rng.integers(0, 8, size=(8, 8, 256), dtype=np.uint16)
    trellis = pack_trellis(encoded, 3)
    suh = rng.choice([-1.0, 1.0], size=128).astype(np.float16)
    svh = rng.choice([-1.0, 1.0], size=128).astype(np.float16)
    expected = reconstruct_public_weights(trellis, suh, svh, 3)
    actual = reconstruct_public_weights_mlx(mx.array(trellis), mx.array(suh), mx.array(svh), 3)
    mx.eval(actual)
    np.testing.assert_allclose(np.asarray(actual), expected, atol=2**-7, rtol=2**-7)
