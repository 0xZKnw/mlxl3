from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlxl3.codec.codebook import CodebookMode
from mlxl3.codec.trellis import pack_trellis
from mlxl3.kernels.qmv import _split_count, qmm_exl3, qmv_exl3
from mlxl3.kernels.reconstruct import reconstruct_public_weights_mlx


@pytest.mark.parametrize("k", (2, 3, 4))
@pytest.mark.parametrize("mode", list(CodebookMode))
def test_qmv_matches_reconstructed_weight(k: int, mode: CodebookMode) -> None:
    rng = np.random.default_rng(44000 + 10 * k + int(mode))
    encoded = rng.integers(0, 1 << k, size=(8, 8, 256), dtype=np.uint16)
    trellis = mx.array(pack_trellis(encoded, k))
    suh = mx.array(rng.uniform(0.7, 1.3, size=128).astype(np.float16))
    svh = mx.array(rng.uniform(0.7, 1.3, size=128).astype(np.float16))
    x = mx.array(rng.normal(size=(1, 1, 128)).astype(np.float16))

    dense = reconstruct_public_weights_mlx(trellis, suh, svh, k, mode)
    expected = x @ dense
    actual = qmv_exl3(x, trellis, suh, svh, k, mode)
    mx.eval(expected, actual)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=0.035, rtol=0.005)


@pytest.mark.parametrize("rows", (2, 4, 7, 9, 64))
def test_qmm_matches_individual_qmv(rows: int) -> None:
    rng = np.random.default_rng(45000 + rows)
    k = 3
    encoded = rng.integers(0, 1 << k, size=(8, 8, 256), dtype=np.uint16)
    trellis = mx.array(pack_trellis(encoded, k))
    suh = mx.array(rng.uniform(0.7, 1.3, size=128).astype(np.float16))
    svh = mx.array(rng.uniform(0.7, 1.3, size=128).astype(np.float16))
    x = mx.array(rng.normal(size=(rows, 128)).astype(np.float16))

    expected = mx.stack(
        [qmv_exl3(x[row], trellis, suh, svh, k, CodebookMode.MCG) for row in range(rows)]
    )
    actual = qmm_exl3(x, trellis, suh, svh, k, CodebookMode.MCG)
    mx.eval(expected, actual)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=0.035, rtol=0.005)


@pytest.mark.parametrize("rows", (5, 64))
@pytest.mark.parametrize("k", (1, 2, 3, 4, 5, 6, 8))
def test_qmm_supports_code_widths(rows: int, k: int) -> None:
    rng = np.random.default_rng(46000 + 100 * rows + k)
    encoded = rng.integers(0, 1 << k, size=(8, 8, 256), dtype=np.uint16)
    trellis = mx.array(pack_trellis(encoded, k))
    suh = mx.array(rng.uniform(0.7, 1.3, size=128).astype(np.float16))
    svh = mx.array(rng.uniform(0.7, 1.3, size=128).astype(np.float16))
    x = mx.array(rng.normal(size=(rows, 128)).astype(np.float16))

    expected = mx.stack(
        [qmv_exl3(x[row], trellis, suh, svh, k, CodebookMode.DEFAULT) for row in range(rows)]
    )
    actual = qmm_exl3(x, trellis, suh, svh, k, CodebookMode.DEFAULT)
    mx.eval(expected, actual)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=0.035, rtol=0.005)


@pytest.mark.parametrize(
    ("input_tiles", "output_tiles", "expected"),
    (
        (64, 384, 1),
        (128, 128, 1),
        (224, 64, 2),
        (256, 8, 8),
        (512, 128, 4),
        (1024, 256, 4),
        (128, 8000, 1),
    ),
)
def test_split_count_balances_width_and_input_work(
    input_tiles: int, output_tiles: int, expected: int
) -> None:
    assert _split_count(input_tiles, output_tiles) == expected
