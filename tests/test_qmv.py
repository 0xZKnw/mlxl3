from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from mlxl3.codec.codebook import CodebookMode
from mlxl3.codec.perm import permutation
from mlxl3.codec.trellis import pack_trellis
from mlxl3.kernels.qmv import (
    _qmv_tiles_per_group,
    _split_count,
    qmm_exl3,
    qmv_exl3,
)
from mlxl3.kernels.reconstruct import reconstruct_public_weights_mlx
from mlxl3.linear import EXL3Linear, EXL3ProjectionProxy, fuse_compatible_linear_groups


def test_exl3_permutation_supports_quadgroup_column_reduction() -> None:
    positions = permutation().reshape(32, 8)
    for lane in range(32):
        row_base = 2 * (lane & 3)
        expected_rows = (row_base, row_base + 1, row_base + 8, row_base + 9)
        expected_column = lane >> 2
        np.testing.assert_array_equal(
            positions[lane] // 16,
            (*expected_rows, *expected_rows),
        )
        np.testing.assert_array_equal(
            positions[lane] % 16,
            (expected_column,) * 4 + (expected_column + 8,) * 4,
        )


@pytest.mark.parametrize("k", (1, 2, 3, 4, 5, 6, 8))
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


@pytest.mark.parametrize("rows", (5, 32, 64))
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
        (224, 64, 4),
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


@pytest.mark.parametrize(
    ("output_tiles", "expected"),
    ((15, 1), (14, 2), (128, 4), (448, 4), (1024, 2), (8000, 2)),
)
def test_qmv_output_group_balances_reuse_and_occupancy(
    output_tiles: int, expected: int
) -> None:
    assert (
        _qmv_tiles_per_group(128, output_tiles, 3, CodebookMode.MCG) == expected
    )


def test_compatible_projection_group_matches_individual_decode_and_batch() -> None:
    rng = np.random.default_rng(47000)
    widths = (128, 256, 128)
    linears = []
    for width in widths:
        encoded = rng.integers(0, 1 << 3, size=(8, width // 16, 256), dtype=np.uint16)
        linears.append(
            EXL3Linear(
                mx.array(pack_trellis(encoded, 3)),
                mx.array(rng.uniform(0.7, 1.3, size=128).astype(np.float16)),
                mx.array(rng.uniform(0.7, 1.3, size=width).astype(np.float16)),
                bits=3,
                mode=CodebookMode.MCG,
            )
        )

    class Projections(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj = linears

    projections = Projections()
    decode_x = mx.array(rng.normal(size=(1, 128)).astype(np.float16))
    batch_x = mx.array(rng.normal(size=(3, 128)).astype(np.float16))
    expected_decode = tuple(linear(decode_x) for linear in linears)
    expected_batch = tuple(linear(batch_x) for linear in linears)

    assert fuse_compatible_linear_groups(projections) == 1
    assert isinstance(projections.q_proj, EXL3ProjectionProxy)
    actual_decode = tuple(
        projection(decode_x)
        for projection in (projections.q_proj, projections.k_proj, projections.v_proj)
    )
    actual_batch = tuple(
        projection(batch_x)
        for projection in (projections.q_proj, projections.k_proj, projections.v_proj)
    )
    mx.eval(*expected_decode, *actual_decode, *expected_batch, *actual_batch)
    for expected, actual in zip(expected_decode, actual_decode, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for expected, actual in zip(expected_batch, actual_batch, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    class GatedDeltaInputs(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj_qkv, self.in_proj_z = linears[:2]

    gated_delta = GatedDeltaInputs()
    assert fuse_compatible_linear_groups(gated_delta) == 1
    assert isinstance(gated_delta.in_proj_qkv, EXL3ProjectionProxy)
    actual_delta = (
        gated_delta.in_proj_qkv(decode_x),
        gated_delta.in_proj_z(decode_x),
    )
    mx.eval(*actual_delta)
    for expected, actual in zip(expected_decode[:2], actual_delta, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
