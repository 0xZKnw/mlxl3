import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    os.environ.get("MLXL3_SKIP_METAL_CHECK") == "1", reason="physical Apple GPU required"
)


@pytest.mark.parametrize("cb", [0, 1, 2])
@pytest.mark.parametrize("special", [False, True])
def test_k4_special_inputs_match_reference(cb, special):
    reference = pytest.importorskip("ponyexl3.convert.metal_search").quantize_tiles_mlx

    from mlxl3_quantizer import quantize_tiles_mlx

    values = np.random.default_rng(173).normal(size=(7, 256)).astype(np.float32)
    if special:
        values[0] = 0
        values[1] = np.inf
        values[2] = -np.inf
        values[3] = np.nan
        values[4] = 1e20
        values[5] = -1e20
    expected_q, expected_indices = reference(values, 4, cb)
    actual_q, actual_indices = quantize_tiles_mlx(values, 4, cb)
    assert actual_q.shape == actual_indices.shape == values.shape and actual_q.size > 0
    np.testing.assert_array_equal(np.asarray(actual_q), np.asarray(expected_q))
    np.testing.assert_array_equal(np.asarray(actual_indices), np.asarray(expected_indices))


@pytest.mark.parametrize("count", [255, 256, 257])
def test_k4_chunk_boundaries_keep_states_and_reconstruction(count):
    reference = pytest.importorskip("ponyexl3.convert.metal_search").quantize_tiles_mlx

    from mlxl3_quantizer import quantize_tiles_mlx

    values = np.random.default_rng(191).normal(size=(count, 256)).astype(np.float32)
    expected_q, expected_indices = reference(values, 4, 1)
    actual_q, actual_indices = quantize_tiles_mlx(values, 4, 1)
    np.testing.assert_array_equal(np.asarray(actual_q), np.asarray(expected_q))
    np.testing.assert_array_equal(np.asarray(actual_indices), np.asarray(expected_indices))
