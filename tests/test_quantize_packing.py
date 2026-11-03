import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(
    os.environ.get("MLXL3_SKIP_METAL_CHECK") == "1", reason="requires a physical Apple GPU"
)


@pytest.mark.parametrize("k", range(1, 9))
def test_packing_matches_independent_cpu_bitstream(k):
    from ponyexl3.ref.trellis import pack_trellis_tile

    from mlxl3_quantizer import pack_trellis_mlx

    # Every low-bit position in a tile, plus arbitrary high bits and all-ones.
    states = np.zeros((256 * k + 3, 256), dtype=np.uint16)
    for position in range(256):
        for bit in range(k):
            states[position * k + bit, position] = 1 << bit
    states[-2] = np.random.default_rng(41 + k).integers(0, 65536, 256, dtype=np.uint16)
    states[-1] = 65535
    expected = np.stack([pack_trellis_tile(row, k) for row in states])
    actual = np.asarray(pack_trellis_mlx(states, k))
    assert actual.shape == expected.shape and actual.size > 0
    assert actual.dtype == np.uint16
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8])
def test_packing_noncontiguous_multidimensional_and_empty(k):
    from ponyexl3.ref.trellis import pack_trellis

    from mlxl3_quantizer import pack_trellis_mlx

    data = mx.arange(3 * 2 * 256).reshape(3, 2, 256).transpose(1, 0, 2)
    np.testing.assert_array_equal(
        np.asarray(pack_trellis_mlx(data, k)), pack_trellis(np.asarray(data), k)
    )
    assert pack_trellis_mlx(mx.zeros((0, 2, 256)), k).shape == (0, 2, 16 * k)
    assert pack_trellis_mlx(mx.zeros((256,)), k).shape == (16 * k,)


@pytest.mark.parametrize("shape", [(), (255,), (2, 257), (256, 2)])
def test_packing_rejects_wrong_shape(shape):
    from mlxl3_quantizer import pack_trellis_mlx

    with pytest.raises(ValueError, match="last dim 256"):
        pack_trellis_mlx(mx.zeros(shape), 4)


@pytest.mark.parametrize("k", [0, 9, -1, 4.5])
def test_packing_rejects_wrong_k(k):
    from mlxl3_quantizer import pack_trellis_mlx

    with pytest.raises((ValueError, TypeError)):
        pack_trellis_mlx(mx.zeros((256,)), k)


def test_actual_ldlq_path_keeps_packed_weights_and_reconstruction(monkeypatch):
    from ponyexl3.convert import metal_search, mlx_trellis
    from ponyexl3.convert.hessian import ldlq_inner_matrix
    from ponyexl3.ref.codebook import CodebookMode

    from mlxl3_quantizer import pack_trellis_mlx, quantize_tiles_mlx

    monkeypatch.setattr(metal_search, "quantize_tiles_mlx", quantize_tiles_mlx)
    rng = np.random.default_rng(47)
    weight = rng.normal(size=(128, 128)).astype(np.float32)
    x = rng.normal(size=(192, 128)).astype(np.float32)
    hessian = x.T @ x / len(x) + np.eye(128, dtype=np.float32) * 0.1
    options = {"k": 4, "cb": CodebookMode.MCG, "search_backend": "metal", "collect_states": False}
    baseline = ldlq_inner_matrix(weight, hessian, **options)
    monkeypatch.setattr(mlx_trellis, "pack_trellis_mlx", pack_trellis_mlx)
    candidate = ldlq_inner_matrix(weight, hessian, **options)
    assert baseline.packed.size > 0 and np.isfinite(candidate.reconstructed).all()
    np.testing.assert_array_equal(candidate.packed, baseline.packed)
    np.testing.assert_array_equal(candidate.reconstructed, baseline.reconstructed)
