import numpy as np
import pytest


@pytest.mark.parametrize("cb", [0, 1, 2])
@pytest.mark.parametrize("budget", [1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024])
def test_compact_k4_preserves_launch_partition_and_explicit_budget(monkeypatch, cb, budget):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("ponyexl3.convert.metal_search")
    from mlxl3_quantizer import metal

    counts = []

    def kernel(**options):
        count = options["inputs"][0].shape[0]
        counts.append(count)
        scratch_bytes = options["output_shapes"][2][1] * count * options["output_dtypes"][2].size
        # One tile is the documented minimum when the requested budget is smaller.
        assert scratch_bytes <= max(budget, scratch_bytes // count)
        return [mx.array(np.asarray(options["inputs"][0])), mx.zeros((count, 256), mx.uint16)]

    monkeypatch.setattr(metal, "_paired_kernel", lambda k, mode: (kernel, ()))
    values = np.random.default_rng(13).normal(size=(257, 256)).astype(np.float32)
    quantized, indices = metal.quantize_tiles_mlx(values, 4, cb, max_scratch_bytes=budget)
    expected_batch = min(128, max(1, budget // ((2 if cb == 2 else 1) * 1024 * 1024)))
    assert counts == [min(expected_batch, 257 - start) for start in range(0, 257, expected_batch)]
    np.testing.assert_array_equal(np.asarray(quantized), values)
    assert indices.shape == values.shape and indices.size > 0
