"""Changing the expert row tile must preserve every output bit, including tails."""
import mlx.core as mx
import numpy as np
import pytest

from mlxl3 import moe
from mlxl3.codec.trellis import pack_trellis
from mlxl3.kernels import qmv


@pytest.mark.skipif(not qmv._tensor_ops_available(), reason="M5 TensorOps required")
@pytest.mark.parametrize("bits", range(1, 9))
@pytest.mark.parametrize("mode", range(3))
def test_segmented_m64_exact(monkeypatch, bits, mode):
    # Empty expert 1, partial blocks, exact blocks and multi-block experts.
    selected = mx.array([0] * 3 + [2] * 31 + [3] * 64 + [4] * 65 + [5] * 129,
                        dtype=mx.int32)
    table, count = moe._segmented_block_table(6, selected.size // 64 + 7)(selected)
    rng = np.random.default_rng(9211 + bits)
    weights = mx.array(pack_trellis(
        rng.integers(0, 1 << bits, (16, 56, 256), dtype=np.uint16), bits))
    x = mx.array(rng.normal(size=(selected.size + 64, 256)).astype(np.float16))
    monkeypatch.setattr(qmv, "_USE_SEGMENTED_BUCKETS", False)
    monkeypatch.setattr(qmv, "_USE_SEGMENTED_LOCALITY", False)
    outputs = []
    for rows in (32, 64):
        monkeypatch.setattr(qmv, "_SEGMENTED_TENSOR_ROWS", rows)
        monkeypatch.setattr(qmv, "_USE_SEGMENTED_ADDRESS_HOIST", rows == 64)
        y = qmv.qmm_exl3_expert_segmented(
            x, weights, table, count, rows=selected.size,
            tiles_per_expert=8, output_dims=128, expert_tile_base=8,
            k=bits, mode=mode)
        mx.eval(y)
        assert mx.all(mx.isfinite(y)).item()
        outputs.append(np.asarray(y).view(np.uint16).copy())
    np.testing.assert_array_equal(*outputs)


@pytest.mark.skipif(not qmv._tensor_ops_available(), reason="M5 TensorOps required")
def test_segmented_hoist_capacity_fallback(monkeypatch):
    factory = mx.fast.metal_kernel
    def limited(**kwargs):
        if "expert_qmm_tensor_segmented" in kwargs["name"]:
            kwargs["source"] = kwargs["source"].replace(
                "ADDRESS_CAPACITY = BN * BK / 32u", "ADDRESS_CAPACITY = 1u")
            kwargs["name"] += "_capacity_one"
        return factory(**kwargs)
    qmv._segmented_expert_qmm_tensor_kernel.cache_clear()
    monkeypatch.setattr(mx.fast, "metal_kernel", limited)
    try:
        test_segmented_m64_exact(monkeypatch, 3, 1)
    finally:
        qmv._segmented_expert_qmm_tensor_kernel.cache_clear()


def test_segmented_auto_tile_and_non_template_call(monkeypatch):
    captured = {}
    def factory(*args):
        captured["tile"] = args[2]
        def run(**kwargs):
            captured.update(kwargs)
            return [mx.zeros(kwargs["output_shapes"][0], dtype=mx.float16)]
        return run
    monkeypatch.setattr(qmv, "_segmented_expert_qmm_tensor_kernel", factory)
    monkeypatch.setattr(qmv, "_tensor_ops_available", lambda: True)
    monkeypatch.setattr(qmv, "_USE_TENSOR_SEGMENTED_QMM", True)
    monkeypatch.setattr(qmv, "_USE_SEGMENTED_BUCKETS", False)
    monkeypatch.setattr(qmv, "_USE_SEGMENTED_LOCALITY", False)
    monkeypatch.setattr(qmv, "_SEGMENTED_TENSOR_ROWS", 0)
    for rows, tile in ((64, 32), (4032, 32), (4096, 64), (8192, 64)):
        qmv.qmm_exl3_expert_segmented(
            mx.zeros((rows + 64, 128), dtype=mx.float16),
            mx.zeros((8, 8, 64), dtype=mx.uint16),
            mx.zeros((3, rows // 64 + 3), dtype=mx.uint32), mx.array([1]),
            rows=rows, tiles_per_expert=8, output_dims=128, expert_tile_base=0,
            k=4, mode=1)
        assert captured["tile"] == tile
        assert captured["grid"][2] == 64 // tile
        assert captured["template"] == []
