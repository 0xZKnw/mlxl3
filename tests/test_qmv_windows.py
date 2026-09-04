import mlx.core as mx
import numpy as np
import pytest

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels import qmv


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("feature", ["_USE_K24_WINDOW_DECODE", "_USE_CODEBOOK_LUT"])
@pytest.mark.parametrize("mode", list(CodebookMode))
@pytest.mark.parametrize("shape", [(128, 128), (2048, 128), (512, 2048)])
def test_dense_window_bit_exact(monkeypatch, bits, mode, shape, feature):
    inputs, outputs = shape
    rng = np.random.default_rng(20 + bits)
    packed = mx.array(
        rng.integers(0, 65536, (inputs // 16, outputs // 16, bits * 16), dtype=np.uint16)
    )
    x = mx.array(rng.normal(size=inputs).astype(np.float16))
    su = mx.array(rng.uniform(0.5, 1.5, inputs).astype(np.float16))
    sv = mx.array(rng.uniform(0.5, 1.5, outputs).astype(np.float16))
    monkeypatch.setattr(qmv, feature, False)
    expected = qmv.qmv_exl3(x, packed, su, sv, bits, mode)
    mx.eval(expected)
    monkeypatch.setattr(qmv, feature, True)
    actual = qmv.qmv_exl3(x, packed, su, sv, bits, mode)
    mx.eval(actual)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("feature", ["_USE_K24_WINDOW_DECODE", "_USE_CODEBOOK_LUT"])
@pytest.mark.parametrize("mode", list(CodebookMode))
def test_grouped_window_bit_exact(monkeypatch, bits, mode, feature):
    rng = np.random.default_rng(40)
    packed = mx.array(rng.integers(0, 65536, (32, 48, bits * 16), dtype=np.uint16))
    x = mx.array(rng.normal(size=512).astype(np.float16))
    su = mx.array(rng.uniform(0.5, 1.5, (2, 512)).astype(np.float16))
    sv = mx.array(rng.uniform(0.5, 1.5, 768).astype(np.float16))
    sub = mx.array([0] * 16 + [1] * 32, dtype=mx.uint32)

    def run():
        return qmv.qmv_exl3_grouped(
            x, packed, su, sv, sub, output_dims=(256, 512), k=bits, mode=mode
        )

    monkeypatch.setattr(qmv, feature, False)
    expected = run()
    mx.eval(expected)
    monkeypatch.setattr(qmv, feature, True)
    actual = run()
    mx.eval(actual)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(np.array(a), np.array(b))
