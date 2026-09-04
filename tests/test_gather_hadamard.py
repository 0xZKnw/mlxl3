import mlx.core as mx
import numpy as np
import pytest

from mlxl3.kernels import qmv


@pytest.mark.parametrize("width", [128, 768, 2048, 2816])
@pytest.mark.parametrize("repeat", [1, 2])
@pytest.mark.parametrize("input_scale", [True, False])
def test_gather_transform_exact(width, repeat, input_scale):
    mx.random.seed(15)
    indices = mx.array([5, 0, 3, 3], dtype=mx.int32)
    scale = mx.random.uniform(0.5, 1.5, (8, repeat, width)).astype(mx.float16)
    x = mx.random.normal((1 if input_scale else 4 * repeat, width)).astype(mx.float16)
    gathered = scale[indices].reshape(4 * repeat, width)
    reference = (
        qmv._reference_scaled_hadamard_input
        if input_scale
        else qmv._reference_scaled_hadamard_output
    )
    expected = reference(x, gathered)
    actual = qmv._gather_scaled_hadamard(x, scale, indices, input_scale, repeat)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.array(expected), np.array(actual))
