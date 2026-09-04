import mlx.core as mx
import numpy as np
import pytest
import mlxl3.kernels.qmv as qmv


@pytest.mark.parametrize('input_scale', [False, True])
@pytest.mark.parametrize('width', [128, 768, 2816, 4096])
@pytest.mark.parametrize('broadcast', ['x', 'scale', 'neither'])
def test_scaled_hadamard_exact(input_scale, width, broadcast):
    x = mx.random.normal((1 if broadcast == 'x' else 8, width)).astype(mx.float16)
    scale = mx.random.uniform(.5, 1.5, (1 if broadcast == 'scale' else 8, width)).astype(mx.float16)
    actual = qmv._run_scaled_hadamard(x, scale, input_scale)
    reference = qmv._reference_scaled_hadamard_input if input_scale else qmv._reference_scaled_hadamard_output
    expected = reference(x, scale)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
