import mlx.core as mx
import numpy as np
import pytest

from mlxl3.kernels.qmv import _qmm_tensor_kernel, _tensor_ops_available
from mlxl3.tuning import tensor_tile


def test_dense_prefill_dispatch_does_not_pad_small_or_ragged_prompts(monkeypatch):
    monkeypatch.delenv('MLXL3_TENSOR_PROFILE', raising=False)
    monkeypatch.setenv('MLXL3_DENSE_PREFILL_M64', '1')
    assert tensor_tile(256, 5120, 17408, 3, 2) == (64,32,16)
    for rows, ni, no, mode in [(69,5120,17408,2), (32,5120,17408,2),
                               (256,2048,4096,2), (256,5120,248320,2),
                               (256,5120,17408,1)]:
        assert tensor_tile(rows,ni,no,3,mode) == (32,32,16)
    monkeypatch.setenv('MLXL3_DENSE_PREFILL_M64', '0')
    assert tensor_tile(256,5120,17408,3,2) == (32,32,16)


@pytest.mark.skipif(not _tensor_ops_available(), reason='M5 TensorOps required')
@pytest.mark.parametrize('bits', [2,3,6])
def test_tensor_m64_preserves_m32_outputs(bits):
    rng = np.random.default_rng(8127+bits)
    x = mx.array(rng.normal(size=(128,256)).astype(np.float16))
    t = mx.array(rng.integers(0,2**32,size=(16*16*bits*8,),dtype=np.uint32))
    outputs = []
    for bm in (32,64):
        out = _qmm_tensor_kernel(bits,2,256,256,bm,32,16)(
            inputs=[x,t], grid=(8*32,128//bm,1), threadgroup=(32,1,1),
            output_shapes=[(128,256)], output_dtypes=[mx.float16],
        )[0]
        mx.eval(out)
        outputs.append(np.asarray(out).view(np.uint16))
    np.testing.assert_array_equal(*outputs)
