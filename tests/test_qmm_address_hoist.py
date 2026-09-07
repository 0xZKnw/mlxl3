"""Exact A/B checks for address hoisting, without changing TensorOps math/tiles."""

import mlx.core as mx
import numpy as np
import pytest

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels import qmv


@pytest.fixture(autouse=True)
def clear_factory_cache():
    qmv._qmm_tensor_kernel.cache_clear()
    yield
    qmv._qmm_tensor_kernel.cache_clear()


@pytest.fixture
def tensor_ops():
    if not qmv._tensor_ops_available():
        pytest.skip('M5 TensorOps required')


def exact_pair(monkeypatch, call):
    outputs = []
    for enabled in (False, True):
        monkeypatch.setattr(qmv, '_USE_QMM_ADDRESS_HOIST', enabled)
        # The configuration global is intentionally fixed per production process.
        qmv._qmm_tensor_kernel.cache_clear()
        value = call()
        mx.eval(value)
        assert mx.all(mx.isfinite(value)).item()
        outputs.append(np.asarray(mx.contiguous(value).view(mx.uint8)).copy())
    np.testing.assert_array_equal(*outputs)


def raw_case(monkeypatch, *, bits=4, mode=CodebookMode.MCG, bm=32, bn=32, bk=16,
             input_dims=128, output_dims=128, offset=0, parent_tiles=None):
    rng = np.random.default_rng(20391 + bits)
    rows = 2 * bm
    parent_tiles = parent_tiles or output_dims // 16
    x = mx.array(rng.normal(0, .02, (rows, input_dims)).astype(np.float16))
    trellis = mx.array(rng.integers(
        0, 2**32, size=input_dims // 16 * parent_tiles * bits * 8, dtype=np.uint32,
    ))

    def call():
        return qmv._qmm_tensor_kernel(
            bits, int(mode), input_dims, output_dims, bm, bn, bk, parent_tiles, offset,
        )(
            inputs=[x, trellis], grid=(output_dims // bn * 32, rows // bm, 1),
            threadgroup=(32, 1, 1), output_shapes=[(rows, output_dims)],
            output_dtypes=[mx.float16], init_value=float('nan'),
        )[0]

    exact_pair(monkeypatch, call)


def test_source_keeps_capacity_fallback_and_distinct_factory_names(monkeypatch):
    monkeypatch.setattr(qmv.mx.fast, 'metal_kernel', lambda **kwargs: kwargs)
    sources = []
    for enabled in (False, True):
        monkeypatch.setattr(qmv, '_USE_QMM_ADDRESS_HOIST', enabled)
        qmv._qmm_tensor_kernel.cache_clear()
        sources.append(qmv._qmm_tensor_kernel(4, 1, 128, 128, 32, 32, 16, 24, 3))
    old, new = sources
    assert old['name'] != new['name']
    assert 'ADDRESS_CAPACITY' not in old['source']
    assert 'constexpr uint ADDRESS_CAPACITY = BN * BK / 32u;' in new['source']
    assert new['source'].count('right.get_capacity() <= ADDRESS_CAPACITY') == 2
    assert new['source'].index('offset0[index] =') < new['source'].index('for (uint depth')
    assert 'words = trellis\n' in new['source']  # Original pointer in the fallback.
    assert 'return;' not in new['source']
    assert old['compile_options'] == new['compile_options'] == {'math_mode': 'safe'}
    assert old['header'] == new['header']
    for source in (old['source'], new['source']):
        assert source.count('operation.run(left, right, accumulator);') == 1
        assert source.count('right[index] = half(mlxl3_decode_codeword(codeword, 0));') == 1


@pytest.mark.usefixtures('tensor_ops')
@pytest.mark.parametrize('bits', range(1, 9))
@pytest.mark.parametrize('mode', list(CodebookMode))
def test_qmm_hoist_all_codebooks_and_bitwidths(monkeypatch, bits, mode):
    raw_case(monkeypatch, bits=bits, mode=mode)


@pytest.mark.usefixtures('tensor_ops')
@pytest.mark.parametrize('bm', [8, 16, 32, 64])
@pytest.mark.parametrize('bn', [16, 32, 64])
@pytest.mark.parametrize('bk', [16, 32, 64])
def test_qmm_hoist_all_tiles_with_strided_weights(monkeypatch, bm, bn, bk):
    raw_case(monkeypatch, bm=bm, bn=bn, bk=bk, offset=3, parent_tiles=24)


@pytest.mark.usefixtures('tensor_ops')
@pytest.mark.parametrize('input_dims,offset', [(16, 0), (256, 1), (2048, 7), (4096, 8)])
def test_qmm_hoist_depth_and_weight_offset_boundaries(monkeypatch, input_dims, offset):
    raw_case(monkeypatch, input_dims=input_dims, offset=offset, parent_tiles=24)


@pytest.mark.usefixtures('tensor_ops')
def test_qmm_hoist_capacity_overflow_falls_back_and_writes_every_output(monkeypatch):
    factory = qmv.mx.fast.metal_kernel
    changed = []

    def constrained_factory(**kwargs):
        if '_h1_v3' in kwargs['name']:
            marker = 'constexpr uint ADDRESS_CAPACITY = BN * BK / 32u;'
            assert kwargs['source'].count(marker) == 1
            kwargs['source'] = kwargs['source'].replace(
                marker, 'constexpr uint ADDRESS_CAPACITY = 1u;',
            )
            kwargs['name'] += '_test_capacity_one'
            changed.append(True)
        return factory(**kwargs)

    monkeypatch.setattr(qmv.mx.fast, 'metal_kernel', constrained_factory)
    # 32x16 right tensor needs more than one element/lane. Outputs start as NaN.
    raw_case(monkeypatch, bits=7, input_dims=256, offset=3, parent_tiles=24)
    assert changed == [True]


@pytest.mark.usefixtures('tensor_ops')
@pytest.mark.parametrize('dtype', [mx.float16, mx.bfloat16, mx.float32])
@pytest.mark.parametrize('rows', [0, 23, 24, 25, 65])
def test_qmm_hoist_public_dispatch_padding_and_dtype(monkeypatch, dtype, rows):
    monkeypatch.setattr(qmv, '_USE_TENSOR_QMM', True)
    monkeypatch.setattr(qmv, 'tensor_tile', lambda *args: (32, 32, 16))
    rng = np.random.default_rng(7401)
    trellis = mx.array(rng.integers(0, 65536, (8, 8, 64), dtype=np.uint16))
    x = mx.array(rng.normal(0, .02, (1, rows, 128)).astype(np.float32)).astype(dtype)
    scales = mx.ones((128,), dtype=mx.float16)

    def call():
        output = qmv.qmm_exl3(x, trellis, scales, scales, 4, CodebookMode.MCG)
        assert output.shape == (1, rows, 128) and output.dtype == dtype
        return output

    exact_pair(monkeypatch, call)


@pytest.mark.usefixtures('tensor_ops')
def test_qmm_hoist_public_strided_view_matches_old_and_contiguous(monkeypatch):
    monkeypatch.setattr(qmv, '_USE_TENSOR_QMM', True)
    monkeypatch.setattr(qmv, 'tensor_tile', lambda *args: (32, 32, 16))
    rng = np.random.default_rng(7402)
    trellis = mx.array(rng.integers(0, 65536, (16, 24, 48), dtype=np.uint16))
    x = mx.array(rng.normal(0, .02, (33, 256)).astype(np.float16))
    suh, svh = mx.ones((256,), dtype=mx.float16), mx.ones((128,), dtype=mx.float16)
    exact_pair(monkeypatch, lambda: qmv.qmm_exl3_view(
        x, trellis, suh, svh, 3, CodebookMode.MUL1, tile_offset=7, output_dims=128,
    ))
    actual = qmv.qmm_exl3_view(
        x, trellis, suh, svh, 3, CodebookMode.MUL1, tile_offset=7, output_dims=128,
    )
    expected = qmv.qmm_exl3(x, mx.contiguous(trellis[:, 7:15]), suh, svh, 3, CodebookMode.MUL1)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint16),
                                  np.asarray(expected).view(np.uint16))


@pytest.mark.parametrize('args', [(4, 1, 128, 128, 7), (4, 1, 128, 128, 32, 8),
                                 (4, 1, 128, 128, 32, 32, 8),
                                 (4, 1, 127, 128, 32), (4, 1, 128, 127, 32)])
def test_source_preserves_invalid_tile_rejection(monkeypatch, args):
    for enabled in (False, True):
        monkeypatch.setattr(qmv, '_USE_QMM_ADDRESS_HOIST', enabled)
        qmv._qmm_tensor_kernel.cache_clear()
        with pytest.raises(ValueError):
            qmv._qmm_tensor_kernel(*args)
