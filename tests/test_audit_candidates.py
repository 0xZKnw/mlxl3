from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import ArraysCache

from mlxl3 import moe, recurrent


@pytest.mark.parametrize('experts', [32, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize('ties', [False, True])
def test_bitonic_router_exact(experts, ties, monkeypatch):
    mx.random.seed(experts)
    values = mx.random.uniform(shape=(1, experts))
    if ties:
        values = mx.floor(values * 3)
    for dtype in (mx.float16, mx.float32):
        x = values.astype(dtype)
        monkeypatch.setattr(moe, '_USE_BITONIC_ROUTER', False)
        old = moe.router_topk(x, 8, normalize=True)
        monkeypatch.setattr(moe, '_USE_BITONIC_ROUTER', True)
        new = moe.router_topk(x, 8, normalize=True)
        assert old[0].tolist() == new[0].tolist()
        np.testing.assert_array_equal(np.asarray(old[1]), np.asarray(new[1]))


def test_bitonic_router_nan_inf_zero(monkeypatch):
    x = mx.array([[float('nan'), float('inf'), -float('inf'), 0., -0., 1., 1., -1.] * 4])
    monkeypatch.setattr(moe, '_USE_BITONIC_ROUTER', False)
    old = moe.router_topk(x, 8, normalize=False)
    monkeypatch.setattr(moe, '_USE_BITONIC_ROUTER', True)
    new = moe.router_topk(x, 8, normalize=False)
    assert old[0].tolist() == new[0].tolist()
    np.testing.assert_array_equal(np.asarray(old[1]), np.asarray(new[1]))


def test_shortconv_long_forced_state(monkeypatch):
    from mlx_lm.models.lfm2 import ShortConv
    args = SimpleNamespace(conv_L_cache=3, conv_bias=False, hidden_size=16)
    mx.random.seed(170)
    layer = ShortConv(args, 0)
    layer.set_dtype(mx.float16)
    original = layer.__call__
    monkeypatch.setattr(recurrent, '_USE_COMPILED_SHORTCONV', True)
    assert recurrent.compile_recurrent_layers(layer) == 1
    old_cache, new_cache = ArraysCache(1), ArraysCache(1)
    x = mx.random.normal((1, 17, 16)).astype(mx.float16)
    original(x, cache=old_cache)
    layer(x, cache=new_cache)
    for _ in range(1000):
        x = (mx.random.normal((1, 1, 16)) * .1).astype(mx.float16)
        expected = original(x, cache=old_cache)
        actual = layer(x, cache=new_cache)
        mx.eval(expected, actual, old_cache.state, new_cache.state)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=.002, rtol=.01)
        np.testing.assert_array_equal(np.asarray(old_cache[0]), np.asarray(new_cache[0]))


@pytest.mark.parametrize('bits', [2, 3, 4, 7])
@pytest.mark.parametrize('rows', [1, 25, 64, 513])
def test_strided_prefill_exact(bits, rows):
    from mlxl3.codec.codebook import CodebookMode
    from mlxl3.codec.trellis import pack_trellis
    from mlxl3.kernels.qmv import qmm_exl3, qmm_exl3_view
    rng = np.random.default_rng(38)
    codes = rng.integers(0, 1 << bits, size=(8, 24, 256), dtype=np.uint16)
    parent = mx.array(pack_trellis(codes, bits))
    x = mx.array(rng.normal(size=(rows, 128)).astype(np.float16))
    suh = mx.ones((128,), dtype=mx.float16)
    svh = mx.ones((128,), dtype=mx.float16)
    expected = qmm_exl3(x, parent[:, 8:16], suh, svh, bits, CodebookMode.MCG)
    actual = qmm_exl3_view(x, parent, suh, svh, bits, CodebookMode.MCG, tile_offset=8, output_dims=128)
    np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))


def test_prefill_gather_hadamard_exact():
    from mlxl3.kernels.qmv import _scaled_hadamard_input
    mx.random.seed(27)
    x = mx.random.normal((19, 256)).astype(mx.float16)
    scales = mx.random.uniform(.7, 1.3, (32, 2, 256)).astype(mx.float16)
    tokens = mx.array([3, 1, 3, 7, 1, 0], dtype=mx.uint32)
    experts = mx.array([2, 5, 2, 5, 7, 0], dtype=mx.int32)
    expected = mx.stack([_scaled_hadamard_input(x[tokens], scales[experts, i]) for i in range(2)])
    actual = moe._prefill_gather_hadamard(x, scales, tokens, experts)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_gemma_weighted_reduction_preserves_fp32():
    from mlxl3.kernels.qmv import _scaled_hadamard_output, _scaled_hadamard_output_reduce
    mx.random.seed(88)
    raw = mx.random.normal((8, 256))
    scales = mx.random.uniform(.7, 1.3, (8, 256)).astype(mx.float16)
    scores = mx.array([[.100003, .2, .15, .05, .125, .075, .1, .200002]], dtype=mx.float32)
    expected = (_scaled_hadamard_output(raw, scales)[None] * scores[..., None]).sum(-2)
    actual = _scaled_hadamard_output_reduce(raw, scales, scores)
    assert actual.dtype == mx.float32
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_cow_kv_owners_and_branches():
    from mlxl3.cache import SharedKVCache
    from mlxl3.cli import _cache_nbytes
    root = SharedKVCache()
    root.update_and_fetch(mx.ones((1, 2, 255, 64)), mx.ones((1, 2, 255, 32)))
    mx.eval(root.keys, root.values)
    fork = root.fork()
    mx.eval(fork.keys, fork.values)
    owner = root.storage_owner
    assert _cache_nbytes([root, fork]) == owner.nbytes
    assert fork.keys is not root.keys
    fork.update_and_fetch(mx.full((1, 2, 2, 64), 3.), mx.full((1, 2, 2, 32), 4.))
    mx.eval(fork.keys, fork.values)
    assert fork.storage_owner is None
    assert root.offset == 255 and fork.offset == 257
    assert root.keys[0, 0, 0, 0].item() == 1
    assert fork.keys[0, 0, 255, 0].item() == 3
    assert _cache_nbytes([root, fork]) == owner.nbytes + fork.keys.nbytes + fork.values.nbytes
    # Updating the original must not corrupt the branch or its rollback.
    root.trim(5)
    root.update_and_fetch(mx.full((1, 2, 1, 64), 9.), mx.full((1, 2, 1, 32), 8.))
    assert fork.keys[0, 0, 250, 0].item() == 1
    assert root.keys[0, 0, 250, 0].item() == 9


def test_cow_kv_fork_no_payload_allocation():
    from mlxl3.cache import SharedKVCache
    root = SharedKVCache()
    root.update_and_fetch(mx.ones((1, 4, 1024, 128)), mx.ones((1, 4, 1024, 128)))
    mx.eval(root.keys, root.values)
    before = mx.get_active_memory()
    forks = [root.fork() for _ in range(16)]
    mx.eval([(f.keys, f.values) for f in forks])
    assert mx.get_active_memory() - before < 4096


@pytest.mark.parametrize('bits', [2, 3, 4, 5, 6])
def test_qmv_output_fusion_exact(bits, monkeypatch):
    from mlxl3.codec.codebook import CodebookMode
    from mlxl3.codec.trellis import pack_trellis
    from mlxl3.kernels import qmv
    rng = np.random.default_rng(556)
    codes = rng.integers(0, 1 << bits, size=(8, 32, 256), dtype=np.uint16)
    trellis = mx.array(pack_trellis(codes, bits))
    x = mx.array(rng.normal(size=(1, 128)).astype(np.float16))
    su = mx.array(rng.uniform(.7, 1.3, size=(128,)).astype(np.float16))
    sv = mx.array(rng.uniform(.7, 1.3, size=(512,)).astype(np.float16))
    monkeypatch.setattr(qmv, '_USE_QMV_OUTPUT_FUSION', False)
    expected = qmv.qmv_exl3(x, trellis, su, sv, bits, CodebookMode.MCG)
    monkeypatch.setattr(qmv, '_USE_QMV_OUTPUT_FUSION', True)
    actual = qmv.qmv_exl3(x, trellis, su, sv, bits, CodebookMode.MCG)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_device_side_route_buckets(monkeypatch):
    from mlxl3.codec.codebook import CodebookMode
    from mlxl3.codec.trellis import pack_trellis
    from mlxl3.kernels import qmv
    selected = mx.array([0] * 3 + [2] * 9 + [3] * 32 + [4] * 65, dtype=mx.int32)
    blocks = selected.size // 64 + 5 + 1
    table, count = moe._segmented_block_table(5, blocks)(selected)
    rng = np.random.default_rng(22)
    parent = mx.array(pack_trellis(rng.integers(0, 8, (8, 40, 256), dtype=np.uint16), 3))
    x = mx.array(rng.normal(size=(selected.size + 64, 128)).astype(np.float16))
    def run():
        return qmv.qmm_exl3_expert_segmented(x, parent, table, count, rows=selected.size,
                                            tiles_per_expert=8, output_dims=128,
                                            expert_tile_base=0, k=3, mode=CodebookMode.MCG)
    monkeypatch.setattr(qmv, '_USE_SEGMENTED_BUCKETS', False)
    expected = run()
    monkeypatch.setattr(qmv, '_USE_SEGMENTED_BUCKETS', True)
    actual = run()
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_compact_conv_state_detaches_large_parent():
    from mlxl3.cache import compact_conv_states
    cache = ArraysCache(2)
    parent = mx.ones((1, 4096, 128))
    mx.eval(parent)
    cache[0] = parent[:, -2:, :]
    del parent
    cache[1] = mx.ones((1, 4, 16, 16))
    mx.eval(cache.state)
    mx.synchronize()
    before = mx.get_active_memory()
    matrix = cache[1]
    compact_conv_states([cache], 4096)
    mx.eval(cache.state)
    mx.synchronize()
    assert mx.get_active_memory() < before - 1_000_000
    assert cache[0].shape == (1, 2, 128)
    assert mx.all(cache[0] == 1).item()
    assert cache[1] is matrix


@pytest.mark.parametrize('kind', ['bpe', 'spm'])
def test_detokenizer_metadata_is_shared_but_text_is_private(kind):
    from mlx_lm.tokenizer_utils import BPEStreamingDetokenizer, SPMStreamingDetokenizer

    from mlxl3.tokenizer import cache_detokenizer_metadata
    cls = BPEStreamingDetokenizer if kind == 'bpe' else SPMStreamingDetokenizer
    class Tokenizer:
        _detokenizer_class = cls
        accesses = 0
        @property
        def vocab(self):
            self.accesses += 1
            return {'hello': 0, 'world': 1, 'Ġ' if kind == 'bpe' else '▁': 2,
                    '<0xC3>' if kind == 'spm' else 'Ã': 3,
                    '<0xA9>' if kind == 'spm' else '©': 4}
    tokenizer = Tokenizer()
    expected = cls(tokenizer)
    assert cache_detokenizer_metadata(tokenizer)
    accesses = tokenizer.accesses
    a, b = tokenizer._detokenizer_class(tokenizer), tokenizer._detokenizer_class(tokenizer)
    assert tokenizer.accesses == accesses
    assert a is not b and a.tokens is not b.tokens
    assert a.tokenmap is b.tokenmap
    for token in [0, 2, 1, 2, 3, 4]:
        expected.add_token(token)
        a.add_token(token)
        assert a.last_segment == expected.last_segment
    expected.finalize()
    a.finalize()
    assert a.text == expected.text
    assert not b.text and not b.tokens
    assert not cache_detokenizer_metadata(tokenizer)
