from itertools import permutations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from mlxl3 import cli
from mlxl3.kernels.routing import inverse_permutation


def kv(length):
    cache = KVCache()
    cache.update_and_fetch(
        mx.arange(length * 4).reshape(1, 1, length, 4).astype(mx.float16),
        mx.arange(length * 2).reshape(1, 1, length, 2).astype(mx.float16),
    )
    mx.eval(cache.keys, cache.values)
    return cache


def test_inverse_permutation_exhaustive():
    for size in range(6):
        for order in permutations(range(size)):
            array = mx.array(order, dtype=mx.uint32)
            assert inverse_permutation(array).tolist() == mx.argsort(array).tolist()


@pytest.mark.parametrize('size', [257, 1024, 8192])
def test_inverse_permutation_duplicate_experts(size):
    mx.random.seed(size)
    selected = mx.random.randint(0, 32, (size,))
    order = mx.argsort(selected)
    assert inverse_permutation(order).tolist() == mx.argsort(order).tolist()


@pytest.mark.parametrize('length', [1, 255, 256, 257, 512])
def test_kv_fork_reserve_and_independence(length):
    original = kv(length)
    expected = [value.tolist() for value in original.state]
    fork = cli.GenerationSession._clone_cache([original])[0]
    capacity = fork.keys.shape[2]
    assert capacity > length
    assert capacity % original.step == 0
    assert [value.tolist() for value in fork.state] == expected
    fork.update_and_fetch(mx.full((1, 1, 1, 4), -1), mx.full((1, 1, 1, 2), -1))
    mx.eval(fork.state)
    assert fork.keys.shape[2] == capacity
    assert [value.tolist() for value in original.state] == expected
    assert original.offset == length
    snapshot = cli.GenerationSession._clone_cache([original], reserve_tokens=0)[0]
    assert snapshot.keys.shape[2] == length
    assert [value.tolist() for value in snapshot.state] == expected


def test_empty_kv_fork():
    fork = cli.GenerationSession._clone_cache([KVCache()])[0]
    assert fork.keys is None and fork.offset == 0


def test_snapshot_rejected_before_clone(monkeypatch):
    session = cli.GenerationSession()
    session.prompt_cache = [kv(256)]
    monkeypatch.setattr(cli, '_PREFIX_CACHE_BUDGET_BYTES', session.nbytes())
    def forbidden(*args, **kwargs):
        pytest.fail('inadmissible snapshot must not allocate')
    monkeypatch.setattr(session, '_clone_cache', forbidden)
    session._store_block_cache(256)
    assert not session.block_caches


def test_snapshot_admitted_without_reserve(monkeypatch):
    session = cli.GenerationSession()
    session.prompt_cache = [kv(257)]
    monkeypatch.setattr(cli, '_PREFIX_CACHE_BUDGET_BYTES', 100_000)
    session._store_block_cache(257)
    assert session.block_caches[257][0].keys.shape[2] == 257
    assert session.nbytes() <= 100_000


@pytest.mark.parametrize('window', [1, 20, None])
@pytest.mark.parametrize('prefix', [[], [1, 2, 1], list(range(1000))])
@pytest.mark.parametrize('count', [0, 1, 19, 20, 40])
def test_processor_history_exact(window, prefix, count):
    tokens = list(range(count))
    processor = cli._prefix_processors([lambda history, logits: history], prefix, [window])[0]
    actual = processor(mx.array(tokens, dtype=mx.int32), None).tolist()
    expected = prefix + tokens
    if window is not None:
        expected = expected[-window:]
    assert actual == expected


def test_prefill_cancel_retires_partial_state(monkeypatch):
    from test_cli import FakeCachedModel
    model = FakeCachedModel()
    session = cli.GenerationSession()
    monkeypatch.setattr(cli, '_PREFILL_STEP_SIZE', 2)
    with pytest.raises(cli.GenerationCancelled):
        session.prepare(model, [1, 2, 3, 4, 9], [1, 2, 3, 4],
                        should_cancel=lambda: bool(model.inputs))
    assert model.inputs == [[1, 2]]
    assert session.prompt_cache is None
    assert session.tokens == []


@pytest.mark.parametrize('penalty', ['repetition_penalty', 'presence_penalty', 'frequency_penalty'])
def test_bounded_processor_logits_bit_exact(penalty):
    from mlx_lm.sample_utils import make_logits_processors
    prefix = [2, 3, 2, 5] * 1000
    suffix = mx.array([2, 6, 2, 3], dtype=mx.int32)
    logits = mx.array([[-1.0, 0.0, 1.0, -2.0, 3.0, 4.0, -5.0]])
    reference = make_logits_processors(**{penalty: 1.1})[0]
    bounded = cli._prefix_processors([reference], prefix, [20])[0]
    expected = reference(mx.concatenate([mx.array(prefix), suffix]), mx.array(logits))
    actual = bounded(suffix, mx.array(logits))
    assert actual.tolist() == expected.tolist()


def test_memory_guard_retires_prefill_state(monkeypatch):
    from test_cli import FakeCachedModel
    model = FakeCachedModel()
    session = cli.GenerationSession()
    monkeypatch.setattr(cli, '_mlx_memory_guard_bytes', lambda: 0)
    with pytest.raises(cli.MemorySafetyError):
        session.prepare(model, [1, 2, 9], [1, 2])
    assert not model.inputs
    assert session.prompt_cache is None
