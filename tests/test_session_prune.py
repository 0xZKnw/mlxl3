import mlx.core as mx
import pytest
from test_cli import FakeCachedModel, FakeStateCache

from mlxl3 import cli


@pytest.mark.parametrize('marker', ['<think>', '<|channel>thought'])
@pytest.mark.parametrize('suffix', ['', '\n', '\n\n', ' '])
def test_restored_reasoning_preserves_exact_template_bytes(marker, suffix):
    generated = 'reasoning\n</think>answer'
    prompt = 'assistant\n' + marker + suffix
    assert cli._restore_reasoning_opener(generated, prompt) == marker + suffix + generated


def cache(tokens):
    return [FakeStateCache(mx.array(tokens, dtype=mx.int32))]


def test_prune_retains_exact_prefix_and_bounds_rerun(monkeypatch):
    monkeypatch.setattr(cli, '_PREFIX_CACHE_BLOCK_SIZE', 0)
    monkeypatch.setattr(cli, '_mlx_memory_guard_bytes', lambda: 1_000_000_000)
    history, answer, question = list(range(8192)), [9000, 9001], [9100, 9101, 9102]
    exact = history + answer
    stable, full = exact + question, exact + question + [9999]
    pool = cli.GenerationSessionPool(budget_bytes=40_000)
    session = pool.acquire('active')
    session.tokens, session.prompt_cache = history, cache(history)
    session.finish(history, answer, cache(exact))
    session.block_caches[4096] = cache(history[:4096])
    assert session.nbytes() > pool.budget_bytes
    retained = session.exact_cache
    pool.prune('active')
    assert session.exact_cache is retained and session.exact_tokens == exact
    assert session.prompt_cache is None and not session.tokens and not session.block_caches
    assert session.nbytes() <= pool.budget_bytes

    model, cold_model = FakeCachedModel(), FakeCachedModel()
    suffix, fork, common, evaluated = session.prepare(model, full, stable)
    _, cold_cache, _, cold_evaluated = cli.GenerationSession().prepare(cold_model, full, stable)
    assert common == len(exact) and suffix == [9999] and evaluated == 4
    assert model.inputs == [question]
    assert sum(map(len, cold_model.inputs)) == len(stable)
    assert fork[0].state.tolist() == cold_cache[0].state.tolist() == stable
    fork[0].state[0] = -1
    assert session.prompt_cache[0].state.tolist() == stable
    print({'old_reset_evaluated_tokens': cold_evaluated, 'retained_exact_evaluated_tokens': evaluated,
           'cached_prefix_tokens': common})


def test_prune_discards_oldest_block_before_live_caches():
    pool = cli.GenerationSessionPool(budget_bytes=44)
    session = pool.acquire('active')
    session.tokens, session.prompt_cache = [1, 2, 3, 4], cache([1, 2, 3, 4])
    session.finish(session.tokens, [5], cache([1, 2, 3, 4, 5]))
    session.block_caches[2] = cache([1, 2])
    session.block_caches[4] = cache([3, 4])
    pool.prune('active')
    assert list(session.block_caches) == [4]
    assert session.prompt_cache is not None and session.exact_cache is not None
    assert session.nbytes() == pool.budget_bytes


def test_pruned_exact_prefix_mismatch_rebuilds_clean_cache(monkeypatch):
    monkeypatch.setattr(cli, '_mlx_memory_guard_bytes', lambda: 1_000_000_000)
    pool = cli.GenerationSessionPool(budget_bytes=24)
    session = pool.acquire('active')
    session.tokens, session.prompt_cache = [1, 2, 3, 4], cache([1, 2, 3, 4])
    session.finish(session.tokens, [5], cache([1, 2, 3, 4, 5]))
    pool.prune('active')
    assert session.prompt_cache is None and session.exact_cache is not None
    model = FakeCachedModel()
    stable = [1, 2, 77, 88]
    _, fork, common, evaluated = session.prepare(model, stable + [9999], stable)
    assert common == 0 and evaluated == 5
    assert model.inputs == [stable] and fork[0].state.tolist() == stable
    assert session.exact_cache is None and not session.exact_tokens


@pytest.mark.parametrize('budget', [0, 4, 16])
def test_prune_oversized_exact_preserves_admissible_fallback(budget):
    pool = cli.GenerationSessionPool(budget_bytes=budget)
    session = pool.acquire('active')
    session.tokens, session.prompt_cache = [1, 2], cache([1, 2])
    session.finish([1, 2], [3, 4, 5, 6], cache([1, 2, 3, 4, 5, 6]))
    pool.prune('active')
    assert session.nbytes() <= budget
    assert session.exact_cache is None and not session.exact_tokens
    assert session.tokens == ([1, 2] if budget >= 8 else [])
    assert (session.prompt_cache is not None) == (budget >= 8)
