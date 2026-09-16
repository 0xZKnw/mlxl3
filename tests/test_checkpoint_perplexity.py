import math
import runpy
from pathlib import Path

import mlx.core as mx


def test_uniform_logits_have_log_vocab_loss():
    loss = runpy.run_path(
        str(Path(__file__).parents[1] / "benchmarks/eval_checkpoint_perplexity.py")
    )["mean_nll"]
    assert math.isclose(
        float(loss(mx.zeros((1, 3, 5)), mx.array([[0, 2, 4]]))), math.log(5), rel_tol=1e-6
    )


def test_chunked_perplexity_resets_cache_and_weights_short_tail(monkeypatch):
    from mlx_lm.models import cache
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "benchmarks/eval_checkpoint_perplexity.py"))
    monkeypatch.setattr(cache, "make_prompt_cache", lambda model: [])
    calls = []
    def model(tokens, cache=None):
        calls.append((tokens.shape[-1], cache))
        return mx.zeros((*tokens.shape, 5))
    tokens = mx.array([[0, 1, 2, 3, 4, 0]])
    assert math.isclose(namespace["window_nll"](model, tokens, 2), math.log(5), rel_tol=1e-6)
    assert [length for length, _ in calls] == [2, 2, 1]
    assert all(state is calls[0][1] for _, state in calls)
