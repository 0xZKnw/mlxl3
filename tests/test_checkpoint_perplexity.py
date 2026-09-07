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
