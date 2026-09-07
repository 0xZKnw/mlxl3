"""Bit-exact gates against the unchanged PonyExl3 search; no weights required."""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "references/PonyExl3"))
ref = pytest.importorskip("ponyexl3.convert.metal_search")
from mlxl3.quantize import _decoded_codebook, quantize_tiles_mlx

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="Metal required")


@pytest.mark.parametrize("cb", range(3))
def test_half_lookup_preserves_all_codebook_values(cb):
    values = _decoded_codebook(cb)
    restored = values.astype(mx.float16).astype(mx.float32)
    assert np.asarray(values).tobytes() == np.asarray(restored).tobytes()


@pytest.mark.parametrize("k", range(2, 9))
@pytest.mark.parametrize("cb", range(3))
def test_exact_search_and_chunking(k, cb):
    x = np.random.default_rng(410 + k * 3 + cb).normal(size=(7, 256)).astype(np.float32)
    x[0] = 0
    x[1] *= 1000  # Saturation and tie-breaking, not just normally scaled weights.
    before = ref.quantize_tiles_mlx(x, k, cb)
    after = quantize_tiles_mlx(x, k, cb, max_scratch_bytes=3 * 256 * (1 << (16 - k)) * 2)
    for a, b in zip(before, after):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


@pytest.mark.parametrize(
    "tiles,k,budget",
    [
        (np.zeros((1, 256)), 1, 100),
        (np.zeros((1, 256)), 9, 100),
        (np.zeros((0, 256)), 4, 100),
        (np.zeros((1, 16)), 4, 100),
        (np.zeros((1, 256)), 4, 0),
    ],
)
def test_invalid_arguments(tiles, k, budget):
    with pytest.raises(ValueError):
        quantize_tiles_mlx(tiles, k, max_scratch_bytes=budget)


def test_cancellation_between_launches(monkeypatch):
    from ponyexl3.convert import cancel

    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise cancel.ConversionCancelled("test")

    monkeypatch.setattr(cancel, "raise_if_requested", check)
    with pytest.raises(cancel.ConversionCancelled):
        quantize_tiles_mlx(np.zeros((7, 256)), 4, max_scratch_bytes=2 * 1024 * 1024)
    assert calls == 3


@pytest.mark.parametrize("k", range(2, 9))
def test_ldlq_feedback_is_exact(k, monkeypatch):
    from ponyexl3.convert.hessian import _ldlq_inner_matrix_mlx
    from ponyexl3.ref.codebook import CodebookMode

    rng = np.random.default_rng(543)
    weights = rng.normal(size=(128, 256)).astype(np.float32)
    factor = np.tril(rng.normal(scale=0.01, size=(128, 128))).astype(np.float32)
    np.fill_diagonal(factor, 1)
    kwargs = {
        "k": k,
        "cb": CodebookMode.MCG,
        "hessian": None,
        "buf_size_rows": 128,
        "feedback_rows": 16,
        "compute_proxy": False,
    }
    before = _ldlq_inner_matrix_mlx(weights, factor, **kwargs)
    monkeypatch.setattr(ref, "quantize_tiles_mlx", quantize_tiles_mlx)
    after = _ldlq_inner_matrix_mlx(weights, factor, **kwargs)
    assert before.packed.tobytes() == after.packed.tobytes()
    assert before.reconstructed.tobytes() == after.reconstructed.tobytes()


@pytest.mark.parametrize("k", range(2, 9))
@pytest.mark.parametrize("cb", range(3))
def test_optimized_transition_extremes(k, cb):
    rng = np.random.default_rng(834 + k + cb)
    x = rng.normal(size=(32, 256)).astype(np.float32)
    x *= np.geomspace(1e-5, 1e4, 32).astype(np.float32)[:, None]
    x[0] = -0.0
    x[1] = 0.5
    x[2] = np.inf
    x[3] = -np.inf
    x[4, ::2] = np.nan
    x[5] = np.resize(np.array([-1, 0, 1], dtype=np.float32), 256)
    before = ref.quantize_tiles_mlx(x, k, cb)
    after = quantize_tiles_mlx(x, k, cb)
    for a, b in zip(before, after):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
