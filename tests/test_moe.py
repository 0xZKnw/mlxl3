from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

import mlxl3.kernels.qmv as qmv_kernels
import mlxl3.moe as moe_module
from mlxl3.codec.codebook import CodebookMode
from mlxl3.codec.trellis import pack_trellis
from mlxl3.kernels.qmv import qmv_exl3
from mlxl3.moe import EXL3SwitchGLU, router_topk, topk_biased


def test_fused_router_topk_matches_mlx_chain() -> None:
    rng = np.random.default_rng(52900)
    logits = mx.array(rng.normal(size=(1, 1, 256)).astype(np.float16))
    values = mx.softmax(logits, axis=-1, precise=True)
    expected_indices = mx.argpartition(values, kth=-8, axis=-1)[..., -8:]
    expected_scores = mx.take_along_axis(values, expected_indices, axis=-1)
    expected_scores = expected_scores / expected_scores.sum(axis=-1, keepdims=True)

    actual_indices, actual_scores = router_topk(values, 8, normalize=True)
    mx.eval(expected_indices, expected_scores, actual_indices, actual_scores)

    np.testing.assert_array_equal(np.asarray(actual_indices), np.asarray(expected_indices))
    np.testing.assert_array_equal(np.asarray(actual_scores), np.asarray(expected_scores))


def test_fused_biased_topk_matches_mlx_selection() -> None:
    rng = np.random.default_rng(53000)
    probabilities = mx.array(rng.uniform(0.0, 1.0, size=(7, 32)).astype(np.float16))
    bias = mx.array(rng.uniform(-0.2, 0.2, size=(32,)).astype(np.float16))

    expected_indices = mx.argpartition(
        probabilities.astype(mx.float32) + bias,
        kth=-4,
        axis=-1,
    )[..., -4:]
    expected_scores = mx.take_along_axis(probabilities, expected_indices, axis=-1)
    actual_indices, actual_scores = topk_biased(probabilities, bias, 4)
    mx.eval(expected_indices, expected_scores, actual_indices, actual_scores)

    np.testing.assert_array_equal(np.asarray(actual_indices), np.asarray(expected_indices))
    np.testing.assert_array_equal(np.asarray(actual_scores), np.asarray(expected_scores))


def test_grouped_switch_glu_matches_individual_qmv(monkeypatch) -> None:
    rng = np.random.default_rng(53100)
    experts = 4
    dims = hidden = 128
    k = 3
    mode = CodebookMode.MCG

    def trellis() -> mx.array:
        codes = rng.integers(0, 1 << k, size=(8, 8, 256), dtype=np.uint16)
        return mx.array(pack_trellis(codes, k))

    gates = [trellis() for _ in range(experts)]
    ups = [trellis() for _ in range(experts)]
    downs = [trellis() for _ in range(experts)]
    gu_suh = mx.array(rng.uniform(0.7, 1.3, size=(experts, 2, dims)).astype(np.float16))
    gu_svh = mx.array(rng.uniform(0.7, 1.3, size=(experts, 2, hidden)).astype(np.float16))
    down_suh = mx.array(rng.uniform(0.7, 1.3, size=(experts, hidden)).astype(np.float16))
    down_svh = mx.array(rng.uniform(0.7, 1.3, size=(experts, dims)).astype(np.float16))
    module = EXL3SwitchGLU(
        gu_trellis=mx.concatenate(gates + ups, axis=1),
        gu_suh=gu_suh,
        gu_svh=gu_svh,
        down_trellis=mx.concatenate(downs, axis=1),
        down_suh=down_suh,
        down_svh=down_svh,
        bits=k,
        mode=mode,
    )
    # Keep the synthetic activations in the range observed after RMSNorm. The
    # fused path deliberately rounds gate/up values once before SwiGLU, matching
    # real FP16 model execution without amplifying unrealistic random weights.
    x = mx.array((rng.normal(size=(1, 1, dims)) * 0.05).astype(np.float16))
    indices = mx.array([[[1, 3]]], dtype=mx.int32)

    expected_rows = []
    for expert in (1, 3):
        gate = qmv_exl3(x, gates[expert], gu_suh[expert, 0], gu_svh[expert, 0], k, mode)
        up = qmv_exl3(x, ups[expert], gu_suh[expert, 1], gu_svh[expert, 1], k, mode)
        hidden_row = nn.silu(gate) * up
        expected_rows.append(
            qmv_exl3(
                hidden_row,
                downs[expert],
                down_suh[expert],
                down_svh[expert],
                k,
                mode,
            )
        )
    expected = mx.stack(expected_rows, axis=-2)
    monkeypatch.setattr(moe_module, "_USE_FUSED_MOE_GLU_PREP", False)
    monkeypatch.setattr(qmv_kernels, "_USE_K3_WINDOW_DECODE", False)
    legacy = module(x, indices)
    monkeypatch.setattr(qmv_kernels, "_USE_K3_WINDOW_DECODE", True)
    windowed = module(x, indices)
    monkeypatch.setattr(moe_module, "_USE_FUSED_MOE_GLU_PREP", True)
    optimized = module(x, indices)
    scores = mx.array([[[0.375, 0.625]]], dtype=mx.float16)
    expected_reduced = (optimized * scores[..., None]).sum(axis=-2)
    reduced = module(x, indices, scores=scores)
    mx.eval(expected, legacy, windowed, optimized, expected_reduced, reduced)
    np.testing.assert_array_equal(np.asarray(windowed), np.asarray(legacy))
    np.testing.assert_array_equal(np.asarray(reduced), np.asarray(expected_reduced))
    np.testing.assert_allclose(
        np.asarray(optimized), np.asarray(legacy), atol=0.1, rtol=0.01
    )
    np.testing.assert_allclose(
        np.asarray(optimized), np.asarray(expected), atol=0.05, rtol=0.005
    )

    batch_x = mx.broadcast_to(x, (1, 4, dims))
    batch_indices = mx.broadcast_to(indices, (1, 4, 2))
    batch_scores = mx.broadcast_to(scores, (1, 4, 2))
    monkeypatch.setattr(moe_module, "_USE_FUSED_MOE_GLU_PREP", False)
    monkeypatch.setattr(moe_module, "_USE_SEGMENTED_MOE_PREFILL", False)
    batch_legacy = module(batch_x, batch_indices, scores=batch_scores)
    monkeypatch.setattr(moe_module, "_USE_SEGMENTED_MOE_PREFILL", True)
    monkeypatch.setattr(moe_module, "_SEGMENTED_MOE_MIN_ROWS", 1)
    batch_segmented = module(batch_x, batch_indices, scores=batch_scores)
    mx.eval(batch_legacy, batch_segmented)
    np.testing.assert_allclose(
        np.asarray(batch_segmented),
        np.asarray(batch_legacy),
        atol=0.1,
        rtol=0.01,
    )
