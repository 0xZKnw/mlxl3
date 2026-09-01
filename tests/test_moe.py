from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from mlxl3.codec.codebook import CodebookMode
from mlxl3.codec.trellis import pack_trellis
from mlxl3.kernels.qmv import qmv_exl3
from mlxl3.moe import EXL3SwitchGLU


def test_grouped_switch_glu_matches_individual_qmv() -> None:
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
    x = mx.array(rng.normal(size=(1, 1, dims)).astype(np.float16))
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
    actual = module(x, indices)
    mx.eval(expected, actual)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=0.05, rtol=0.005)
