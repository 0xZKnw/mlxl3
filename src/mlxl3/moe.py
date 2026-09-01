"""Grouped EXL3 SwitchGLU for routed mixture-of-experts models."""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels.qmv import qmv_exl3_mapped


class EXL3SwitchGLU(nn.Module):
    """Evaluate every selected expert with two mapped Metal QMV launches.

    Gate and up matrices share the first launch.  The activated intermediate
    rows feed a second mapped launch for down projection.  The interface is
    intentionally identical to MLX-LM's ``SwitchGLU`` so Bailing's router and
    weighted reduction remain upstream code.
    """

    def __init__(
        self,
        *,
        gu_trellis: mx.array,
        gu_suh: mx.array,
        gu_svh: mx.array,
        down_trellis: mx.array,
        down_suh: mx.array,
        down_svh: mx.array,
        bits: int,
        mode: CodebookMode | int,
    ):
        super().__init__()
        self.gu_trellis = gu_trellis.view(mx.uint16) if gu_trellis.dtype == mx.int16 else gu_trellis
        self.gu_suh = gu_suh
        self.gu_svh = gu_svh
        self.down_trellis = (
            down_trellis.view(mx.uint16) if down_trellis.dtype == mx.int16 else down_trellis
        )
        self.down_suh = down_suh
        self.down_svh = down_svh
        self.bits = int(bits)
        self.mode = CodebookMode(mode)

        if gu_suh.ndim != 3 or gu_suh.shape[1] != 2:
            raise ValueError(f"gu_suh must have shape (experts, 2, input), got {gu_suh.shape}")
        if gu_svh.ndim != 3 or gu_svh.shape[:2] != gu_suh.shape[:2]:
            raise ValueError(f"gu_svh must have shape (experts, 2, hidden), got {gu_svh.shape}")
        if down_suh.ndim != 2 or down_svh.ndim != 2:
            raise ValueError("down scales must be rank-2 expert matrices")
        if down_suh.shape[0] != gu_suh.shape[0] or down_svh.shape[0] != gu_suh.shape[0]:
            raise ValueError("gate/up/down expert counts differ")

        self.num_experts = int(gu_suh.shape[0])
        self.input_dims = int(gu_suh.shape[2])
        self.hidden_dims = int(gu_svh.shape[2])
        self._hidden_tiles = self.hidden_dims // 16
        self._output_tiles = self.input_dims // 16

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if x.shape[:-1] != indices.shape[:-1]:
            raise ValueError(f"input/routing prefixes differ: {x.shape} and {indices.shape}")
        if x.shape[-1] != self.input_dims:
            raise ValueError(f"expected hidden width {self.input_dims}, got {x.shape[-1]}")

        rows = x.size // self.input_dims
        top_k = indices.shape[-1]
        slots = rows * top_k
        selected = indices.reshape(-1).astype(mx.int32)
        x_rows = x.reshape(rows, self.input_dims)
        x_gu = mx.broadcast_to(
            x_rows[:, None, None, :],
            (rows, top_k, 2, self.input_dims),
        ).reshape(slots * 2, self.input_dims)

        selected_gu = mx.repeat(selected, 2)
        projection = mx.arange(slots * 2, dtype=mx.uint32) & 1
        gate_or_up_base = (
            selected_gu.astype(mx.uint32) + projection * self.num_experts
        ) * self._hidden_tiles
        gu_tile_map = (
            gate_or_up_base[:, None] + mx.arange(self._hidden_tiles, dtype=mx.uint32)[None, :]
        ).reshape(-1)
        gu = qmv_exl3_mapped(
            x_gu,
            self.gu_trellis,
            self.gu_suh[selected].reshape(slots * 2, self.input_dims),
            self.gu_svh[selected].reshape(slots * 2, self.hidden_dims),
            gu_tile_map,
            output_dims=self.hidden_dims,
            k=self.bits,
            mode=self.mode,
        ).reshape(slots, 2, self.hidden_dims)
        hidden = nn.silu(gu[:, 0]) * gu[:, 1]

        down_base = selected.astype(mx.uint32) * self._output_tiles
        down_tile_map = (
            down_base[:, None] + mx.arange(self._output_tiles, dtype=mx.uint32)[None, :]
        ).reshape(-1)
        output = qmv_exl3_mapped(
            hidden,
            self.down_trellis,
            self.down_suh[selected],
            self.down_svh[selected],
            down_tile_map,
            output_dims=self.input_dims,
            k=self.bits,
            mode=self.mode,
        )
        return output.reshape(*indices.shape, self.input_dims)
