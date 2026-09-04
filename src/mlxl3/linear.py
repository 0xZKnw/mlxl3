"""EXL3-backed linear layers that stay serialized during inference."""

from __future__ import annotations

import os
import weakref
from pathlib import Path

import mlx.core as mx
from mlx import nn
from safetensors import safe_open

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels.qmv import qmm_exl3, qmm_exl3_view, qmv_exl3, qmv_exl3_grouped
from mlxl3.kernels.reconstruct import reconstruct_public_weights_mlx

_USE_FUSED_GATED_DELTA_INPUTS = (
    os.environ.get("MLXL3_FUSED_GATED_DELTA_INPUTS", "1") != "0"
)
_USE_PADDED_GROUPS = os.environ.get('MLXL3_PADDED_LINEAR_GROUPS', '1') != '0'
_USE_QK_GROUPS = os.environ.get('MLXL3_QK_GROUPS', '0') == '1'
_USE_STRIDED_PREFILL = os.environ.get('MLXL3_STRIDED_PREFILL', '0') == '1'


class EXL3Linear(nn.Module):
    """An EXL3 matrix kept in serialized form for decode and arbitrary prefill."""

    def __init__(
        self,
        trellis: mx.array,
        suh: mx.array,
        svh: mx.array,
        *,
        bits: int,
        mode: CodebookMode | int = CodebookMode.DEFAULT,
        bias: mx.array | None = None,
        logical_shape: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.trellis = trellis.view(mx.uint16) if trellis.dtype == mx.int16 else trellis
        self.suh = suh
        self.svh = svh
        self.bits = int(bits)
        self.mode = CodebookMode(mode)
        self.bias = bias
        self.logical_shape = logical_shape
        if logical_shape is not None and (
            len(logical_shape) != 2
            or not 0 < logical_shape[0] <= self.output_dims
            or not 0 < logical_shape[1] <= self.input_dims
        ):
            raise ValueError('logical EXL3 shape exceeds serialized dimensions')
        self._dense_cache: mx.array | None = None

    @property
    def input_dims(self) -> int:
        return self.trellis.shape[0] * 16

    @property
    def output_dims(self) -> int:
        return self.trellis.shape[1] * 16

    def reconstruct(self, *, cache: bool = True) -> mx.array:
        if self._dense_cache is None:
            weight = reconstruct_public_weights_mlx(
                self.trellis,
                self.suh,
                self.svh,
                self.bits,
                self.mode,
            )
            if cache:
                self._dense_cache = weight
            return weight
        return self._dense_cache

    def __call__(self, x: mx.array) -> mx.array:
        if self.logical_shape is not None:
            if x.shape[-1] != self.logical_shape[1]:
                raise ValueError('EXL3 logical input width mismatch')
            if x.shape[-1] < self.input_dims:
                x = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, self.input_dims - x.shape[-1])])
        if x.size == self.input_dims:
            output = qmv_exl3(
                x,
                self.trellis,
                self.suh,
                self.svh,
                self.bits,
                self.mode,
            )
        else:
            output = qmm_exl3(
                x,
                self.trellis,
                self.suh,
                self.svh,
                self.bits,
                self.mode,
            )
        if self.logical_shape is not None:
            output = output[..., :self.logical_shape[0]]
        return output if self.bias is None else output + self.bias

    @classmethod
    def from_safetensors(cls, path: str | Path, prefix: str) -> EXL3Linear:
        """Load one EXL3 matrix without materializing the rest of its shard."""

        with safe_open(str(path), framework="numpy") as handle:
            trellis = handle.get_tensor(f"{prefix}.trellis")
            suh = handle.get_tensor(f"{prefix}.suh")
            svh = handle.get_tensor(f"{prefix}.svh")
            keys = set(handle.keys())
            if f"{prefix}.mul1" in keys:
                mode = CodebookMode.MUL1
            elif f"{prefix}.mcg" in keys:
                mode = CodebookMode.MCG
            else:
                mode = CodebookMode.DEFAULT
        bits = trellis.shape[-1] // 16
        return cls(
            mx.array(trellis.view("uint16")),
            mx.array(suh),
            mx.array(svh),
            bits=bits,
            mode=mode,
        )


class _GroupRuntime:
    """Ephemeral Python-only cache shared by consecutive projection proxies."""

    def __init__(self) -> None:
        self.input: mx.array | None = None
        self.outputs: tuple[mx.array, ...] | None = None


class EXL3LinearGroup(nn.Module):
    """Compatible ragged EXL3 projections evaluated by one decode QMV launch."""

    def __init__(self, linears: tuple[EXL3Linear, ...]):
        super().__init__()
        if len(linears) < 2:
            raise ValueError("an EXL3 linear group needs at least two projections")
        first = linears[0]
        if first.bits == 7:
            raise ValueError("K=7 cannot use grouped EXL3 QMV")
        if any(
            linear.input_dims != first.input_dims
            or linear.bits != first.bits
            or linear.mode != first.mode
            for linear in linears[1:]
        ):
            raise ValueError("grouped EXL3 projections require equal input/K/codebook")

        self.trellis = mx.concatenate([linear.trellis for linear in linears], axis=1)
        self.suh = mx.stack([linear.suh for linear in linears])
        self.svh = mx.concatenate([linear.svh for linear in linears])
        self.output_widths = tuple(linear.output_dims for linear in linears)
        self.logical_output_widths = tuple(linear.logical_shape[0] if linear.logical_shape else linear.output_dims for linear in linears)
        self.output_tiles = tuple(width // 16 for width in self.output_widths)
        self.tile_sub = mx.concatenate(
            [
                mx.full((tiles,), index, dtype=mx.uint32)
                for index, tiles in enumerate(self.output_tiles)
            ]
        )
        self.biases = tuple(linear.bias for linear in linears)
        self.input_dims = first.input_dims
        self.bits = first.bits
        self.mode = first.mode
        self._runtime = _GroupRuntime()

    def _evaluate(self, x: mx.array) -> tuple[mx.array, ...]:
        padded = self.logical_output_widths != self.output_widths
        if x.size == self.input_dims and (not padded or _USE_PADDED_GROUPS):
            outputs = qmv_exl3_grouped(
                x,
                self.trellis,
                self.suh,
                self.svh,
                self.tile_sub,
                output_dims=self.output_widths,
                k=self.bits,
                mode=self.mode,
            )
        else:
            tile_cursor = 0
            scale_cursor = 0
            parts = []
            for index, (width, tiles) in enumerate(
                zip(self.output_widths, self.output_tiles, strict=True)
            ):
                if _USE_STRIDED_PREFILL and x.size != self.input_dims:
                    parts.append(qmm_exl3_view(
                        x, self.trellis, self.suh[index],
                        self.svh[scale_cursor:scale_cursor + width], self.bits, self.mode,
                        tile_offset=tile_cursor, output_dims=width,
                    ))
                    tile_cursor += tiles
                    scale_cursor += width
                    continue
                parts.append(
                    (qmv_exl3 if x.size == self.input_dims else qmm_exl3)(
                        x,
                        self.trellis[:, tile_cursor : tile_cursor + tiles],
                        self.suh[index],
                        self.svh[scale_cursor : scale_cursor + width],
                        self.bits,
                        self.mode,
                    )
                )
                tile_cursor += tiles
                scale_cursor += width
            outputs = tuple(parts)
        return tuple(
            output[..., :width] if bias is None else output[..., :width] + bias
            for output, bias, width in zip(outputs, self.biases, self.logical_output_widths, strict=True)
        )

    def project(self, index: int, x: mx.array) -> mx.array:
        runtime = self._runtime
        if runtime.input is not x or runtime.outputs is None:
            runtime.input = x
            runtime.outputs = self._evaluate(x)
        output = runtime.outputs[index]
        if index == len(self.output_widths) - 1:
            runtime.input = None
            runtime.outputs = None
        return output


class EXL3ProjectionProxy(nn.Module):
    """Preserve an architecture's individual projection call sites."""

    def __init__(self, group: EXL3LinearGroup, index: int):
        super().__init__()
        self._group = weakref.ref(group)
        self.index = index
        self.input_dims = group.input_dims
        self.output_dims = group.output_widths[index]
        self.bits = group.bits
        self.mode = group.mode

    def __call__(self, x: mx.array) -> mx.array:
        group = self._group()
        if group is None:
            raise RuntimeError("grouped EXL3 projection owner was released")
        return group.project(self.index, x)


def fuse_compatible_linear_groups(model: nn.Module) -> int:
    """Fuse common QKV and gate/up call sequences without architecture forks."""

    fused = 0
    patterns = [
        ("q_proj", "k_proj", "v_proj"),
        ("gate_proj", "up_proj"),
    ]
    if _USE_QK_GROUPS:
        # QKV is considered first. Its proxies prevent a second/nested group.
        # Useful for shared-KV Gemma layers with Q and K but no V projection.
        patterns.append(("q_proj", "k_proj"))
    if _USE_FUSED_GATED_DELTA_INPUTS:
        patterns.append(("in_proj_qkv", "in_proj_z"))
    for _, module in list(model.named_modules()):
        for names in patterns:
            linears = tuple(getattr(module, name, None) for name in names)
            if not all(isinstance(linear, EXL3Linear) for linear in linears):
                continue
            first = linears[0]
            if any(linear.logical_shape is not None and (
                linear.logical_shape[1] != linear.input_dims or
                (not _USE_PADDED_GROUPS and linear.logical_shape[0] != linear.output_dims)
            ) for linear in linears):
                continue
            if first.bits == 7 or any(
                linear.input_dims != first.input_dims
                or linear.bits != first.bits
                or linear.mode != first.mode
                for linear in linears[1:]
            ):
                continue
            group = EXL3LinearGroup(linears)
            group_name = "_mlxl3_" + "_".join(names)
            setattr(module, group_name, group)
            for index, name in enumerate(names):
                setattr(module, name, EXL3ProjectionProxy(group, index))
            fused += 1
    return fused
