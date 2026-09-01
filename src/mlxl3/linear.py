"""EXL3-backed linear layer with a correctness-first dense fallback."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx import nn
from safetensors import safe_open

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels.qmv import qmm_exl3, qmv_exl3
from mlxl3.kernels.reconstruct import reconstruct_public_weights_mlx


class EXL3Linear(nn.Module):
    """An EXL3 matrix kept in serialized form for decode and small prefill."""

    def __init__(
        self,
        trellis: mx.array,
        suh: mx.array,
        svh: mx.array,
        *,
        bits: int,
        mode: CodebookMode | int = CodebookMode.DEFAULT,
        bias: mx.array | None = None,
    ):
        super().__init__()
        self.trellis = trellis.view(mx.uint16) if trellis.dtype == mx.int16 else trellis
        self.suh = suh
        self.svh = svh
        self.bits = int(bits)
        self.mode = CodebookMode(mode)
        self.bias = bias
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
        if x.size == self.input_dims:
            output = qmv_exl3(
                x,
                self.trellis,
                self.suh,
                self.svh,
                self.bits,
                self.mode,
            )
        elif x.size // self.input_dims <= 64 and self.bits != 7:
            output = qmm_exl3(
                x,
                self.trellis,
                self.suh,
                self.svh,
                self.bits,
                self.mode,
            )
        else:
            # Long offline batches use a transient public weight. Generation
            # never reaches this path and the dense matrix is not retained.
            output = x @ self.reconstruct(cache=False)
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
