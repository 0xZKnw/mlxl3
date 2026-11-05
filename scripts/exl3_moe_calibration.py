"""Opt-in routed Hessians for the pinned ExLlamaV3 converter; inference is unchanged."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

MODES = ("routed", "affinity", "squared")
BLOCK_SHA = "ec05c84fc6213221797256568a0f05d43b07737ef1c36a320f1926b254e3ecd7"


def expert_qmap(block: str, expert: int, projection: str) -> str:
    if expert < 0 or projection not in ("gate", "up", "down"):
        raise ValueError("Invalid expert or projection")
    return f"{block}.expert.{expert}.{'down' if projection == 'down' else 'input'}"


def check_coverage(count: int, mass: float, mass2: float, minimum: int) -> float:
    if minimum < 1 or count < minimum:
        raise ValueError("Insufficient real expert rows")
    if not math.isfinite(mass) or not math.isfinite(mass2) or mass <= 0 or mass2 <= 0:
        raise ValueError("Invalid expert affinity mass")
    effective = (mass / math.sqrt(mass2)) ** 2
    if not math.isfinite(effective) or effective + 1e-6 < minimum:
        raise ValueError("Insufficient effective expert rows")
    return effective


def expert_capture_params(params: dict, weights):
    if "capture" not in params or "mlxl3_moe_mode" not in params:
        return params
    return {**params, "mlxl3_expert_weights": weights.reshape(-1)}


def capture_weighted(linear, x, params: dict) -> None:
    import torch

    mode = params["mlxl3_moe_mode"]
    if mode not in MODES:
        raise ValueError("Unknown MoE calibration mode")
    weights = params["mlxl3_expert_weights"].float()
    values = x.reshape(-1, x.shape[-1]).float()
    if (
        weights.ndim != 1
        or weights.shape[0] != values.shape[0]
        or values.shape[0] == 0
        or values.shape[1] != linear.in_features
        or not torch.isfinite(values).all().item()
        or not torch.isfinite(weights).all().item()
        or not (weights > 0).all().item()
    ):
        raise ValueError("Invalid routed activations or affinities")
    if mode == "routed":
        weights = torch.ones_like(weights)
    elif mode == "squared":
        weights = weights.square()
    scaled = values * weights.sqrt().unsqueeze(1)
    if not torch.isfinite(scaled).all().item() or not (weights > 0).all().item():
        raise ValueError("Affinity scaling overflow or underflow")
    capture = params["capture"]
    if linear.qmap not in capture:
        capture[linear.qmap] = linear.init_H_data(True)
    data = capture[linear.qmap]
    if data["first_key"] != linear.key:
        return  # Gate/up share identical inputs only within their own expert.
    data["H"].addmm_(scaled.T, scaled)
    if not torch.isfinite(data["H"]).all().item():
        raise ValueError("Nonfinite weighted Hessian")
    data["count"] += values.shape[0]
    data["num_total"] += x.numel()
    data["affinity_mass"] = data.get("affinity_mass", 0.0) + weights.double().sum().item()
    data["affinity_mass2"] = (
        data.get("affinity_mass2", 0.0) + weights.double().square().sum().item()
    )


def patch_converter(directory: Path) -> str:
    """Pass the exact sorted routing weights alongside each real expert's inputs."""
    path = directory / "modules/block_sparse_mlp.py"
    original = path.read_bytes()
    if hashlib.sha256(original).hexdigest() != BLOCK_SHA:
        raise ValueError("Expected the unmodified pinned ExLlamaV3 MoE source")
    source = original.decode()
    changes = {
        "from typing_extensions import override": (
            "from typing_extensions import override\n"
            "from exl3_moe_calibration import expert_capture_params"
        ),
        "def mlp(exp_i, xc):\n": (
            "def mlp(exp_i, xc):\n"
            "                            expert_params = expert_capture_params(params, w)\n"
        ),
        "self.ups[exp_i].forward(xc, params)": "self.ups[exp_i].forward(xc, expert_params)",
        "self.gates[exp_i].forward(xc, params)": "self.gates[exp_i].forward(xc, expert_params)",
        "self.downs[exp_i].forward(a, params)": "self.downs[exp_i].forward(a, expert_params)",
    }
    for old, new in changes.items():
        if source.count(old) != 1:
            raise ValueError(f"Ambiguous converter patch: {old}")
        source = source.replace(old, new)
    compile(source, str(path), "exec")
    path.write_text(source)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(mode: str, minimum: int = 256) -> None:
    if mode not in MODES or minimum < 1:
        raise ValueError("Invalid mode or minimum expert coverage")
    import json
    import os

    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.linear import Linear
    from exllamav3.modules.quant.exl3_lib import quantize

    if getattr(BlockSparseMLP, "_mlxl3_calibration", False):
        raise RuntimeError("Calibration hooks already installed")
    original_init = BlockSparseMLP.__init__
    original_forward = BlockSparseMLP.forward
    original_capture = Linear.capture_H
    original_finalize = quantize.finalize_capture_H

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for index in range(len(self.ups)):
            for projection, linears in (
                ("gate", self.gates),
                ("up", self.ups),
                ("down", self.downs),
            ):
                if linears:
                    linears[index].qmap = expert_qmap(self.key, index, projection)

    def forward(self, x, params):
        if "capture" in params:
            params = {**params, "activate_all_experts": False, "mlxl3_moe_mode": mode}
        return original_forward(self, x, params)

    def capture(self, x, params):
        if "mlxl3_expert_weights" in params:
            return capture_weighted(self, x, params)
        return original_capture(self, x, params)

    def finalize(data, quant_args, verbose):
        if ".experts." in data["first_key"] and not data["finalized"]:
            effective = check_coverage(
                data["count"], data.get("affinity_mass", 0), data.get("affinity_mass2", 0), minimum
            )
            report = {
                "key": data["first_key"],
                "count": data["count"],
                "effective_rows": effective,
                "mode": mode,
                "minimum": minimum,
            }
            if destination := os.environ.get("MLXL3_MOE_COVERAGE"):
                with Path(destination).open("a") as handle:
                    handle.write(json.dumps(report, allow_nan=False) + "\n")
        result = original_finalize(data, quant_args, verbose)
        if ".experts." in data["first_key"] and result[0]:
            raise ValueError("Refusing an uncalibrated expert fallback")
        return result

    BlockSparseMLP.__init__ = init
    BlockSparseMLP.forward = forward
    BlockSparseMLP._mlxl3_calibration = True
    Linear.capture_H = capture
    quantize.finalize_capture_H = finalize


if __name__ == "__main__":
    import importlib.metadata
    import os
    import runpy

    from quantize_lfm25_modal import check_converter_sources

    package = Path(importlib.metadata.distribution("exllamav3").locate_file("exllamav3"))
    check_converter_sources(package)
    patch_converter(package)
    install(os.environ["MLXL3_MOE_CALIBRATION"], int(os.environ.get("MLXL3_MOE_MINIMUM", "256")))
    runpy.run_module("exllamav3.conversion.convert_model", run_name="__main__")
