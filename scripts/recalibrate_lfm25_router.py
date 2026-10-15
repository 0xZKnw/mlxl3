"""Fit and export the preregistered LFM25-38 router correction; never modify V2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

LAYERS = tuple(range(2, 40))
FEATURES = 2048
EXPERTS = 64
SAMPLES = 32768
MIN_SAMPLES = 2048
TOP_K = 20
ALPHAS = {"alpha-025": 0.25, "alpha-050": 0.5, "alpha-100": 1.0}
GATE_PATTERN = re.compile(r"model\.layers\.\d+\.feed_forward\.gate\.weight")


def checked_alpha(alpha: float) -> float:
    """Validate a convex interpolation, including the identity control.

    raises: ValueError
    post: 0.0 <= __return__ <= 1.0
    post: __return__ == alpha
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and between zero and one")
    return float(alpha)


def enough_samples(count: int, minimum: int) -> bool:
    """Whether an expert has the required independent calibration rows.

    raises: ValueError
    post: count >= minimum or not __return__
    post: count < minimum or __return__
    """
    if type(count) is not int or type(minimum) is not int or count < 0 or minimum <= 0:
        raise ValueError("sample counts must be nonnegative with a positive minimum")
    return count >= minimum


def _finite(tensor, label: str) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise ValueError(f"{label} must be a floating tensor")
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{label} contains nonfinite values")


def fit_delta(x, zq, zsource, bias, min_samples=2048, rho=0.1, top_k=20):
    """Return (delta [experts, features] F64, coverage/training-error statistics).

    Scores and hidden states are paired, actual runtime captures. This is a local
    linear proxy; its errors are not model KL/PPL or a CUDA parity claim.
    """
    import torch

    for tensor, label in ((x, "x"), (zq, "zq"), (zsource, "zsource"), (bias, "bias")):
        _finite(tensor, label)
    enough_samples(0, min_samples)
    if (
        isinstance(rho, bool)
        or not isinstance(rho, (int, float))
        or not math.isfinite(rho)
        or rho <= 0
    ):
        raise ValueError("rho must be finite and positive")
    if x.ndim != 2 or min(x.shape) <= 0 or zq.ndim != 2 or zsource.shape != zq.shape:
        raise ValueError("expected nonempty x[N,D] and matched zq/zsource[N,E]")
    n, d = x.shape
    if zq.shape[0] != n or zq.shape[1] <= 0 or bias.shape != (zq.shape[1],):
        raise ValueError("router captures/bias do not match x rows and expert count")
    e = zq.shape[1]
    if type(top_k) is not int or not 1 <= top_k <= e:
        raise ValueError("top_k must be between one and the expert count")
    if any(t.device != x.device for t in (zq, zsource, bias)):
        raise ValueError("all calibration tensors must use the same device")
    with torch.inference_mode():
        xd = x.to(torch.float64)
        residual = zsource.to(torch.float64) - zq.to(torch.float64)
        scores = zsource.to(torch.float64).sigmoid() + bias.to(torch.float64)
        _finite(residual, "residual")
        _finite(scores, "selection scores")
        # Fixed index tie-breaking keeps the mask reproducible across CPU/GPU sorts.
        selected = scores.argsort(dim=1, descending=True, stable=True)[:, :top_k]
        delta = torch.zeros((e, d), dtype=torch.float64, device=x.device)
        rows = []
        for expert in range(e):
            mask = (selected == expert).any(dim=1)
            count = int(mask.sum().item())
            row = {"expert": expert, "samples": count, "status": "cold", "ridge_lambda": 0.0}
            if enough_samples(count, min_samples):
                xe, target = xd[mask], residual[mask, expert]
                h = xe.T @ xe / count
                rhs = xe.T @ target / count
                _finite(h, "H")
                _finite(rhs, "rhs")
                trace = float(h.diagonal().sum().item())
                row["trace"] = trace
                if trace == 0:
                    row["status"] = "zero_energy"
                else:
                    ridge = float(rho) * trace / d
                    if not math.isfinite(ridge) or ridge <= 0:
                        raise ValueError("ridge coefficient is nonfinite or underflowed")
                    h.diagonal().add_(ridge)
                    solution = torch.linalg.solve(h, rhs)
                    _finite(solution, "delta")
                    delta[expert] = solution
                    row.update(
                        status="fit",
                        ridge_lambda=ridge,
                        delta_l2=float(solution.norm().item()),
                        residual_mse_before=float(target.square().mean().item()),
                        residual_mse_after=float((target - xe @ solution).square().mean().item()),
                    )
            if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
                raise ValueError("nonfinite fit statistics")
            rows.append(row)
        _finite(delta, "delta")
    return delta, {
        "status": "complete",
        "samples": n,
        "features": d,
        "experts": e,
        "top_k": top_k,
        "rho": float(rho),
        "min_samples": min_samples,
        "fitted_experts": sum(row["status"] == "fit" for row in rows),
        "per_expert": rows,
        "quality_validation": False,
    }


def alpha_weight(weight, delta, alpha):
    """Interpolate checkpoint-orientation weights, preserving alpha=0 bytes."""
    import torch

    factor = checked_alpha(alpha)
    _finite(weight, "weight")
    _finite(delta, "delta")
    if weight.dtype != torch.float16 or weight.ndim != 2 or weight.shape != delta.shape:
        raise ValueError("expected matching F16 checkpoint weight and delta[E,D]")
    if factor == 0:
        return weight.clone()
    result = (weight.to(torch.float64) + factor * delta.to(weight.device, torch.float64)).half()
    _finite(result, "interpolated F16 weight")
    return result.contiguous()


def tensor_sha(tensor) -> str:
    import torch

    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _save_report(path: Path, report: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _owners(source: Path) -> dict[str, str]:
    from safetensors import safe_open

    owners = {}
    files = sorted(source.glob("model*.safetensors"))
    if not files or set(files) != set(source.glob("*.safetensors")):
        raise ValueError("expected only model safetensors shards")
    for shard in files:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 -- safe_open is not iterable.
                if key in owners:
                    raise ValueError(f"duplicate checkpoint tensor: {key}")
                owners[key] = shard.name
    index = source / "model.safetensors.index.json"
    if index.exists() and json.loads(index.read_text()).get("weight_map") != owners:
        raise ValueError("checkpoint index does not match physical tensor owners")
    return owners


def export_checkpoint(source, destination, overrides) -> dict:
    """Rewrite physical shards: both CUDA and native loaders scan beyond the index."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, destination = Path(source), Path(destination)
    if (
        source.resolve() == destination.resolve()
        or source.resolve() in destination.resolve().parents
    ):
        raise ValueError("destination must be outside the immutable source checkpoint")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("preserve existing export evidence")
    owners = _owners(source)
    gates = {key for key in owners if GATE_PATTERN.fullmatch(key)}
    if not gates or set(overrides) != gates:
        raise ValueError("overrides must cover exactly all existing router gates")
    for key, weight in overrides.items():
        _finite(weight, key)
        if weight.dtype != torch.float16 or weight.ndim != 2:
            raise ValueError("router overrides must remain two-dimensional F16 tensors")
    destination.mkdir(parents=False)
    report = {
        "status": "running",
        "publication": False,
        "non_gate_sha256": {},
        "gate_sha256": {},
        "shards": {},
    }
    report_path = destination / "router-export.json"
    _save_report(report_path, report)
    try:
        for name in sorted(set(owners.values())):
            with safe_open(source / name, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                tensors = {key: handle.get_tensor(key) for key in handle.keys()}  # noqa: SIM118
            expected = {}
            for key, tensor in tensors.items():
                if key in gates:
                    replacement = overrides[key].detach().cpu().contiguous()
                    if replacement.shape != tensor.shape or replacement.dtype != tensor.dtype:
                        raise ValueError(f"router override changed shape/dtype: {key}")
                    tensors[key] = replacement
                expected[key] = (tensors[key].shape, tensors[key].dtype, tensor_sha(tensors[key]))
            save_file(tensors, destination / name, metadata=metadata)
            del tensors
            with safe_open(destination / name, framework="pt", device="cpu") as handle:
                if set(handle.keys()) != set(expected):
                    raise ValueError("export lost or added tensor keys")
                for key, (shape, dtype, digest) in expected.items():
                    actual = handle.get_tensor(key)
                    if (
                        actual.shape != shape
                        or actual.dtype != dtype
                        or tensor_sha(actual) != digest
                    ):
                        raise ValueError(f"export changed tensor bytes: {key}")
                    report["gate_sha256" if key in gates else "non_gate_sha256"][key] = digest
            report["shards"][name] = {
                "sha256": file_sha(destination / name),
                "bytes": (destination / name).stat().st_size,
            }
            _save_report(report_path, report)
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix != ".safetensors":
                if path.name == report_path.name:
                    raise ValueError("source contains conflicting export metadata")
                shutil.copy2(path, destination / path.name)
        if _owners(destination) != owners:
            raise ValueError("exported checkpoint ownership changed")
        report["status"] = "complete"
        _save_report(report_path, report)
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        _save_report(report_path, report)
        raise


def run(source, teacher_dir, quant_dir, output) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    source, teacher_dir, quant_dir, output = map(Path, (source, teacher_dir, quant_dir, output))
    output.mkdir(exist_ok=False)
    report_path = output / "report.json"
    report = {
        "status": "running",
        "publication": False,
        "quality_validation": False,
        "fits": {},
        "candidates": {},
    }
    _save_report(report_path, report)
    try:
        owners = _owners(source)
        keys = {f"model.layers.{i}.feed_forward.gate.weight" for i in LAYERS}
        if {key for key in owners if GATE_PATTERN.fullmatch(key)} != keys:
            raise ValueError("checkpoint must contain exactly the registered 38 gates")
        teacher_parameters = load_file(teacher_dir / "routing-parameters.safetensors", device="cpu")
        quant_parameters = load_file(quant_dir / "routing-parameters.safetensors", device="cpu")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = False
        report["fit_device"] = str(device)
        weights, deltas = {}, {}
        for layer in LAYERS:
            key = f"model.layers.{layer}.feed_forward.gate.weight"
            weight = quant_parameters[key]
            bias = teacher_parameters[key.removesuffix("gate.weight") + "expert_bias"]
            if weight.shape != (EXPERTS, FEATURES) or weight.dtype != torch.float16:
                raise ValueError("captured V2 gate shape/dtype mismatch")
            if bias.shape != (EXPERTS,) or bias.dtype != torch.float32:
                raise ValueError("teacher selection bias must retain F32 precision")
            _finite(weight, "captured V2 gate")
            _finite(bias, "teacher bias")
            with safe_open(source / owners[key], framework="pt", device="cpu") as handle:
                original_weight = handle.get_tensor(key)
                if (
                    original_weight.shape != weight.shape
                    or original_weight.dtype != weight.dtype
                    or tensor_sha(weight) != tensor_sha(original_weight)
                ):
                    raise ValueError("captured V2 gate differs from immutable checkpoint")
            q = load_file(quant_dir / f"layer-{layer}.safetensors", device=str(device))
            teacher = load_file(teacher_dir / f"layer-{layer}.safetensors", device=str(device))
            if set(q) != {"x", "zq"} or set(teacher) != {"zsource"}:
                raise ValueError("unexpected calibration capture keys")
            x, zq, zsource = q["x"], q["zq"], teacher["zsource"]
            if (
                x.shape != (SAMPLES, FEATURES)
                or zq.shape != (SAMPLES, EXPERTS)
                or zsource.shape != zq.shape
                or x.dtype != torch.float16
                or zq.dtype != torch.float16
                or zsource.dtype != torch.bfloat16
            ):
                raise ValueError("calibration captures do not match the registered shapes/dtypes")
            delta, stats = fit_delta(
                x, zq, zsource, bias.to(device), min_samples=MIN_SAMPLES, rho=0.1, top_k=TOP_K
            )
            if tensor_sha(alpha_weight(weight, delta, 0)) != tensor_sha(weight):
                raise ValueError("alpha-zero checkpoint control is not byteexact")
            weights[key], deltas[key] = weight, delta.cpu()
            report["fits"][str(layer)] = stats
            _save_report(report_path, report)
            del q, teacher, x, zq, zsource, delta
        for name, alpha in ALPHAS.items():
            overrides = {
                key: alpha_weight(weights[key], deltas[key], alpha) for key in sorted(keys)
            }
            gate_file = output / f"{name}.gates.safetensors"
            save_file(overrides, gate_file)
            checkpoint = output / name
            exported = export_checkpoint(source, checkpoint, overrides)
            report["candidates"][name] = {
                "alpha": alpha,
                "checkpointdir": str(checkpoint),
                "gatefile": str(gate_file),
                "gatefile_sha256": file_sha(gate_file),
                "gatefile_bytes": gate_file.stat().st_size,
                "non_gate_sha256": exported["non_gate_sha256"],
                "gate_sha256": exported["gate_sha256"],
                "shards": exported["shards"],
            }
            _save_report(report_path, report)
        report["status"] = "complete"
        _save_report(report_path, report)
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        _save_report(report_path, report)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("teacher_dir", type=Path)
    parser.add_argument("quant_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    return run(args.source, args.teacher_dir, args.quant_dir, args.output)


if __name__ == "__main__":
    main()
