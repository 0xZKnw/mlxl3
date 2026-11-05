"""Capture actual LDLQ products in an exited MLX process, then replay native FP32.

Requires the coordinated exclusive physical GPU slot. This is a numerical
component test; full advanced-CLI LDLQ parity is a separate subsequent gate.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "apps/MLXL3Studio/Vendor/PonyExl3"
KEY = "model.layers.0.mlp.gate_proj"


def oracle(root):
    from importlib.metadata import version

    import mlx.core as mx
    from ponyexl3.cli import convert_advanced
    from ponyexl3.convert import hessian

    calls, searches = [], []
    original_matmul = mx.matmul
    original_search = hessian.quantize_inner_matrix_direct_mlx

    def product(left, right, *args, **kwargs):
        output = original_matmul(left, right, *args, **kwargs)
        mx.eval(left, right, output)
        name = f"product-{len(calls):02d}.npz"
        arrays = [np.asarray(value).copy() for value in [left, right, output]]
        assert all(value.dtype == np.float32 and np.isfinite(value).all() for value in arrays)
        np.savez(root / name, left=arrays[0], right=arrays[1], output=arrays[2])
        calls.append({"path": name, "shape": [*left.shape, right.shape[1]]})
        return output

    def search(rows, *args, **kwargs):
        packed, reconstructed = original_search(rows, *args, **kwargs)
        mx.eval(rows, packed, reconstructed)
        name = f"search-{len(searches):02d}.npz"
        np.savez(
            root / name,
            rows=np.asarray(rows),
            packed=np.asarray(packed),
            decoded=np.asarray(reconstructed),
        )
        searches.append(name)
        return packed, reconstructed

    mx.matmul = product
    hessian.quantize_inner_matrix_direct_mlx = search
    sys.argv = [
        "convert_advanced",
        "--in-dir",
        str(root / "source"),
        "--out-dir",
        str(root / "oracle-checkpoint"),
        "--ldlq-layer",
        "--only-module",
        KEY,
        "--scale-mode",
        "identity",
        "--search-backend",
        "metal",
        "--codebook",
        "mcg",
        "--buf-size-rows",
        "32",
        "--calibration-activations",
        str(root / "activations.npy"),
        "--json",
    ]
    try:
        assert convert_advanced.main() == 0
        if len(calls) != 7 or len(searches) != 8:
            raise ValueError("incomplete LDLQ observer capture")
        (root / "oracle.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "mlx": version("mlx"),
                    "products": calls,
                    "searches": searches,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        mx.matmul = original_matmul
        hessian.quantize_inner_matrix_direct_mlx = original_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-child", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--activations", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--oracle-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    if args.oracle_child:
        oracle(args.oracle_child.absolute())
        return
    if not all([args.engine, args.source, args.activations, args.baseline, args.output]):
        parser.error("engine, source, activations, baseline and output are required")
    root = args.output.absolute()
    root.mkdir(parents=True, exist_ok=False)
    report_path = root / "report.json"
    report = {
        "status": "in_progress",
        "parity": False,
        "cases": [],
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        runtime = subprocess.run(
            [str(args.engine.absolute()), "runtime-info"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        report["runtime_info"] = json.loads(runtime.stdout)
        if (
            report["runtime_info"].get("backend") != "metal"
            or report["runtime_info"].get("mlx_enabled") is not False
        ):
            raise ValueError("feedback candidate is not the autonomous Metal backend")
        shutil.copytree(args.source, root / "source")
        shutil.copyfile(args.activations, root / "activations.npy")
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(VENDOR)])}
        process = subprocess.run(
            [
                str(args.oracle_python.absolute()),
                str(Path(__file__).absolute()),
                "--oracle-child",
                str(root),
            ],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        (root / "oracle.stdout").write_text(process.stdout)
        (root / "oracle.stderr").write_text(process.stderr)
        if process.returncode:
            raise RuntimeError(f"LDLQ oracle exited {process.returncode}")
        from physical_native_converter_host import compare_checkpoints

        report["observer_baseline_words"] = compare_checkpoints(
            args.baseline, root / "oracle-checkpoint"
        )
        reference = json.loads((root / "oracle.json").read_text())
        if reference.get("status") != "complete" or len(reference["products"]) != 7:
            raise ValueError("incomplete actual-product corpus")
        from mlxl3_quantizer.feedback import matmul_native

        # Oracle has physically exited before the first native Metal command.
        for case in reference["products"]:
            with np.load(root / case["path"], allow_pickle=False) as data:
                left, right, expected = data["left"], data["right"], data["output"]
                actual = matmul_native(left, right, engine=args.engine)
                if (
                    actual.shape != expected.shape
                    or expected.size == 0
                    or not np.isfinite(expected).all()
                ):
                    raise ValueError("invalid product reference")
                unequal = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
                np.save(root / case["path"].replace(".npz", "-native.npy"), actual)
                report["cases"].append(
                    {
                        "path": case["path"],
                        "shape": case["shape"],
                        "words": actual.size,
                        "unequal": unequal,
                        "max_abs": float(np.max(np.abs(actual - expected))),
                    }
                )
                report_path.write_text(json.dumps(report, indent=2) + "\n")
        if any(case["unequal"] for case in report["cases"]):
            raise ValueError("native FP32 product diverged")
        report.update(status="verified", parity=True)
    except BaseException as exc:
        report.update(status="failed", parity=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
