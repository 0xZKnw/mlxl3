"""Explicit physical test: real advanced converter CLI, MLX oracle then native.

No model is loaded. The two backends run in separate sequential processes.
Run only with the exclusive physical GPU slot; the standard suite excludes this file.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "apps/MLXL3Studio/Vendor/PonyExl3"
KEY = "model.layers.0.mlp.gate_proj"


def compare_checkpoints(oracle: Path, candidate: Path):
    arrays = []
    for folder in [oracle, candidate]:
        files = sorted(folder.glob("*.safetensors"))
        if not files:
            raise ValueError("converter did not emit weights")
        tensors = {}
        for file in files:
            loaded = load_file(file)
            if tensors.keys() & loaded.keys():
                raise ValueError("duplicate converted tensor")
            tensors.update(loaded)
        arrays.append(tensors)
    left, right = arrays
    if left.keys() != right.keys() or KEY + ".trellis" not in left:
        raise ValueError("incomplete converted module")
    for tensors in arrays:
        trellis = tensors[KEY + ".trellis"]
        if trellis.shape != (8, 8, 64) or trellis.dtype != np.dtype("uint16"):
            raise ValueError("invalid converted EXL3 trellis")
    configs = [
        json.loads((path / "quantization_config.json").read_text()) for path in [oracle, candidate]
    ]
    if configs[0] != configs[1] or configs[0].get("quant_method") != "exl3":
        raise ValueError("converted quantization metadata differs")
    total = 0
    for key, expected in left.items():
        actual = right[key]
        if (
            expected.size == 0
            or expected.shape != actual.shape
            or expected.dtype != actual.dtype
            or not np.isfinite(expected).all()
            or not np.isfinite(actual).all()
            or not np.array_equal(expected, actual)
        ):
            raise ValueError(f"native conversion diverged: {key}")
        total += expected.size
    if total <= 0:
        raise ValueError("zero converted values")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--oracle-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--include-ldlq", action="store_true", help="also diagnose the pending LDLQ path"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report_file = args.output / "report.json"
    report = {
        "status": "in_progress",
        "parity": False,
        "engine": str(args.engine.absolute()),
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        "cases": [],
    }
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    source = args.output / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "llama", "hidden_size": 128, "num_hidden_layers": 1})
    )
    rng = np.random.default_rng(7508)
    weights = rng.normal(0, 0.5, (128, 128)).astype(np.float16)
    save_file({KEY + ".weight": weights}, source / "model.safetensors")
    activations = rng.normal(size=(192, 128)).astype(np.float16)
    # Add correlated rows so the LDLQ feedback and cross-buffer product are used.
    activations[:, 1::2] += np.float16(0.5) * activations[:, ::2]
    np.save(args.output / "activations.npy", activations)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(VENDOR)]),
        "MLXL3_NATIVE_ENGINE": str(args.engine.absolute()),
    }
    guard = args.output / "guard"
    guard.mkdir()
    (guard / "sitecustomize.py").write_text(
        "import importlib.abc,sys\nfrom pathlib import Path\n"
        "class Reject(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self,fullname,path,target=None):\n"
        "  if fullname.split('.')[0] in {'mlx','mlx_lm'}:\n"
        f"   Path({str((guard / 'forbidden').absolute())!r}).write_text(fullname)\n"
        "   raise ImportError('native path imported MLX')\n"
        "sys.meta_path.insert(0,Reject())\n"
    )
    try:
        info = subprocess.run(
            [str(args.engine.absolute()), "runtime-info"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        report["runtime_info"] = json.loads(info.stdout)
        if report["runtime_info"].get("mlx_enabled") is not False:
            raise ValueError("physical candidate engine enables MLX")
        modes = ["direct", "ldlq"] if args.include_ldlq else ["direct"]
        for mode in modes:
            for backend in ["metal", "native"]:
                destination = args.output / f"{mode}-{backend}"
                command = [
                    str(args.oracle_python.absolute()) if backend == "metal" else sys.executable,
                    "-m",
                    "ponyexl3.cli.convert_advanced",
                    "--in-dir",
                    str(source.absolute()),
                    "--out-dir",
                    str(destination.absolute()),
                    f"--{mode}-layer",
                    "--only-module",
                    KEY,
                    "--scale-mode",
                    "identity",
                    "--search-backend",
                    backend,
                    "--codebook",
                    "mcg",
                    "--buf-size-rows",
                    "32",
                    "--calibration-activations",
                    str((args.output / "activations.npy").absolute()),
                    "--json",
                ]
                child_env = dict(env)
                if backend == "native":
                    child_env["PYTHONPATH"] = str(guard.absolute()) + os.pathsep + env["PYTHONPATH"]
                result = subprocess.run(
                    command,
                    env=child_env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                (args.output / f"{mode}-{backend}.stdout").write_text(result.stdout)
                (args.output / f"{mode}-{backend}.stderr").write_text(result.stderr)
                if result.returncode:
                    raise ValueError(f"{mode}/{backend} CLI exited {result.returncode}")
                summary = json.loads(result.stdout)
                if (
                    summary.get("shape") != [128, 128]
                    or summary.get("search_backend") != backend
                    or summary.get("k") != 4
                    or (guard / "forbidden").exists()
                ):
                    raise ValueError("incomplete or foreign public converter result")
                print(f"completed {mode}/{backend}", flush=True)
            words = compare_checkpoints(
                args.output / f"{mode}-metal", args.output / f"{mode}-native"
            )
            report["cases"].append({"mode": mode, "words": words, "unequal": 0})
            report_file.write_text(json.dumps(report, indent=2) + "\n")
        if len(report["cases"]) != len(modes):
            raise ValueError("incomplete physical conversion campaign")
        report.update(status="verified", parity=True)
    except BaseException as exc:
        report.update(status="failed", parity=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report_file.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
