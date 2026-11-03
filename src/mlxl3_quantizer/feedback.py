"""Owned FP32 converter matrix products, with an explicit native engine."""

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from .process import run_native


def matrix_values(m: int, n: int, k: int) -> int:
    """Validate the standalone product protocol before allocating/writing files.

    raises: ValueError
    post: __return__ == m * n
    post: 0 < __return__ <= 67108864
    post: m * k <= 67108864 and k * n <= 67108864
    """
    if min(m, n, k) <= 0 or max(m, n, k) > 0xFFFFFFFF or max(m * k, k * n, m * n) > 67108864:
        raise ValueError("invalid converter matrix dimensions")
    return m * n


def matmul_native(left, right, *, engine, timeout=300.0, cancel_check=None, full_fp32=False):
    """Return FP32 storage with reference NAX/TF32 or explicit full precision."""
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("converter matrix shape mismatch")
    m, k = left.shape
    _, n = right.shape
    count = matrix_values(m, n, k)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("converter matrix inputs must be finite")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("conversion deadline must be positive and finite")
    if type(full_fp32) is not bool:
        raise ValueError("full_fp32 must be a boolean")
    executable = Path(engine).expanduser().resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("native engine must be executable")
    if cancel_check is not None:
        cancel_check()
    with tempfile.TemporaryDirectory(prefix="mlxl3-native-matrix-") as directory:
        root = Path(directory)
        for name, values in [("left", left), ("right", right)]:
            (root / f"{name}.f32").write_bytes(np.asarray(values, dtype="<f4").tobytes())
        output = root / "result"
        response = run_native(
            [
                str(executable),
                "convert-matmul",
                str(root / "left.f32"),
                str(root / "right.f32"),
                str(output),
                "--m",
                str(m),
                "--n",
                str(n),
                "--k",
                str(k),
            ]
            + (["--full-fp32"] if full_fp32 else []),
            root=root,
            timeout=timeout,
            cancel_check=cancel_check,
            label="matrix product",
        )
        report = json.loads(response)
        arithmetic = report.get("arithmetic") if isinstance(report, dict) else None
        expected = {
            "schema": 1,
            "status": "complete",
            "m": m,
            "n": n,
            "k": k,
            "values": count,
            "arithmetic": arithmetic,
        }
        manifest = output / "report.json"
        payload = output / "output.f32"
        if (
            output.is_symlink()
            or not output.is_dir()
            or not isinstance(report, dict)
            or arithmetic not in {"nax-tf32", "simdgroup-fp32"}
            or (full_fp32 and arithmetic != "simdgroup-fp32")
            or report != expected
            or any(type(report[key]) is not int for key in ["schema", "m", "n", "k", "values"])
            or manifest.is_symlink()
            or not manifest.is_file()
            or manifest.stat().st_size > 65536
            or json.loads(manifest.read_text()) != report
            or {p.name for p in output.iterdir()} != {"report.json", "output.f32"}
            or payload.is_symlink()
            or not payload.is_file()
            or payload.stat().st_size != count * 4
        ):
            raise ValueError("incomplete or foreign native matrix result")
        values = np.fromfile(payload, dtype="<f4").reshape(m, n)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite native matrix result")
        return values


_CONVERTER_MATRIX_VALUES = 67108864


def matrix_blocks(m: int, n: int, k: int, limit: int) -> tuple[int, int]:
    """Bound each product without splitting its reduction dimension.

    raises: ValueError
    post: 0 < __return__[0] <= m and 0 < __return__[1] <= n
    post: __return__[0] * k <= limit and __return__[1] * k <= limit
    post: __return__[0] * __return__[1] <= limit
    """
    if min(m, n, k, limit) <= 0 or max(m, n, k) > 0xFFFFFFFF or limit > 67108864 or k > limit:
        raise ValueError("invalid converter matrix transport budget")
    cols = min(n, limit // k, limit // min(m, 16, limit))
    if cols < n and cols >= 32:
        cols = cols // 32 * 32
    rows = min(m, limit // k, limit // cols)
    if rows < m and rows >= 16:
        rows = rows // 16 * 16
    return rows, cols


def converter_matmul(left, right):
    """Use bounded row/column slices with one unchanged K accumulation each."""
    from ponyexl3.convert.cancel import raise_if_requested

    from .native import _CONVERTER_LOCK

    engine = os.environ.get("MLXL3_NATIVE_ENGINE")
    if not engine:
        raise RuntimeError("native conversion requires MLXL3_NATIVE_ENGINE")
    raise_if_requested()
    left, right = np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise RuntimeError("native converter matrix shape mismatch")
    m, k = left.shape
    _, n = right.shape
    try:
        rows, cols = matrix_blocks(m, n, k, _CONVERTER_MATRIX_VALUES)
        result = np.empty((m, n), dtype=np.float32)
        with _CONVERTER_LOCK:
            for r in range(0, m, rows):
                for c in range(0, n, cols):
                    raise_if_requested()
                    result[r : r + rows, c : c + cols] = matmul_native(
                        left[r : r + rows],
                        right[:, c : c + cols],
                        engine=engine,
                        cancel_check=raise_if_requested,
                    )
        return result
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"native converter matrix service failed: {exc}") from exc
