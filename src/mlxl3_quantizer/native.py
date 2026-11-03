"""Bounded tile conversion through the standalone native engine."""

import json
import math
import os
import subprocess
import tempfile
import time
from operator import index
from pathlib import Path

import numpy as np


def launch_tiles(values: int, k: int, codebook: int, scratch: int) -> int:
    """Validate the native binary protocol's shape and scratch limit.

    raises: ValueError
    post: __return__ > 0
    post: __return__ * 256 <= values
    post: __return__ * 516 * (1 << (16 - k)) <= scratch
    """
    if not (
        2 <= k <= 8
        and 0 <= codebook <= 2
        and 0 < values <= 67108864
        and values % 256 == 0
        and 0 < scratch < 1 << 64
    ):
        raise ValueError("invalid native tile dimensions or budget")
    count = min(values // 256, scratch // (516 * (1 << (16 - k))))
    if count == 0:
        raise ValueError("scratch budget is smaller than one trellis tile")
    return count


def quantize_tiles_native(
    tiles,
    k,
    cb=0,
    *,
    engine,
    max_scratch_bytes=256 * 1024 * 1024,
    timeout=300.0,
    cancel_check=None,
):
    """Return finite FP32 decoded tiles and full uint16 states in kernel order.

    A cancellation callback may raise the converter's existing cancellation
    exception. The child is physically reaped before its temporary files vanish.
    """
    k, cb, scratch = index(k), index(cb), index(max_scratch_bytes)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("conversion deadline must be positive and finite")
    values = np.asarray(tiles, dtype=np.float32)
    if values.ndim == 1 and values.size == 256:
        values = values.reshape(1, 256)
    if (
        values.ndim != 2
        or values.shape[0] == 0
        or values.shape[1] != 256
        or values.nbytes > 256 * 1024 * 1024
        or not np.isfinite(values).all()
    ):
        raise ValueError("invalid native tile shape, values or scratch budget")
    expected_launch = launch_tiles(values.size, k, cb, scratch)
    executable = Path(engine).expanduser().resolve(strict=True)
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("native engine must be executable")
    if cancel_check is not None:
        cancel_check()
    with tempfile.TemporaryDirectory(prefix="mlxl3-native-tiles-") as directory:
        root = Path(directory)
        source, output = root / "tiles.f32", root / "result"
        source.write_bytes(np.asarray(values, dtype="<f4").tobytes())
        with (root / "stdout").open("w+b") as stdout, (root / "stderr").open("w+b") as stderr:
            process = subprocess.Popen(
                [
                    str(executable),
                    "quantize-tiles",
                    str(source),
                    str(output),
                    "--k",
                    str(k),
                    "--codebook",
                    str(cb),
                    "--max-scratch-bytes",
                    str(scratch),
                ],
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    if (
                        max(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size)
                        > 65536
                    ):
                        raise ValueError("oversized native conversion output")
                    if cancel_check is not None:
                        cancel_check()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("native tile conversion timed out")
                    time.sleep(0.01)
                stdout.seek(0)
                response = stdout.read(65537)
                stderr.seek(0)
                error = stderr.read(4096).decode("utf-8", errors="replace")
                if process.returncode != 0:
                    raise RuntimeError(f"native tile conversion failed: {error}")
                if len(response) > 65536:
                    raise ValueError("oversized native tile response")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
        report_file = output / "report.json"
        if (
            report_file.is_symlink()
            or not report_file.is_file()
            or report_file.stat().st_size > 65536
        ):
            raise ValueError("incomplete native tile report")
        report = json.loads(response)
        if (
            not isinstance(report, dict)
            or report.get("schema") != 1
            or report.get("status") != "complete"
            or any(
                type(report.get(field)) is not int
                for field in [
                    "schema",
                    "tiles",
                    "values",
                    "k",
                    "codebook",
                    "max_scratch_bytes",
                    "launch_tiles",
                ]
            )
            or report["launch_tiles"] != expected_launch
            or report.get("tiles") != values.shape[0]
            or report.get("values") != values.size
            or report.get("k") != k
            or report.get("codebook") != cb
            or report.get("max_scratch_bytes") != scratch
            or {p.name for p in output.iterdir()}
            != {"decoded.f32", "states.u16", "packed.u16", "report.json"}
            or json.loads(report_file.read_text()) != report
        ):
            raise ValueError("incomplete or foreign native tile response")
        for name, size in [
            ("decoded.f32", values.size * 4),
            ("states.u16", values.size * 2),
            ("packed.u16", values.shape[0] * 16 * k * 2),
        ]:
            file = output / name
            if file.is_symlink() or not file.is_file() or file.stat().st_size != size:
                raise ValueError("incomplete native tile payload")
        decoded = np.fromfile(output / "decoded.f32", dtype="<f4").reshape(values.shape)
        states = np.fromfile(output / "states.u16", dtype="<u2").reshape(values.shape)
        if not np.isfinite(decoded).all():
            raise ValueError("nonfinite native tile output")
        return decoded, states
