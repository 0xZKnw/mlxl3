"""Compare the direct Metal CLI to the stock H128 on a physical Apple GPU."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np


def check(cli: Path, report: dict) -> None:
    rng = np.random.default_rng(75)
    cases = []
    packets = []
    refs = []

    def add(name, x, scales, packet=None):
        rows, width = x.shape
        assert scales.shape in ((1, width), (rows, width))
        # Preserve FP16 products and the reference kernel’s two rounding boundaries.
        product = mx.array(np.ascontiguousarray(x)).astype(mx.float16) * mx.array(
            np.ascontiguousarray(scales)
        )
        rotated = mx.hadamard_transform(product.reshape(-1, 128), scale=1.0 / (128**0.5))
        ref = np.array(rotated).reshape(x.shape)
        assert ref.dtype == np.float16 and np.isfinite(ref).all() and ref.size == x.size
        cases.append(name)
        refs.append(ref.view(np.uint16).ravel())
        packets.append(
            packet
            or {
                "op": "metal-prepare128",
                "cols": width,
                "x": x.view(np.uint16).ravel().tolist(),
                "data": scales.view(np.uint16).ravel().tolist(),
            }
        )

    basis = np.eye(128, dtype=np.float16)
    add("all-128-basis-rows", basis, np.ones((1, 128), dtype=np.float16))
    for rows in (1, 3, 20):
        for width in (128, 640, 2560):
            x = rng.normal(0, 1.125, (rows, width)).astype(np.float16)
            for scale_rows in sorted({1, rows}):
                scales = rng.normal(0, 0.75, (scale_rows, width)).astype(np.float16)
                add(f"mixed-{rows}x{width}-scale-rows-{scale_rows}", x, scales)
    bits = np.array([0x0000, 0x8000, 1, 0x8001, 0x3C00, 0xBC00, 0x0400, 0x8400], dtype=np.uint16)
    special = np.resize(bits, (3, 128)).view(np.float16)
    special[1] = np.float16(-0.0)
    special[2] = np.resize(bits[::-1], 128).view(np.float16)
    add("subnormal-signed-zero", special, np.ones_like(special))
    # Rank-two NumPy slicing builds the independent reference without deriving indices
    # from the child's layout. Guard words surround each backing allocation.
    for width in (128, 640, 2560):
        rows = 3
        for dtype in (np.float16, np.float32):
            base = rng.normal(0, 0.5, (2 * width + 2, rows + 2)).astype(dtype)
            scales = rng.normal(0, 0.75, (rows + 2, width + 2)).astype(np.float16)
            for name, x, sv in (
                ("transpose-reverse", base[1:-1:2, -2:0:-1].T, scales[1:-1, -2:0:-1]),
                ("negative-row-col", base[-2:0:-2, -2:0:-1].T, scales[-2:0:-1, 1:-1]),
                (
                    "row-broadcast",
                    np.broadcast_to(base[1:-1:2, 1], (rows, width)),
                    scales[1:-1, 1:-1],
                ),
                (
                    "column-broadcast",
                    np.broadcast_to(base[1, 1:-1, None], (rows, width)),
                    np.broadcast_to(scales[1, 1:-1], (rows, width)),
                ),
            ):

                def layout(view, backing):
                    return {
                        "shape": list(view.shape),
                        "strides": [v // view.itemsize for v in view.strides],
                        "offset": (view.ctypes.data - backing.ctypes.data) // view.itemsize,
                    }

                field = "x" if dtype == np.float16 else "x_f32"
                raw = base.view(np.uint16) if dtype == np.float16 else base
                add(
                    f"view-{name}-{width}-{dtype.__name__}",
                    x,
                    sv,
                    {
                        "op": "metal-prepare-view",
                        field: raw.ravel().tolist(),
                        "x_layout": layout(x, base),
                        "scale_layout": layout(sv, scales),
                        "data": scales.view(np.uint16).ravel().tolist(),
                    },
                )
    invalid = [
        {"op": "metal-prepare128", "cols": 127, "x": [0] * 127, "data": [0] * 127},
        {"op": "metal-prepare128", "cols": 128, "x": [0] * 128, "data": [0] * 127},
        {"op": "metal-prepare128", "cols": 128, "x": [0] * 127, "data": [0] * 128},
    ]
    empty = {"op": "metal-prepare128", "cols": 128, "x": [], "data": []}
    errors = ["expected complete 128-aligned rows"] * len(invalid)
    view = {"shape": [1, 128], "strides": [0, 0], "offset": 0}
    zero = {
        "op": "metal-prepare-view",
        "x": [0],
        "data": [0],
        "x_layout": view,
        "scale_layout": view,
    }
    for fields, error in [
        ({"x_layout": {**view, "strides": [0, -1]}}, "outside initialized storage"),
        ({"scale_layout": {**view, "strides": [0, 2**63 - 1]}}, "outside initialized storage"),
        ({"x_layout": {**view, "offset": -1}}, "outside initialized storage"),
        ({"x_layout": {**view, "shape": [2, 128]}}, "invalid H128 view shape"),
        ({"x_f32": [0]}, "conflicting input buffer dtypes"),
        ({"x_layout": None}, "missing input matrix layout"),
        ({"scale_layout": None}, "missing scale matrix layout"),
        (
            {"x_layout": {**view, "shape": [1, 127]}, "scale_layout": {**view, "shape": [1, 127]}},
            "128-aligned",
        ),
        (
            {
                "x_layout": {**view, "shape": [2**31, 128]},
                "scale_layout": {**view, "shape": [2**31, 128]},
            },
            "size overflow",
        ),
    ]:
        invalid.append({**zero, **fields})
        errors.append(error)
    one_past = {**view, "shape": [0, 128], "offset": 1}
    empties = [empty, {**zero, "x_layout": one_past, "scale_layout": one_past}]
    stream = [*invalid, *empties, *packets]
    child = subprocess.run(
        [cli, "codec"],
        input="".join(json.dumps(p) + "\n" for p in stream),
        env={**os.environ, "DYLD_PRINT_LIBRARIES": "1"},
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    images = [line.lower() for line in child.stderr.splitlines() if line.startswith("dyld[")]
    assert any("/metal.framework/" in image for image in images)
    assert not any(
        marker in image
        for image in images
        for marker in ("/libmlx", "/mlx.framework", "/site-packages/mlx/")
    )
    results = [json.loads(line) for line in child.stdout.splitlines()]
    assert len(results) == len(stream)
    for error, result in zip(errors, results[: len(invalid)], strict=True):
        assert error in result.get("error", ""), result
    assert results[len(invalid) : len(invalid) + len(empties)] == [{"data": []}] * len(empties)
    report.update(
        cases={},
        rejected=len(invalid),
        dyld_images=len(images),
        mlx_loaded=False,
        cli_sha256=hashlib.sha256(cli.read_bytes()).hexdigest(),
        oracle_version=importlib.metadata.version("mlx"),
        finite_values=0,
    )
    for name, ref, result in zip(cases, refs, results[len(invalid) + len(empties) :], strict=True):
        raw = result["data"]
        assert len(raw) == len(ref) and all(type(v) is int and 0 <= v <= 65535 for v in raw)
        actual = np.array(raw, dtype=np.uint16)
        bad = np.flatnonzero(actual != ref)
        if bad.size:
            i = int(bad[0])
            raise AssertionError(
                f"{name}[{i}] expected {int(ref[i]):04x} got {int(actual[i]):04x}, {bad.size}/{len(ref)} mismatches"
            )
        report["cases"][name] = {
            "values": int(ref.size),
            "f16le_sha256": hashlib.sha256(ref.astype("<u2").tobytes()).hexdigest(),
        }
        report["finite_values"] += int(ref.size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    doc = {"status": "started", "seed": 75}
    with args.report.open("x") as out:
        try:
            check(args.cli.resolve(), doc)
            doc["status"] = "passed"
        except BaseException as exc:
            doc.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            json.dump(doc, out, indent=2)
            out.write("\n")
