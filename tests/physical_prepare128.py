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

    def add(name, x, scales):
        rows, width = x.shape
        assert scales.shape in ((1, width), (rows, width))
        # Preserve FP16 products and the reference kernel’s two rounding boundaries.
        product = mx.array(x) * mx.array(scales)
        rotated = mx.hadamard_transform(product.reshape(-1, 128), scale=1.0 / (128**0.5))
        ref = np.array(rotated).reshape(x.shape)
        assert ref.dtype == np.float16 and np.isfinite(ref).all() and ref.size == x.size
        cases.append(name)
        refs.append(ref.view(np.uint16).ravel())
        packets.append(
            {
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
    invalid = [
        {"op": "metal-prepare128", "cols": 127, "x": [0] * 127, "data": [0] * 127},
        {"op": "metal-prepare128", "cols": 128, "x": [0] * 128, "data": [0] * 127},
        {"op": "metal-prepare128", "cols": 128, "x": [0] * 127, "data": [0] * 128},
    ]
    empty = {"op": "metal-prepare128", "cols": 128, "x": [], "data": []}
    stream = [*invalid, empty, *packets]
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
    for result in results[: len(invalid)]:
        assert "expected complete 128-aligned rows" in result.get("error", ""), result
    assert results[len(invalid)] == {"data": []}
    report.update(
        cases={},
        rejected=len(invalid),
        dyld_images=len(images),
        mlx_loaded=False,
        cli_sha256=hashlib.sha256(cli.read_bytes()).hexdigest(),
        oracle_version=importlib.metadata.version("mlx"),
        finite_values=0,
    )
    for name, ref, result in zip(cases, refs, results[len(invalid) + 1 :], strict=True):
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
