"""Qualify the real shared native model operators against a separate MLX process."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def packed(values, *, strided=False):
    import numpy as np

    values = np.asarray(values)
    rows, cols = values.shape
    offset, col_stride = (3, -2) if strided else (0, 1)
    row_stride = cols * abs(col_stride) + (7 if strided else 0)
    if strided:
        offset += (cols - 1) * 2
    storage = np.full(offset + (rows - 1) * row_stride + cols + 7, 0.125, values.dtype)
    for row in range(rows):
        for col in range(cols):
            storage[offset + row * row_stride + col * col_stride] = values[row, col]
    return {
        "values": storage.astype(np.float32).tolist(),
        "layout": {"shape": [rows, cols], "offset": offset, "strides": [row_stride, col_stride]},
        "dtype": str(values.dtype),
    }


def oracle(output):
    from importlib.metadata import version

    import mlx.core as mx
    import numpy as np

    if version("mlx") != "0.32.2":
        raise RuntimeError("this oracle is pinned to MLX 0.32.2")
    rng = np.random.default_rng(750703)
    cases = []

    def random(shape):
        return (rng.standard_normal(shape) * 0.2).astype(np.float16)

    def add(name, op, x, result, *, weight=None, keys=None, values=None, **params):
        mx.eval(result)
        result = np.asarray(result.astype(mx.float16)).reshape(-1)
        if not result.size or not np.isfinite(result).all():
            raise RuntimeError(f"invalid oracle output for {name}")
        case = {
            "name": name,
            "op": op,
            "x": packed(x, strided=True),
            "weight": None,
            "keys": None,
            "values": None,
            "heads": 1,
            "kv_heads": 1,
            "head_dim": 64,
            "rotary": 64,
            "theta": 10000000.0,
            "position": 0,
            "eps": 1e-5,
            "expected": result.view(np.uint16).tolist(),
        }
        for field, array in (("weight", weight), ("keys", keys), ("values", values)):
            if array is not None:
                case[field] = packed(array, strided=True)
        case.update(params)
        cases.append(case)

    for width in (64, 128, 2048, 2560):
        for rows in (1, 3):
            for dtype in (np.float16, np.float32):
                x = random((rows, width))
                gain = (1 + random((1, width))).astype(dtype)
                add(
                    f"rms-{rows}-{width}-{dtype.__name__}",
                    "rms",
                    x,
                    mx.fast.rms_norm(mx.array(x), mx.array(gain.reshape(-1)), 1e-5),
                    weight=gain,
                )

    for rows in (1, 3):
        for outputs in (1, 32, 64, 256):
            x, weight = random((rows, 2048)), random((outputs, 2048))
            add(
                f"dense-{rows}-2048-{outputs}",
                "dense",
                x,
                mx.array(x) @ mx.array(weight).T,
                weight=weight,
            )

    for dim in (64, 128, 256):
        for count in (1, 4, 31, 32, 33, 257):
            heads, kv_heads = (32, 8) if dim == 64 else (8, 2)
            q = random((1, heads * dim))
            keys, values = random((count, kv_heads * dim)), random((count, kv_heads * dim))
            q_array = mx.array(q).reshape(1, heads, 1, dim)
            k_array = mx.array(keys).reshape(1, count, kv_heads, dim).transpose(0, 2, 1, 3)
            v_array = mx.array(values).reshape(1, count, kv_heads, dim).transpose(0, 2, 1, 3)
            out = mx.fast.scaled_dot_product_attention(q_array, k_array, v_array, scale=dim**-0.5)
            add(
                f"gqa-{dim}-{count}",
                "gqa",
                q,
                out,
                keys=keys,
                values=values,
                heads=heads,
                kv_heads=kv_heads,
                head_dim=dim,
            )

    for dim, rotary in ((64, 64), (128, 64), (256, 64), (256, 256)):
        for position in (0, 1, 4096, 131071):
            x = random((4, dim))
            out = mx.fast.rope(
                mx.array(x).reshape(1, 4, 1, dim),
                dims=rotary,
                traditional=False,
                base=10000000.0,
                scale=1.0,
                offset=position,
            )
            add(
                f"rope-{dim}-{rotary}-{position}",
                "rope",
                x,
                out,
                heads=4,
                head_dim=dim,
                rotary=rotary,
                position=position,
            )
    for rows in (1, 3):
        for outputs in (1, 32, 64, 256):
            x = random((rows, 2048))
            weight = (rng.standard_normal((outputs, 2048)) * 0.2).astype(np.float32)
            add(
                f"dense-f32-{rows}-2048-{outputs}",
                "dense",
                x,
                mx.array(x) @ mx.array(weight).T,
                weight=weight,
                expected_error="FP32-weight GEMM batches" if rows > 1 else None,
            )
    for width in (1, 31, 32, 33, 128, 256, 1024):
        x = random((3, width))
        x[0] = np.resize(np.array([-20, -0.0, 0, 1, 1, 20], np.float16), width)
        add(f"softmax-{width}", "softmax", x, mx.softmax(mx.array(x), axis=-1, precise=True))
        for broadcast in (False, True):
            y = random((1, 1) if broadcast else x.shape)
            for op in ("add", "mul", "gate"):
                a, b = mx.array(x), mx.array(y)
                result = a + b if op == "add" else a * b if op == "mul" else a * mx.sigmoid(b)
                add(f"binary-{op}-{width}-{broadcast}", op, x, result, weight=y)
    output.write_text(json.dumps(cases, allow_nan=False))


def run(binary, output, timeout=120):
    if not binary.is_file() or not 0 < timeout < float("inf"):
        raise ValueError("existing test binary and finite positive timeout required")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "in_progress",
        "parity": False,
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    }

    def save():
        pending = output / "report.json.tmp"
        pending.write_text(json.dumps(report, indent=2) + "\n")
        pending.replace(output / "report.json")

    save()
    try:
        fixture = output / "fixtures.json"
        with (output / "oracle.log").open("w") as log:
            subprocess.run(
                [sys.executable, __file__, "--oracle", str(fixture)],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )
        cases = json.loads(fixture.read_text())
        positive_cases = [case for case in cases if not case.get("expected_error")]
        rejected_cases = len(cases) - len(positive_cases)
        expected_words = sum(len(case["expected"]) for case in positive_cases)
        if len(cases) < 20 or expected_words == 0:
            raise RuntimeError("empty or incomplete oracle")
        env = {**os.environ, "MLXL3_MODEL_OPS_FIXTURE": str(fixture), "DYLD_PRINT_LIBRARIES": "1"}
        with (output / "native.log").open("w") as log:
            subprocess.run(
                [
                    str(binary),
                    "gpu::model_ops::tests::shared_operators_match_the_independent_mlx_fixtures",
                    "--exact",
                    "--ignored",
                    "--nocapture",
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout,
            )
        text = (output / "native.log").read_text()
        if (
            "test result: ok. 1 passed; 0 failed; 0 ignored;" not in text
            or f"model-ops checked {expected_words} finite words" not in text
            or len(re.findall(r"model-op .*: words=\d+, unequal=0", text)) != len(positive_cases)
            or len(re.findall(r"model-op .*: rejected as specified", text)) != rejected_cases
            or re.search(r"lib(?:mlx|jaccl)|mlx\.metallib|site-packages/mlx/", text)
        ):
            raise RuntimeError("incomplete native validation or unexpected MLX dependency")
        report.update(
            status="validated",
            parity=True,
            cases=len(cases),
            rejected_cases=rejected_cases,
            finite_words=expected_words,
            fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        )
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--test-binary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.oracle:
        oracle(args.oracle)
    elif args.test_binary and args.output:
        print(json.dumps(run(args.test_binary.absolute(), args.output.absolute())))
    else:
        parser.error("--test-binary and --output are required")
