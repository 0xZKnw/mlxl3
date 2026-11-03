"""Check each finite half gate against the compiled stock Flash SwiGLU on Apple Silicon."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np


def verify(cli, report):
    words = np.arange(65536, dtype=np.uint16)
    words = words[(words & 0x7C00) != 0x7C00]
    gate = words.view(np.float16)
    stock = mx.compile(lambda g, u: (g * mx.sigmoid(g)) * u, shapeless=True)
    up_words = (0, 0x8000, 0x03FF, 0x3555, 0xB800, 0x3C00)
    report.update(stock=importlib.metadata.version("mlx"), gates=len(words), cases=[], values=0)
    for bits in up_words:
        up = np.full(words.shape, bits, dtype=np.uint16)
        expected = np.asarray(stock(mx.array(gate), mx.array(up.view(np.float16)))).view(np.uint16)
        assert expected.shape == words.shape and np.isfinite(expected.view(np.float16)).all()
        req = {"op": "metal-swiglu", "x": words.tolist(), "data": up.tolist()}
        bad_requests = [
            {"op": "metal-swiglu", "x": [], "data": []},
            {"op": "metal-swiglu", "x": [0], "data": []},
        ]
        child = subprocess.run(
            [cli, "codec"],
            input="".join(json.dumps(v) + "\n" for v in [*bad_requests, req]),
            text=True,
            capture_output=True,
            env={**os.environ, "DYLD_PRINT_LIBRARIES": "1"},
            timeout=45,
            check=True,
        )
        imgs = [s.lower() for s in child.stderr.splitlines() if s.startswith("dyld[")]
        assert any("/metal.framework/" in s for s in imgs) and not any(
            "/libmlx" in s or "/libjaccl" in s for s in imgs
        )
        replies = [json.loads(line) for line in child.stdout.splitlines()]
        assert len(replies) == 3 and all(
            "invalid SwiGLU input extents" in r.get("error", "") for r in replies[:2]
        )
        raw = replies[-1]["data"]
        assert len(raw) == len(words) and all(type(v) is int and 0 <= v <= 65535 for v in raw)
        actual = np.array(raw, dtype=np.uint16)
        bad = np.flatnonzero(actual != expected)
        if bad.size:
            i = bad[0]
            raise ValueError(
                f"up {bits:04x}, gate {int(words[i]):04x}: native {int(actual[i]):04x}, stock {int(expected[i]):04x}, {bad.size}/{len(words)} differ"
            )
        report["cases"].append(
            {
                "up_f16_bits": bits,
                "output_f16le_sha256": hashlib.sha256(expected.astype("<u2").tobytes()).hexdigest(),
            }
        )
        report["values"] += len(words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    doc = {"status": "started", "cli_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest()}
    with args.report.open("x") as output:
        try:
            verify(args.cli, doc)
            doc["status"] = "passed"
        except BaseException as exc:
            doc.update(status="failed", error=str(exc))
            raise
        finally:
            json.dump(doc, output, indent=2)
            output.write("\n")
