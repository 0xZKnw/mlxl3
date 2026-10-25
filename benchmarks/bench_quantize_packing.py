"""Alternating, warmed packing benchmarks; never quantizes or writes a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np


def checked_hash(value, expected):
    value = np.asarray(value)
    if value.size == 0 or value.shape != expected.shape or value.dtype != expected.dtype:
        raise ValueError("Invalid packing result shape/dtype/count")
    if not np.array_equal(value, expected):
        raise ValueError("Packing parity failed")
    return hashlib.sha256(value.tobytes()).hexdigest()


def run(output, *, repeats=7, iterations=20, tiles=(96, 768), bits=(2, 3, 4, 5, 6, 8)):
    if repeats < 1 or iterations < 1 or not tiles or not bits:
        raise ValueError("Benchmark needs positive repeats/iterations and nonempty cases")
    if any(count < 1 for count in tiles) or any(not 1 <= k <= 8 for k in bits):
        raise ValueError("Benchmark needs positive tile counts and K in [1, 8]")
    import mlx.core as mx
    from ponyexl3.convert.mlx_trellis import pack_trellis_mlx as reference
    from ponyexl3.ref.trellis import pack_trellis

    from mlxl3_quantizer import pack_trellis_mlx

    report = {
        "status": "in_progress",
        "parity": None,
        "cases": [],
        "repeats": repeats,
        "iterations": iterations,
        "warmup": 3,
        "scope": "packing only; host dispatch + GPU synchronization, not full conversion",
        "device": mx.device_info(),
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ("mlx", "numpy")},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    try:
        for k in bits:
            for count in tiles:
                values = np.random.default_rng(1000 + k).integers(
                    0, 65536, (count, 256), dtype=np.uint16
                )
                expected = pack_trellis(values.reshape(count, 1, 256), k).reshape(count, 16 * k)
                data = mx.array(values)
                mx.eval(data)
                operations = {"reference": reference, "candidate": pack_trellis_mlx}
                case = {"k": k, "tiles": count, "runs": [], "status": "in_progress"}
                report["cases"].append(case)
                save()
                for operation in operations.values():
                    for _ in range(3):
                        result = operation(data, k)
                        mx.eval(result)
                        checked_hash(result, expected)
                for pair in range(repeats):
                    order = (
                        ("reference", "candidate") if pair % 2 == 0 else ("candidate", "reference")
                    )
                    for name in order:
                        mx.reset_peak_memory()
                        started = time.perf_counter_ns()
                        for _ in range(iterations):
                            result = operations[name](data, k)
                            mx.eval(result)
                        elapsed = (time.perf_counter_ns() - started) / 1e9 / iterations
                        if elapsed <= 0 or not np.isfinite(elapsed):
                            raise ValueError("Invalid elapsed time")
                        case["runs"].append(
                            {
                                "pair": pair,
                                "name": name,
                                "seconds": elapsed,
                                "hash": checked_hash(result, expected),
                                "peak_mlx_bytes": mx.get_peak_memory(),
                            }
                        )
                        save()
                medians = {
                    name: float(
                        np.median([row["seconds"] for row in case["runs"] if row["name"] == name])
                    )
                    for name in operations
                }
                case.update(
                    status="complete",
                    medians_s=medians,
                    speedup=medians["reference"] / medians["candidate"],
                )
                save()
                print(f"K={k} tiles={count}: {case['speedup']:.3f}x", flush=True)
        report.update(status="complete", parity=True)
    except BaseException as exc:
        report.update(status="failed", parity=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1 or args.iterations < 1:
        parser.error("repeats and iterations must be positive")
    run(args.output, repeats=args.repeats, iterations=args.iterations)


if __name__ == "__main__":
    main()
