"""Shape-diverse benchmark gate for MLXL3 decode and small-batch kernels."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlxl3.codec.codebook import CodebookMode
from mlxl3.kernels.qmv import qmm_exl3, qmv_exl3


@dataclass(frozen=True)
class Case:
    input_dims: int
    output_dims: int
    k: int
    mode: CodebookMode
    rows: int = 1

    @property
    def name(self) -> str:
        return (
            f"m{self.rows}_{self.input_dims}x{self.output_dims}_k{self.k}_{self.mode.name.lower()}"
        )


QUICK_CASES = (
    Case(1024, 1024, 3, CodebookMode.DEFAULT),
    Case(1024, 6144, 3, CodebookMode.MUL1),
    Case(2048, 2048, 5, CodebookMode.MCG),
    Case(2048, 6144, 5, CodebookMode.MCG),
    Case(3584, 1024, 3, CodebookMode.DEFAULT),
    Case(2048, 1792, 3, CodebookMode.MCG, rows=8),
    Case(2048, 2048, 5, CodebookMode.MCG, rows=32),
)

FULL_CASES = QUICK_CASES + (
    Case(1024, 4096, 2, CodebookMode.DEFAULT),
    Case(1024, 4096, 4, CodebookMode.DEFAULT),
    Case(2048, 2048, 6, CodebookMode.MCG),
    Case(4096, 4096, 4, CodebookMode.MUL1),
    Case(2048, 128000, 6, CodebookMode.MCG),
    Case(1024, 4096, 3, CodebookMode.DEFAULT, rows=2),
    Case(1024, 4096, 3, CodebookMode.DEFAULT, rows=16),
    Case(1024, 4096, 3, CodebookMode.DEFAULT, rows=64),
)


def _arrays(case: Case):
    rng = np.random.default_rng(81000 + zlib.crc32(case.name.encode()) % 10000)
    trellis = mx.array(
        rng.integers(
            0,
            65536,
            size=(case.input_dims // 16, case.output_dims // 16, 16 * case.k),
            dtype=np.uint16,
        )
    )
    x = mx.array(rng.standard_normal((case.rows, case.input_dims)).astype(np.float16))
    suh = mx.array(rng.choice((-1.0, 1.0), case.input_dims).astype(np.float16))
    svh = mx.array(rng.choice((-1.0, 1.0), case.output_dims).astype(np.float16))
    mx.eval(trellis, x, suh, svh)
    return x, trellis, suh, svh


def _operation(case: Case):
    x, trellis, suh, svh = _arrays(case)
    if case.rows == 1:
        return lambda: qmv_exl3(x, trellis, suh, svh, case.k, case.mode)
    return lambda: qmm_exl3(x, trellis, suh, svh, case.k, case.mode)


def measure(case: Case, *, warmup: int, iterations: int) -> dict:
    operation = _operation(case)
    for _ in range(warmup):
        mx.eval(operation())
    samples = []
    checksum = 0.0
    for _ in range(iterations):
        started = time.perf_counter_ns()
        output = operation()
        mx.eval(output)
        samples.append((time.perf_counter_ns() - started) / 1e6)
        checksum = float(output.reshape(-1)[0].item())
    ordered = sorted(samples)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "case": case.name,
        "input_dims": case.input_dims,
        "output_dims": case.output_dims,
        "k": case.k,
        "mode": case.mode.name.lower(),
        "rows": case.rows,
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[p95_index],
        "min_ms": min(samples),
        "checksum": checksum,
    }


def compare(results: list[dict], baseline_path: Path) -> dict:
    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = {result["case"]: result for result in baseline_payload["results"]}
    ratios = []
    cases = []
    for result in results:
        previous = baseline.get(result["case"])
        if previous is None:
            continue
        ratio = previous["median_ms"] / result["median_ms"]
        ratios.append(ratio)
        cases.append(
            {
                "case": result["case"],
                "baseline_ms": previous["median_ms"],
                "current_ms": result["median_ms"],
                "speedup": ratio,
            }
        )
    return {
        "median_speedup": statistics.median(ratios) if ratios else None,
        "worst_speedup": min(ratios) if ratios else None,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("quick", "full"), default="quick")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=31)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    cases = QUICK_CASES if args.suite == "quick" else FULL_CASES

    results = []
    for case in cases:
        result = measure(case, warmup=args.warmup, iterations=args.iterations)
        results.append(result)
        print(
            f"{case.name:<42} median {result['median_ms']:>8.3f} ms  "
            f"p95 {result['p95_ms']:>8.3f} ms"
        )

    payload = {
        "schema": 1,
        "suite": args.suite,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "system": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "mlx_device": mx.device_info(),
        },
        "cases": [
            {**asdict(case), "mode": case.mode.name.lower(), "name": case.name} for case in cases
        ],
        "results": results,
    }
    if args.compare:
        payload["comparison"] = compare(results, args.compare)
        comparison = payload["comparison"]
        print(
            f"median speedup {comparison['median_speedup']:.3f}x, "
            f"worst {comparison['worst_speedup']:.3f}x"
        )
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
