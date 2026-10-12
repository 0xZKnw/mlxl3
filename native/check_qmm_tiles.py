"""Compare two native QMM implementations in bits, including grouped strides.

The binaries run small synthetic projections, sequentially, without a model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_json_process import JsonProcess, positive_seconds


def validate_output(data: list[int] | list[list[int]], rows: int, widths: list[int]) -> list[int]:
    """Validate the codec's flat FP16 words for every requested projection.

    raises: ValueError
    post: len(__return__) == rows * sum(widths) > 0
    post: all(type(word) is int and 0 <= word < 65536 and (word & 0x7C00) != 0x7C00 for word in __return__)
    """
    if type(rows) is not int or rows <= 0 or not widths:
        raise ValueError("output dimensions must be positive")
    if any(type(width) is not int or width <= 0 for width in widths):
        raise ValueError("output widths must be positive integers")
    if not isinstance(data, list) or not data:
        raise ValueError("output data must be a nonempty list")
    parts = [data] if len(widths) == 1 else data
    if len(parts) != len(widths):
        raise ValueError("output must contain one group per projection")
    flat = []
    for part, width in zip(parts, widths):
        if not isinstance(part, list) or len(part) != rows * width:
            raise ValueError(f"output must contain exactly {rows * width} words per projection")
        for word in part:
            if type(word) is not int or not 0 <= word < 65536:
                raise ValueError("output words must be uint16 integers")
            if (word & 0x7C00) == 0x7C00:
                raise ValueError("output FP16 words must be finite")
            flat.append(word)
    return flat


def main():
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request-timeout", type=positive_seconds, default=120.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be a new file; preserve previous checks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(6405)
    cases = [
        (k, cb, rows, 4096, [128]) for k in range(1, 9) for cb in range(3) for rows in [128, 256]
    ]
    cases += [(k, 2, rows, 4096, [128, 256]) for k in [1, 2, 3, 4, 8] for rows in [128, 192]]
    cases += [
        (2, 2, rows, dims, [128])
        for rows, dims in [
            (24, 4096),
            (127, 4096),
            (129, 4096),
            (64, 4096),
            (128, 2048),
            (256, 2048),
        ]
    ]
    results = []
    report = {
        "status": "running",
        "parity": None,
        "cases": results,
        "request_timeout": args.request_timeout,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        with ExitStack() as stack:
            processes = [
                stack.enter_context(JsonProcess([str(binary.resolve()), "codec"]))
                for binary in [args.baseline, args.candidate]
            ]
            for k, cb, rows, dims, widths in cases:
                data = np.concatenate(
                    [
                        rng.integers(
                            0, 65536, size=(dims // 16 * width // 16 * 16 * k), dtype=np.uint16
                        )
                        for width in widths
                    ]
                )
                x = rng.normal(0, 0.2, size=rows * dims).astype(np.float16).view(np.uint16)
                suh = (
                    rng.uniform(0.5, 1.5, size=dims * len(widths))
                    .astype(np.float16)
                    .view(np.uint16)
                )
                svh = rng.uniform(0.5, 1.5, size=sum(widths)).astype(np.float16).view(np.uint16)
                request = {
                    "op": "mlx-linear" if len(widths) == 1 else "mlx-group",
                    "k": k,
                    "mode": cb,
                    "data": data.tolist(),
                    "x": x.tolist(),
                    "suh": suh.tolist(),
                    "svh": svh.tolist(),
                    "cols": widths[0],
                    "widths": widths,
                }
                outputs = []
                for process in processes:
                    deadline = time.monotonic() + args.request_timeout
                    process.send(request, deadline)
                    value = process.receive(deadline)
                    if "error" in value:
                        raise RuntimeError(value["error"])
                    if "data" not in value:
                        raise RuntimeError("codec output has no data")
                    outputs.append(validate_output(value["data"], rows, widths))
                if outputs[0] != outputs[1]:
                    report["parity"] = False
                    raise AssertionError(
                        f"QMM parity differs at K{k}/CB{cb}/M{rows}/{dims}/{widths}"
                    )
                result = {
                    "k": k,
                    "cb": cb,
                    "rows": rows,
                    "input": dims,
                    "widths": widths,
                    "finite_fp16_outputs": len(outputs[1]),
                    "bit_exact": True,
                }
                results.append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
        report.update(status="complete", parity=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(results)} native QMM cases: all FP16 outputs finite and bit-identical")


if __name__ == "__main__":
    main()
