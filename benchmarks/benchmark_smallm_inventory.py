"""Inventory production small-M geometry, checkpoint weights and measured cost.

The eight-step dependency includes exact checkpoint transforms and scales, with
bounded synthetic feedback. This is not a full-model timing attribution.
"""

import argparse
import json
from pathlib import Path

from benchmark_smallm import Checkpoint, assert_exact, layout, paired, qmv


def run(path, iterations, report):
    import mlx.core as mx
    import numpy as np

    checkpoint = Checkpoint(path)
    report["headers"] = checkpoint.headers
    rng = np.random.default_rng(27600)
    for shape in checkpoint.inventory():
        dims, widths = shape["input"], shape["widths"]
        groups, cols = len(widths), sum(widths)
        weights = mx.array(
            np.concatenate(
                [checkpoint.load(p + ".trellis").view(np.uint32) for p in shape["prefixes"]], axis=1
            )
        )
        suh = mx.array(
            np.stack([checkpoint.load(p + ".suh") for p in shape["prefixes"]]).astype(np.float16)
        )
        svh = mx.array(
            np.concatenate([checkpoint.load(p + ".svh") for p in shape["prefixes"]]).astype(
                np.float16
            )
        )
        sub = mx.array(np.repeat(np.arange(groups, dtype=np.uint32), np.array(widths) // 16))
        for rows in (2, 3, 4, 6, 8):
            x = mx.array(rng.normal(0, 0.2, (rows, dims)).astype(np.float16))
            kernel = qmv(mx, shape, rows)
            single = qmv(mx, shape, 1, 1, 1)

            def transform(value, suh=suh, dims=dims, groups=groups):
                count = value.shape[0]
                return mx.hadamard_transform(
                    (value[:, None, :] * suh).reshape(count * groups, dims // 128, 128),
                    scale=1 / 128**0.5,
                ).reshape(count, groups, dims)

            xhat = transform(x)
            partials = kernel(xhat, weights, sub)
            reference = mx.concatenate(
                [single(xhat[row : row + 1], weights, sub) for row in range(rows)], axis=0
            )
            mx.eval(partials, reference)
            assert_exact(partials, reference)

            def chain(
                steps,
                x=x,
                kernel=kernel,
                transform=transform,
                weights=weights,
                sub=sub,
                svh=svh,
                dims=dims,
                cols=cols,
                rows=rows,
                validate=False,
            ):
                value = x
                for _ in range(steps):
                    yhat = kernel(transform(value), weights, sub).sum(axis=1).astype(mx.float16)
                    output = (
                        mx.hadamard_transform(
                            yhat.reshape(rows, cols // 128, 128), scale=1 / 128**0.5
                        ).reshape(rows, cols)
                        * svh
                    )
                    if validate and not np.isfinite(np.asarray(output)).all():
                        raise AssertionError("non-finite dependent output")
                    value = (
                        mx.tanh(mx.tile(output, (1, (dims + cols - 1) // cols))[:, :dims]) * 0.2
                    ).astype(mx.float16)
                return value

            chain(8, validate=True)
            isolated_fn = lambda _, kernel=kernel, xhat=xhat, weights=weights, sub=sub: kernel(
                xhat, weights, sub
            )
            isolated = paired(mx, [isolated_fn, isolated_fn], iterations)
            dependent = paired(mx, [chain, chain], iterations, 8)
            report["rows"].append(
                {
                    **shape,
                    "m": rows,
                    **layout(rows, groups > 1, cols, dims, shape["k"]),
                    "fp32_bit_exact_to_serial": True,
                    "isolated": isolated,
                    "dependent": dependent,
                }
            )
            print(
                json.dumps(
                    {
                        "input": dims,
                        "widths": widths,
                        "k": shape["k"],
                        "m": rows,
                        "isolated_ms": isolated["medians_ms"][0],
                        "dependent_ms": dependent["medians_ms"][0],
                    }
                ),
                flush=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    if args.output.exists() or args.iterations <= 0 or not args.checkpoint.is_dir():
        parser.error("new output, positive iterations and checkpoint required")
    report = {"status": "running", "parity": None, "rows": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args.checkpoint, args.iterations, report)
        if len(report["rows"]) != 80:
            raise AssertionError("incomplete real-shape inventory")
        report.update(status="complete", parity=True)
    except BaseException as error:
        report.update(
            status="failed",
            parity=False,
            error={"type": type(error).__name__, "message": str(error)},
        )
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
