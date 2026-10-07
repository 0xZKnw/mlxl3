"""Screen column-vector batched MLX GEMV against canonical M1 GDN gates.

No GEMM MxK activation matrix is used by the candidate. MLX remains responsible
for Metal dispatch; exactness is measured, not inferred from the shape alone.
"""

import argparse
import json
from pathlib import Path

from benchmark_smallm import Checkpoint, assert_exact, paired


def run(checkpoint_path, iterations, report):
    import mlx.core as mx
    import numpy as np

    checkpoint = Checkpoint(checkpoint_path)
    report["headers"] = checkpoint.headers
    prefixes = sorted(
        name.removesuffix(".weight")
        for name in checkpoint.tensors
        if name.endswith((".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight"))
    )
    if len(prefixes) != 96:
        raise AssertionError("expected all 96 Qwen27B GDN gates")
    rng = np.random.default_rng(27602)
    for prefix in prefixes:
        weight = mx.array(checkpoint.load(prefix + ".weight"))
        if weight.shape != (48, 5120) or weight.dtype != mx.float16:
            raise AssertionError("unexpected GDN gate geometry/dtype")
        for rows in (2, 3, 4, 5, 6, 7, 8) if ".layers.0." in prefix else (3,):
            x = mx.array(rng.normal(0, 0.2, (rows, 5120)).astype(np.float16))

            def serial(value, weight=weight, rows=rows):
                return mx.concatenate(
                    [mx.matmul(value[row : row + 1], weight.T) for row in range(rows)], axis=0
                )

            def column_gemv(value, weight=weight, rows=rows):
                return mx.matmul(weight[None, :, :], value[:, :, None]).reshape(rows, 48)

            outputs = [serial(x), column_gemv(x)]
            mx.eval(*outputs)
            assert_exact(*outputs)
            report["checks"].append(
                {"prefix": prefix, "m": rows, "outputs": rows * 48, "fp16_bit_exact": True}
            )
            if ".layers.0." not in prefix:
                continue

            def chain(steps, forward, x=x, rows=rows):
                value = x
                for _ in range(steps):
                    output = forward(value)
                    value = (mx.tanh(mx.tile(output, (1, 107))[:, :5120]) * 0.2).astype(mx.float16)
                return value.reshape(rows, 5120)

            assert_exact(chain(8, serial), chain(8, column_gemv))
            isolated = paired(
                mx,
                [lambda _, forward=forward, x=x: forward(x) for forward in (serial, column_gemv)],
                iterations,
            )
            dependent = paired(
                mx,
                [
                    lambda steps, forward=forward, chain=chain: chain(steps, forward)
                    for forward in (serial, column_gemv)
                ],
                iterations,
                8,
            )
            report["timings"].append(
                {"prefix": prefix, "m": rows, "isolated": isolated, "dependent": dependent}
            )
            print(
                json.dumps(
                    {
                        "prefix": prefix,
                        "m": rows,
                        "isolated_gain_pct": isolated["paired_gain_pct"],
                        "dependent_gain_pct": dependent["paired_gain_pct"],
                    }
                ),
                flush=True,
            )

    for layer in range(64):
        base = f"model.language_model.layers.{layer}.linear_attn."
        if base + "in_proj_a.weight" not in checkpoint.tensors:
            continue
        weights = [
            mx.array(checkpoint.load(base + name + ".weight"))
            for name in ("in_proj_a", "in_proj_b")
        ]
        combined = mx.concatenate(weights, axis=0)
        mx.eval(combined)
        for rows in (2, 3, 4, 5, 6, 7, 8) if layer == 0 else (3,):
            x = mx.array(rng.normal(0, 0.2, (rows, 5120)).astype(np.float16))

            def serial_pair(value, weights=weights, rows=rows):
                return mx.concatenate(
                    [
                        mx.concatenate(
                            [mx.matmul(value[row : row + 1], weight.T) for row in range(rows)],
                            axis=0,
                        )
                        for weight in weights
                    ],
                    axis=1,
                )

            def paired_columns(value, combined=combined, rows=rows):
                return mx.matmul(combined[None, :, :], value[:, :, None]).reshape(rows, 96)

            outputs = [serial_pair(x), paired_columns(x)]
            mx.eval(*outputs)
            assert_exact(*outputs)
            report["checks"].append(
                {"prefix": base + "a+b", "m": rows, "outputs": rows * 96, "fp16_bit_exact": True}
            )
            if layer == 0:

                def chain(steps, forward, x=x):
                    value = x
                    for _ in range(steps):
                        output = forward(value)
                        value = (mx.tanh(mx.tile(output, (1, 54))[:, :5120]) * 0.2).astype(
                            mx.float16
                        )
                    return value

                assert_exact(chain(8, serial_pair), chain(8, paired_columns))
                isolated = paired(
                    mx,
                    [
                        lambda _, forward=forward, x=x: forward(x)
                        for forward in (serial_pair, paired_columns)
                    ],
                    iterations,
                )
                dependent = paired(
                    mx,
                    [
                        lambda steps, forward=forward, chain=chain: chain(steps, forward)
                        for forward in (serial_pair, paired_columns)
                    ],
                    iterations,
                    8,
                )
                report["timings"].append(
                    {
                        "prefix": base + "a+b",
                        "m": rows,
                        "isolated": isolated,
                        "dependent": dependent,
                    }
                )
                print(
                    json.dumps(
                        {
                            "pair": "a+b",
                            "m": rows,
                            "isolated_gain_pct": isolated["paired_gain_pct"],
                            "dependent_gain_pct": dependent["paired_gain_pct"],
                        }
                    ),
                    flush=True,
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args(argv)
    if args.output.exists() or args.iterations <= 0 or not args.checkpoint.is_dir():
        parser.error("new output, positive iterations and checkpoint directory required")
    report = {"status": "running", "parity": None, "checks": [], "timings": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args.checkpoint, args.iterations, report)
        if len(report["checks"]) != 162 or len(report["timings"]) != 21:
            raise AssertionError("incomplete gate screen")
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
