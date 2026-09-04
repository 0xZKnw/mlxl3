"""Isolated d=512 decode microbenchmark, not a model tok/s benchmark."""

import json
import statistics
import time
from functools import partial

import mlx.core as mx

from mlxl3.kernels.attention import grouped_sdpa_decode_512, matrix_sdpa_decode_512


def main():
    mx.random.seed(7)
    for length in (128, 1024, 4096, 16384):
        q = (mx.random.normal((1, 16, 1, 512)) * 0.1).astype(mx.float16)
        k = mx.random.normal((1, 2, length, 512)).astype(mx.float16)
        v = mx.random.normal(k.shape).astype(mx.float16)
        mx.eval(q, k, v)
        functions = {
            "mlx": partial(mx.fast.scaled_dot_product_attention, q, k, v, scale=1.0),
            "grouped": partial(grouped_sdpa_decode_512, q, k, v),
            "matrix": partial(matrix_sdpa_decode_512, q, k, v),
        }
        timings = {name: [] for name in functions}
        expected = functions["mlx"]()
        errors = {}
        for name, fn in functions.items():
            actual = fn()
            mx.eval(actual, expected)
            errors[name] = float(
                mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))).item()
            )
            for _ in range(3):
                mx.eval(fn())
        for pair in range(5):
            names = list(functions)
            if pair % 2:
                names.reverse()
            for name in names:
                mx.synchronize()
                start = time.perf_counter()
                for _ in range(40):
                    mx.eval(functions[name]())
                timings[name].append((time.perf_counter() - start) * 1e6 / 40)
        print(
            json.dumps(
                {
                    "length": length,
                    "median_us": {name: statistics.median(t) for name, t in timings.items()},
                    "max_abs_error": errors,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
