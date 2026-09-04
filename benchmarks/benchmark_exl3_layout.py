"""Exact tile-permutation experiment, kernel wall time (not GPU timestamps)."""

import json
import statistics
import time

import mlx.core as mx
import numpy as np

from mlxl3.kernels.qmv import _qmv_tile_kernel


def main():
    for inputs, outputs, bits in ((2048, 248320, 6), (2816, 262144, 6), (2048, 6144, 5)):
        tk, tn = inputs // 16, outputs // 16
        rng = np.random.default_rng(5)
        original = mx.array(rng.integers(0, 2**32, (tk, tn, bits * 8), dtype=np.uint32))
        x = mx.array(rng.normal(size=inputs).astype(np.float16))
        layouts = {0: original}
        for nt in (1, 2, 4, 8, 16, 32):
            layouts[nt] = mx.contiguous(
                original.reshape(tk, tn // nt, nt, bits * 8).transpose(1, 0, 2, 3)
            )
        mx.eval(x, *layouts.values())
        calls = {}
        for nt, packed in layouts.items():
            kernel = _qmv_tile_kernel(bits, 2, inputs, outputs, 2, 8, nt)

            def call(kernel=kernel, packed=packed):
                return kernel(
                    inputs=[x, packed.reshape(-1)],
                    template=[
                        ("K", bits),
                        ("CB", 2),
                        ("PACKED_U32", bits * 8),
                        ("TILES_K", tk),
                        ("TILES_N", tn),
                        ("N_SPLITS", 1),
                        ("OUTPUT_DIMS", outputs),
                    ],
                    grid=(tn // 2 * 256, 1, 1),
                    threadgroup=(256, 1, 1),
                    output_shapes=[(1, outputs)],
                    output_dtypes=[mx.float32],
                )[0]

            calls[nt] = call
        expected = calls[0]()
        mx.eval(expected)
        for nt, call in calls.items():
            actual = call()
            mx.eval(actual)
            assert bool(mx.array_equal(expected, actual).item()), nt
            for _ in range(3):
                mx.eval(call())
        samples = {nt: [] for nt in calls}
        for pair in range(5):
            for nt in list(calls) if pair % 2 == 0 else list(calls)[::-1]:
                start = time.perf_counter()
                for _ in range(10):
                    mx.eval(calls[nt]())
                samples[nt].append((time.perf_counter() - start) * 1000 / 10)
        medians = {nt: statistics.median(v) for nt, v in samples.items()}
        print(
            json.dumps(
                {
                    "shape": [inputs, outputs, bits],
                    "ms": medians,
                    "logical_gb_s": {nt: original.nbytes / 1e6 / ms for nt, ms in medians.items()},
                    "bit_exact": True,
                }
            ),
            flush=True,
        )
        del layouts, calls, packed, original, expected, actual, call, kernel
        mx.clear_cache()


if __name__ == "__main__":
    main()
