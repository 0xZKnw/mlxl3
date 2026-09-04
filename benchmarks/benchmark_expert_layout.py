"""Routed EXL3 tile-layout microbenchmark, identical arithmetic and codewords."""

import json
import statistics
import time

import mlx.core as mx
import numpy as np

from mlxl3.kernels.qmv import _qmv_mapped_tile_kernel


def main():
    for inputs, outputs, bits in ((2048, 512, 2), (2048, 512, 3), (512, 2048, 2)):
        tk, ot, experts, rows = inputs // 16, outputs // 16, 256, 8
        tn = ot * experts
        ntg, sg = 4, (4 if bits == 2 else 8)
        rng = np.random.default_rng(5)
        original = mx.array(rng.integers(0, 2**32, (tk, tn, bits * 8), dtype=np.uint32))
        x = mx.array(rng.normal(size=(rows, inputs)).astype(np.float16))
        routes = mx.array([2, 11, 49, 111, 163, 175, 203, 255], dtype=mx.uint32)
        layouts = {0: original}
        for nt in (2, 4, 8, 16, 32):
            layouts[nt] = mx.contiguous(
                original.reshape(tk, tn // nt, nt, bits * 8).transpose(1, 0, 2, 3)
            )
        mx.eval(x, routes, *layouts.values())
        calls = {}
        for nt, packed in layouts.items():
            kernel = _qmv_mapped_tile_kernel(bits, 1, inputs, tn, ntg, sg, bits == 3, nt)

            def call(kernel=kernel, packed=packed):
                return kernel(
                    inputs=[x.reshape(-1), packed.reshape(-1), routes, routes],
                    template=[
                        ("K", bits),
                        ("CB", 1),
                        ("PACKED_U32", bits * 8),
                        ("INPUT_DIMS", inputs),
                        ("TILES_K", tk),
                        ("TILES_N", tn),
                        ("N_SPLITS", 1),
                        ("LOCAL_OUTPUT_DIMS", rows * outputs),
                        ("IDENTITY_MAP", 0),
                        ("EXPERT_MAP", 1),
                        ("OUTPUT_TILES", ot),
                        ("ROUTING_REPEAT", 1),
                        ("PROJECTION_STRIDE_TILES", 0),
                    ],
                    grid=(rows * ot // ntg * sg * 32, 1, 1),
                    threadgroup=(sg * 32, 1, 1),
                    output_shapes=[(1, rows, outputs)],
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
        for pair in range(7):
            for nt in list(calls) if pair % 2 == 0 else list(calls)[::-1]:
                start = time.perf_counter()
                for _ in range(30):
                    mx.eval(calls[nt]())
                samples[nt].append((time.perf_counter() - start) * 1000 / 30)
        print(
            json.dumps(
                {
                    "shape": [inputs, outputs, bits],
                    "ms": {nt: statistics.median(v) for nt, v in samples.items()},
                    "bit_exact": True,
                }
            ),
            flush=True,
        )
        del layouts, calls, packed, original, expected, actual, call, kernel
        mx.clear_cache()


if __name__ == "__main__":
    main()
