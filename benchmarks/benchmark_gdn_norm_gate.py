"""Microfilter for RMSNorm128 + precise SiLU, preserving FP16 boundaries.

The reduction follows MLX v0.32.2 rms_single_row (Apple, MIT); the gate-table
idea is informed by MTPLX gdn_gated_norm.py (Apache-2.0). This is an EXL3/FP16
SiLU adaptation, not the upstream BF16 sigmoid gate. Run alone on Apple GPU.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

SOURCE = """
uint lane = thread_index_in_simdgroup;
uint row = threadgroup_position_in_grid.x;
uint base = row * 128u + lane * 4u;
float values[4];
float acc = 0.0f;
for (uint i = 0u; i < 4u; ++i) {
    values[i] = float(x[base + i]);
    acc += values[i] * values[i];
}
acc = simd_sum(acc);
threadgroup float sums[32];
threadgroup float inv[1];
sums[lane] = lane == 0u ? acc : 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
acc = simd_sum(sums[lane]);
if (lane == 0u) inv[0] = metal::precise::rsqrt(acc / 128.0f + eps);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = 0u; i < 4u; ++i) {
    half scaled = half(values[i] * inv[0]);
    half normed = weight[lane * 4u + i] * scaled;
    float silu = silu_table[as_type<ushort>(gate[base + i])];
    out[base + i] = half(silu * float(normed));
}
"""


def run(iterations):
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(2709)
    silu = mx.compile(lambda z: z * mx.sigmoid(z), shapeless=True)
    bits = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16)
    table = silu(bits.view(mx.float16).astype(mx.float32))
    mx.eval(table)
    kernel = mx.fast.metal_kernel(
        name="mlxl3_norm_gate128_probe",
        input_names=["x", "gate", "weight", "silu_table", "eps"],
        output_names=["out"],
        source=SOURCE,
    )

    def reference(x, gate, w, eps):
        norm = mx.fast.rms_norm(x, w, eps)
        return (silu(gate.astype(mx.float32)) * norm.astype(mx.float32)).astype(mx.float16)

    def fused(x, gate, w, eps):
        return kernel(
            inputs=[x, gate, w, table, float(eps)],
            output_shapes=[x.shape],
            output_dtypes=[mx.float16],
            grid=(x.size // 4, 1, 1),
            threadgroup=(32, 1, 1),
        )[0]

    checks = []
    for rows in (48, 144, 384, 512):
        for scale in (0.0, 1e-4, 0.2, 100.0, 10000.0):
            for eps in (1e-6, 1e-5):
                x = mx.array((rng.normal(size=(rows, 128)) * scale).astype(np.float16))
                # The 512-row fixture includes every finite FP16 gate pattern.
                gbits = np.arange(rows * 128, dtype=np.uint16).reshape(rows, 128)
                gates = gbits.view(np.float16).copy()
                gates[~np.isfinite(gates)] = 0
                gate = mx.array(gates)
                # Keep exhaustive finite gates below FP16 output overflow.
                w = mx.array(rng.uniform(0.03125, 0.0625, 128).astype(np.float16))
                a, b = reference(x, gate, w, eps), fused(x, gate, w, eps)
                mx.eval(a, b)
                actual, expected = np.asarray(b), np.asarray(a)
                np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
                assert np.isfinite(actual).all()
                checks.append({"rows": rows, "scale": scale, "eps": eps, "fp16_bit_exact": True})
    timings = []
    for rows in (48, 144, 384):
        x = mx.array(rng.normal(0, 0.2, (rows, 128)).astype(np.float16))
        gate = mx.array(rng.normal(0, 2, (rows, 128)).astype(np.float16))
        w = mx.array(rng.uniform(0.25, 1.25, 128).astype(np.float16))
        funcs = [
            lambda x=x, gate=gate, w=w: reference(x, gate, w, 1e-6),
            lambda x=x, gate=gate, w=w: fused(x, gate, w, 1e-6),
        ]
        for _ in range(4):
            for fn in funcs:
                mx.eval(fn())
        samples = [[], []]
        for repeat in range(iterations):
            for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
                start = time.perf_counter_ns()
                mx.eval(funcs[index]())
                samples[index].append((time.perf_counter_ns() - start) / 1e6)
        timings.append(
            {
                "rows": rows,
                "medians_ms": [statistics.median(s) for s in samples],
                "samples_ms": samples,
            }
        )
    return {
        "checks": checks,
        "timings": timings,
        "table_bytes": 262144,
        "timing_boundary": "host+MLX+GPU synchronized",
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    if args.output.exists() or args.iterations <= 0:
        parser.error("use a new output path and positive iterations")
    try:
        result = run(args.iterations)
        if not result["checks"] or not result["timings"]:
            raise RuntimeError("empty comparison cannot validate parity")
    except BaseException as error:
        args.output.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "parity": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                indent=2,
            )
            + "\n"
        )
        raise
    result.update(status="complete", parity=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "checks": len(result["checks"]),
                "timings": [
                    {k: v for k, v in item.items() if k != "samples_ms"}
                    for item in result["timings"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
