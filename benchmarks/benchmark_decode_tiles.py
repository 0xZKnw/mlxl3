"""Compare mono-token EXL3 tile widths without changing reductions.

Run alone on a physical Apple GPU. Times include host submission and MLX/GPU
synchronization. FP32 partials are checked before any FP16 rounding.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def run(iterations: int) -> dict:
    import mlx.core as mx
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    bodies = [
        (
            "uint bits=x*89226354u+64248484u; bits=0x3B603B60u^(bits&0x8FFF8FFFu); "
            "half2 v=as_type<half2>(bits); return float(v.x+v.y);"
        ),
        (
            "uint bits=x*0xCBAC1FEDu; bits=0x3B603B60u^(bits&0x8FFF8FFFu); "
            "half2 v=as_type<half2>(bits); return float(v.x+v.y);"
        ),
        (
            "uint bits=x*0x83DCD12Du; uint pairs=(bits&0x00ff00ffu)+"
            "((bits>>8)&0x00ff00ffu); uint sum=0x6400u+(pairs&0xffffu)+(pairs>>16); "
            "half value=as_type<half>(ushort(sum)); half inv=as_type<half>(ushort(0x1EEE)); "
            "half bias=as_type<half>(ushort(0xC931)); return float(value*inv+bias);"
        ),
    ]
    rng = np.random.default_rng(2708)

    def fixture(kind, dims, widths, k):
        cols = sum(widths)
        x = mx.array(rng.normal(0, 0.2, (len(widths), dims)).astype(np.float16)).reshape(-1)
        w = mx.array(rng.integers(0, 2**32, (dims // 16, cols // 16, k * 8), dtype=np.uint32))
        if kind == "group":
            sub = np.repeat(np.arange(len(widths), dtype=np.uint32), np.array(widths) // 16)
            inputs = [x, w, mx.array([0], mx.uint32), mx.array(sub)]
        else:
            inputs = [x, w, mx.ones((cols,), mx.float16)]
        mx.eval(*inputs)
        return inputs

    def functions(kind, dims, widths, k, cb, sg, splits, inputs):
        cols = sum(widths)
        defs = {
            "MLXL3_QMV_MB": 1,
            "MLXL3_QMV_SG": sg,
            "MLXL3_MATRIX_ROWS": 1,
            "MLXL3_K_BITS": k,
            "MLXL3_FUSE_OUTPUT": 0,
            "K": k,
            "CB": cb,
            "PACKED_U32": k * 8,
            "INPUT_DIMS": dims,
            "TILES_K": dims // 16,
            "TILES_N": cols // 16,
            "N_SPLITS": splits,
            "OUTPUT_DIMS": cols,
            "LOCAL_OUTPUT_DIMS": cols,
            "MLXL3_K3_WINDOW_DECODE": 0,
            "MLXL3_BATCH_ROWS": 0,
            "GROUPS": len(widths),
            "IDENTITY_MAP": 1,
            "EXPERT_MAP": 0,
            "OUTPUT_TILES": cols // 16,
            "ROUTING_REPEAT": 1,
            "PROJECTION_STRIDE_TILES": 0,
        }
        mapped = kind == "group"
        filename = "_qmv_mapped_tile_kernel.metal" if mapped else "_qmv_tile_kernel.metal"
        source = (root / "native/shaders" / filename).read_text()
        result = []
        for nt in (1, 2, 4, 8):
            defs["MLXL3_QMV_NT"] = nt
            header = (
                f"inline float mlxl3_decode_codeword(uint x,int cb) {{x &= 0xffffu; {bodies[cb]}}}\n"
                + "\n".join(f"#define {name} {value}" for name, value in defs.items())
                + "\n"
            )
            kernel = mx.fast.metal_kernel(
                name=f"nt_{kind}_{k}_{cb}_{sg}_{splits}_{dims}_{cols}_{nt}",
                input_names=["xhat", "trellis", "tile_map", "tile_sub"]
                if mapped
                else ["xhat", "trellis", "svh"],
                output_names=["yhat"],
                source=source,
                header=header,
            )

            def call(kernel=kernel, nt=nt):
                return kernel(
                    inputs=inputs,
                    output_shapes=[(splits, cols)],
                    output_dtypes=[mx.float32],
                    grid=(cols // 16 // nt * sg * 32, 1, splits),
                    threadgroup=(sg * 32, 1, 1),
                )[0]

            result.append(call)
        outputs = [f() for f in result]
        mx.eval(*outputs)
        reference = np.asarray(outputs[0]).view(np.uint32)
        assert np.isfinite(np.asarray(outputs[0])).all()
        for output in outputs[1:]:
            np.testing.assert_array_equal(reference, np.asarray(output).view(np.uint32))
        return result

    checks = []
    for kind in ("dense", "group"):
        for k in (1, 2, 3, 4):
            for cb in range(3):
                for sg, splits in ((4, 1), (8, 4)):
                    widths = [128, 256] if kind == "group" else [384]
                    inputs = fixture(kind, 512, widths, k)
                    functions(kind, 512, widths, k, cb, sg, splits, inputs)
                    checks.append(
                        {
                            "kind": kind,
                            "k": k,
                            "cb": cb,
                            "sg": sg,
                            "splits": splits,
                            "fp32_bit_exact": True,
                        }
                    )
    timings = []
    for kind, dims, widths, k, sg, splits in (
        ("dense", 17408, [5120], 2, 8, 4),
        ("group", 5120, [17408, 17408], 2, 4, 1),
        ("group", 5120, [17408, 17408], 1, 8, 1),
        ("group", 5120, [17408, 17408], 3, 8, 1),
        ("group", 5120, [10240, 6144], 2, 4, 1),
        ("dense", 5120, [248320], 3, 8, 1),
    ):
        inputs = fixture(kind, dims, widths, k)
        funcs = functions(kind, dims, widths, k, 2, sg, splits, inputs)
        for _ in range(3):
            for fn in funcs:
                mx.eval(fn())
        samples = [[] for _ in funcs]
        for repeat in range(iterations):
            order = range(4) if repeat % 2 == 0 else range(3, -1, -1)
            for index in order:
                start = time.perf_counter_ns()
                mx.eval(funcs[index]())
                samples[index].append((time.perf_counter_ns() - start) / 1e6)
        timings.append(
            {
                "kind": kind,
                "input": dims,
                "widths": widths,
                "k": k,
                "sg": sg,
                "splits": splits,
                "nt": [1, 2, 4, 8],
                "medians_ms": [statistics.median(s) for s in samples],
                "samples_ms": samples,
                "fp32_bit_exact": True,
            }
        )
    return {"checks": checks, "timings": timings, "timing_boundary": "host+MLX+GPU synchronized"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args(argv)
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if args.output.exists():
        parser.error("preserve the existing result; use a new output path")
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
