"""Opt-in BM16 TensorOps screen; stop at the first final FP16 difference."""

import argparse
import json
import os
from pathlib import Path

from benchmark_smallm import CODEBOOKS, ROOT, Checkpoint, assert_exact, paired, qmv


def run(path, iterations, report):
    import mlx.core as mx
    import numpy as np

    checkpoint = Checkpoint(path)
    prefix = "model.language_model.layers.10.mlp.gate_proj"
    shape = next(item for item in checkpoint.inventory() if prefix in item["prefixes"])
    shape = {**shape, "widths": [17408], "prefixes": [prefix]}
    if shape["input"] != 5120 or shape["k"] != 2 or shape["cb"] != 2:
        raise AssertionError("unexpected TensorOps fixture")
    report.update(headers=checkpoint.headers, shape=shape)
    weights = mx.array(checkpoint.load(prefix + ".trellis").view(np.uint32))
    suh = mx.array(checkpoint.load(prefix + ".suh"))
    svh = mx.array(checkpoint.load(prefix + ".svh"))
    permutation = [0] * 256
    for lane in range(32):
        r, c = lane % 4 * 2, lane // 4
        for cg, col in enumerate((c, c + 8)):
            for ri, row in enumerate((r, r + 1, r + 8, r + 9)):
                permutation[lane * 8 + cg * 4 + ri] = row * 16 + col
    inverse = ",".join(str(v) for v in np.argsort(permutation))
    header = (
        "#include <metal_tensor>\n"
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "using namespace metal; using namespace mpp;\n"
        f"inline float mlxl3_decode_codeword(uint x,int cb) {{x &= 0xffffu; {CODEBOOKS[2]}}}\n"
        f"constant ushort mlxl3_perm_inv[256] = {{{inverse}}};\n"
        "#define BM 16u\n#define BN 32u\n#define BK 16u\n#define K_BITS 2u\n"
        "#define PACKED_U32 16u\n#define INPUT_DIMS 5120u\n#define OUTPUT_DIMS 17408u\n"
        "#define TILES_N 1088u\n#define WEIGHT_TILE_OFFSET 0u\n"
    )
    kernel = mx.fast.metal_kernel(
        name="smallm_tensor_bm16",
        input_names=["xhat", "trellis"],
        output_names=["yhat"],
        source=(ROOT / "native/shaders/_qmm_tensor_kernel.metal").read_text(),
        header=header,
    )
    rng = np.random.default_rng(27603)
    for rows in (2, 3, 4, 6, 8):
        stock = qmv(mx, shape, rows)
        x = mx.array(rng.normal(0, 0.2, (rows, 5120)).astype(np.float16))

        def transform(value, rows=rows):
            return mx.hadamard_transform(
                (value * suh).reshape(rows, 40, 128), scale=1 / 128**0.5
            ).reshape(rows, 5120)

        def reference(value, rows=rows, stock=stock):
            return (
                stock(transform(value), weights, mx.array([0], mx.uint32))
                .sum(axis=1)
                .astype(mx.float16)
                .reshape(rows, 17408)
            )

        def tensor(value, rows=rows, transform=transform):
            padded = mx.concatenate(
                [transform(value), mx.zeros((16 - rows, 5120), mx.float16)], axis=0
            )
            return kernel(
                inputs=[padded, weights],
                output_shapes=[(16, 17408)],
                output_dtypes=[mx.float16],
                grid=(17408 // 32 * 32, 1, 1),
                threadgroup=(32, 1, 1),
            )[0][:rows]

        def final(value, rows=rows):
            return (
                mx.hadamard_transform(value.reshape(rows, 136, 128), scale=1 / 128**0.5).reshape(
                    rows, 17408
                )
                * svh
            )

        inner = [reference(x), tensor(x)]
        outputs = [final(value) for value in inner]
        mx.eval(*inner, *outputs)
        a, b = (np.asarray(value).view(np.uint16) for value in outputs)
        report["checks"].append(
            {
                "m": rows,
                "values": rows * 17408,
                "finite": all(bool(np.isfinite(np.asarray(value)).all()) for value in outputs),
                "different_final_fp16_words": int(np.count_nonzero(a != b)),
                "different_inner_fp16_words": int(
                    np.count_nonzero(
                        np.asarray(inner[0]).view(np.uint16) != np.asarray(inner[1]).view(np.uint16)
                    )
                ),
            }
        )
        assert_exact(*outputs)

        def chain(steps, forward, x=x):
            value = x
            for _ in range(steps):
                value = (mx.tanh(final(forward(value))[:, :5120]) * 0.2).astype(mx.float16)
            return value

        assert_exact(chain(8, reference), chain(8, tensor))
        isolated = paired(
            mx,
            [lambda _, forward=forward, x=x: final(forward(x)) for forward in (reference, tensor)],
            iterations,
        )
        dependent = paired(
            mx,
            [
                lambda steps, forward=forward, chain=chain: chain(steps, forward)
                for forward in (reference, tensor)
            ],
            iterations,
            8,
        )
        report["timings"].append({"m": rows, "isolated": isolated, "dependent": dependent})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args(argv)
    if os.environ.get("MLXL3_EXPERIMENTAL_SMALLM_TENSOR") != "1":
        parser.error(
            "experimental TensorOps is OFF; explicit MLXL3_EXPERIMENTAL_SMALLM_TENSOR=1 required"
        )
    if args.output.exists() or args.iterations <= 0 or not args.checkpoint.is_dir():
        parser.error("new output, positive iterations and checkpoint required")
    report = {"status": "running", "parity": None, "checks": [], "timings": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run(args.checkpoint, args.iterations, report)
        if len(report["checks"]) != 5 or len(report["timings"]) != 5:
            raise AssertionError("incomplete TensorOps screen")
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
