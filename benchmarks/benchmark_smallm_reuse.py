"""M2/M4 mapped EXL3 screen; preserves per-output arithmetic and negative results."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from benchmark_smallm import CODEBOOKS, ROOT, Checkpoint, assert_exact, paired, qmv


def shared_decode(source: str) -> str:
    """Hoist only the unique mapped row loop, preserving other decode branches.

    post: __return__.count("fma(") == source.count("fma(")
    """
    marker = (
        "                        for (uint batch_row = 0u; batch_row < MLXL3_QMV_MB; ++batch_row) {\n"
        "                            uint j = group * 4u;\n"
    )
    if source.count(marker) != 1:
        raise ValueError("mapped decode loop changed; inspect the source")
    prefix, body = source.split(marker, 1)
    end = "                        }\n                    }\n#endif\n"
    body, closing, suffix = body.partition(end)
    if not closing or body.count("fma(") != 4 or "{" in body or "}" in body:
        raise ValueError("mapped decode loop changed; inspect the source")
    for i in range(4):
        expression = f"float(mlxl3_decode_codeword(cw{i}, CB)),"
        if body.count(expression) != 1:
            raise ValueError("missing or duplicate mapped decode")
        body = body.replace(expression, f"weight{i},")
    weights = "".join(
        f"                        float weight{i} = float(mlxl3_decode_codeword(cw{i}, CB));\n"
        for i in range(4)
    )
    # GROUPED_ROUTES already hoists its own decoded[4]; leave that path untouched.
    return (
        prefix
        + "#if !MLXL3_GROUPED_ROUTES\n"
        + weights
        + "#endif\n"
        + marker
        + body
        + closing
        + suffix
    )


def lookup_decode(source: str) -> str:
    """Replace the declarations generated here, never an unrelated decode site.

    post: __return__.count("fma(") == source.count("fma(")
    """
    source = shared_decode(source)
    for i in range(4):
        declaration = f"float weight{i} = float(mlxl3_decode_codeword(cw{i}, CB));"
        if source.count(declaration) != 1:
            raise ValueError("missing or duplicate shared decode declaration")
        source = source.replace(
            declaration, f"float weight{i} = float(codebook_table[cw{i} & 0xffffu]);"
        )
    return source


def mul1_table_cpu():
    import numpy as np

    bits = np.arange(65536, dtype=np.uint32) * np.uint32(0x83DCD12D)
    sums = 0x6400 + sum((bits >> shift) & 255 for shift in (0, 8, 16, 24))
    values = sums.astype(np.uint16).view(np.float16).astype(np.float64)
    inv, bias = np.array([0x1EEE, 0xC931], dtype=np.uint16).view(np.float16).astype(np.float64)
    # One final half rounding, independent of the GPU expression and its compiler.
    return (values * inv + bias).astype(np.float16)


def mul1_table_gpu(mx, report):
    cpu = mul1_table_cpu()
    kernel = mx.fast.metal_kernel(
        name="smallm_full_mul1_table",
        input_names=[],
        output_names=["table"],
        header=f"inline float decode(uint x) {{{CODEBOOKS[2]}}}\n",
        source="uint i=thread_position_in_grid.x; if(i<65536u) table[i]=half(decode(i));",
    )
    table = kernel(
        inputs=[],
        output_shapes=[(65536,)],
        output_dtypes=[mx.float16],
        grid=(65536, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
    assert_exact(cpu, table)
    report["table"] = {
        "count": cpu.size,
        "bytes": cpu.nbytes,
        "sha256": hashlib.sha256(cpu.tobytes()).hexdigest(),
        "bit_exact_cpu_gpu": True,
    }
    return table


def speed_gate(samples):
    a, b = samples
    if (
        len(a) != len(b)
        or len(a) < 8
        or len(a) % 2
        or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in a + b)
    ):
        raise ValueError("incomplete or invalid paired timings")
    median = statistics.median
    gains = [x / y - 1 for x, y in zip(a, b, strict=True)]
    half = len(a) // 2
    drift = [median(v[half:]) / median(v[:half]) - 1 for v in (a, b)]
    orders = [median(gains[::2]), median(gains[1::2])]
    wins = sum(g > 0 for g in gains)
    return {
        "gain_percent": median(gains) * 100,
        "wins": wins,
        "drift_percent": [d * 100 for d in drift],
        "AB_BA_gain_percent": [g * 100 for g in orders],
        "passed": median(gains) >= 0.03
        and wins * 5 >= len(a) * 4
        and min(orders) >= 0.03
        and max(abs(d) for d in drift) <= 0.03,
    }


def geometries(rows):
    if type(rows) is not int or rows not in (2, 4):
        raise ValueError("only M2 and M4 are preregistered")
    result = [
        (name, rows, nt)
        for name in ("stock", "shared")
        for nt in ((1, 2, 4) if rows == 2 else (1, 2))
    ]
    return [("stock", 2, 2), *result] if rows == 4 else result


def run(iterations, checkpoint, report, save):
    import mlx.core as mx
    import numpy as np

    mx.set_cache_limit(256 * 1024**2)
    rows = report["rows"]
    stock = (ROOT / "native/shaders/_qmv_mapped_tile_kernel.metal").read_text()
    lookup = report.get("codebook_table", False)
    sources = (
        {"stock": stock, "lookup": lookup_decode(stock)}
        if lookup
        else {"stock": stock, "shared": shared_decode(stock)}
    )
    table = mul1_table_gpu(mx, report) if lookup else None
    report["sources"] = {
        name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()
    }
    checkpoint = Checkpoint(checkpoint)
    report["headers"] = checkpoint.headers
    shapes = [
        s
        for s in checkpoint.inventory()
        if len(s["widths"]) == 2
        and s["input"] == 5120
        and s["cb"] == 2
        and (
            (sum(s["widths"]) == 16384 and s["k"] == 2)
            or (sum(s["widths"]) == 34816 and s["k"] in (1, 2, 3))
        )
    ]
    assert len(shapes) == 4, "requires all four Qwen27 grouped shapes"
    rng = np.random.default_rng(270616)
    for shape in shapes:
        dims, widths = shape["input"], shape["widths"]
        weights = mx.array(
            np.concatenate(
                [checkpoint.load(p + ".trellis").view(np.uint32) for p in shape["prefixes"]],
                axis=1,
            )
        )
        suh = mx.array(np.stack([checkpoint.load(p + ".suh") for p in shape["prefixes"]]))
        svh = mx.array(np.concatenate([checkpoint.load(p + ".svh") for p in shape["prefixes"]]))
        x = mx.array(rng.normal(0, 0.2, (rows, dims)).astype(np.float16))
        sub = mx.array(np.repeat(np.arange(2, dtype=np.uint32), np.array(widths) // 16))

        def transform(value, suh=suh, dims=dims):
            return mx.hadamard_transform(
                (value[:, None, :] * suh).reshape(rows * 2, dims // 128, 128), scale=1 / 128**0.5
            ).reshape(rows, 2, dims)

        def chain(
            steps,
            kernel,
            x=x,
            weights=weights,
            sub=sub,
            widths=widths,
            svh=svh,
            dims=dims,
            transform=transform,
        ):
            value = x
            for _ in range(steps):
                y = kernel(transform(value), weights, sub).sum(axis=1).astype(mx.float16)
                y = (
                    mx.hadamard_transform(
                        y.reshape(rows, sum(widths) // 128, 128), scale=1 / 128**0.5
                    ).reshape(rows, sum(widths))
                    * svh
                )
                value = (mx.tanh(y[:, :dims]) * 0.2).astype(mx.float16)
            return value

        xhat = transform(x)
        oracle = qmv(mx, shape, rows, 1, 2)
        reference, feedback = oracle(xhat, weights, sub), chain(8, oracle)
        mx.eval(reference, feedback)
        variants = [
            (
                name,
                mb,
                nt,
                qmv(
                    mx,
                    shape,
                    rows,
                    mb,
                    nt,
                    source=sources[name],
                    codebook_table=table if name == "lookup" else None,
                ),
            )
            for name, mb, nt in report["configurations"]
        ]
        for name, mb, nt, kernel in variants:
            assert_exact(reference, kernel(xhat, weights, sub))
            assert_exact(feedback, chain(8, kernel))
            report["checks"].append(
                {**shape, "source": name, "mb": mb, "nt": nt, "bit_exact": True}
            )
        if lookup and "mutation" not in report:
            bad = qmv(
                mx,
                shape,
                rows,
                2,
                2,
                source=sources["lookup"],
                codebook_table=mx.array(np.roll(mul1_table_cpu(), 1)),
            )(xhat, weights, sub)
            mx.eval(bad)
            different = int(
                np.count_nonzero(
                    np.array(reference).view(np.uint32) != np.array(bad).view(np.uint32)
                )
            )
            try:
                assert_exact(reference, bad)
            except AssertionError:
                assert different > 0, "mutation did not change finite FP32 bits"
                assert np.isfinite(np.array(bad)).all(), "mutation must remain finite"
                report["mutation"] = {
                    "shift": 1,
                    "different_fp32_words": different,
                    "rejected": True,
                }
            else:
                raise AssertionError("rotated full codebook escaped the independent oracle")
        save()
        baseline = next(k for name, mb, nt, k in variants if (name, mb, nt) == ("stock", 2, 2))
        for name, mb, nt, kernel in variants:
            if (name, mb, nt) == ("stock", 2, 2):
                continue
            entry = {**shape, "source": name, "mb": mb, "nt": nt, "status": "running"}
            report["timings"].append(entry)
            save()
            entry["isolated"] = paired(
                mx,
                [
                    lambda _, k=k, xhat=xhat, weights=weights, sub=sub: k(xhat, weights, sub)
                    for k in (baseline, kernel)
                ],
                iterations,
            )
            save()
            entry["dependent"] = paired(
                mx,
                [lambda steps, k=k, chain=chain: chain(steps, k) for k in (baseline, kernel)],
                iterations,
                8,
            )
            entry["gate"] = speed_gate(entry["dependent"]["samples_ms"])
            entry["status"] = "complete"
            save()
            print(
                json.dumps(
                    {"k": shape["k"], "widths": widths, **entry["gate"], "source": name, "nt": nt}
                ),
                flush=True,
            )
        mx.synchronize()
        mx.clear_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--rows", type=int, choices=(2, 4), default=2)
    parser.add_argument("--codebook-table", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations < 8 or args.iterations % 2 or args.output.exists():
        parser.error("use an even iteration count >=8 and a new output path")
    if args.codebook_table and args.rows != 2:
        parser.error("the full codebook screen is preregistered for M2 only")
    report = {
        "status": "running",
        "parity": False,
        "checks": [],
        "timings": [],
        "iterations": args.iterations,
        "rows": args.rows,
        "configurations": [("stock", 2, 2), ("lookup", 2, 2)]
        if args.codebook_table
        else geometries(args.rows),
        "codebook_table": args.codebook_table,
        "allocator_cache_mib": 256,
        "boundary": "host+MLX+GPU sync; eight-projection feedback proxy, not delivered speed",
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        run(args.iterations, args.checkpoint, report, save)
        cells = len(report["configurations"])
        assert len(report["checks"]) == 4 * cells and len(report["timings"]) == 4 * (cells - 1), (
            "incomplete screen"
        )
        assert all(r["status"] == "complete" for r in report["timings"]), "incomplete timings"
        if args.codebook_table:
            assert report["table"]["count"] == 65536 and report["table"]["bit_exact_cpu_gpu"]
            assert report["mutation"]["rejected"] and report["mutation"]["different_fp32_words"] > 0
        report.update(status="complete", parity=True)
    except BaseException as error:
        report.update(status="failed", parity=False, error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
