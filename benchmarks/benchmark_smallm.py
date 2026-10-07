"""Physical small-M EXL3 screen using checkpoint weights and stock shaders.

Times include host submission and synchronization. The feedback chain is a
projection proxy, not a model verification or delivered-token benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import struct
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEBOOKS = [
    (
        "uint bits=x*89226354u+64248484u; bits=0x3B603B60u^(bits&0x8FFF8FFFu); "
        "half2 v=as_type<half2>(bits); return float(v.x+v.y);"
    ),
    (
        "uint bits=x*0xCBAC1FEDu; bits=0x3B603B60u^(bits&0x8FFF8FFFu); "
        "half2 v=as_type<half2>(bits); return float(v.x+v.y);"
    ),
    (
        "uint bits=x*0x83DCD12Du; uint pairs=(bits&0x00ff00ffu)+((bits>>8)&0x00ff00ffu); "
        "uint sum=0x6400u+(pairs&0xffffu)+(pairs>>16); half value=as_type<half>(ushort(sum)); "
        "half inv=as_type<half>(ushort(0x1EEE)); half bias=as_type<half>(ushort(0xC931)); "
        "return float(value*inv+bias);"
    ),
]


class Checkpoint:
    def __init__(self, directory):
        self.tensors = {}
        self.headers = []
        for path in sorted(directory.glob("*.safetensors")):
            with path.open("rb") as stream:
                length = struct.unpack("<Q", stream.read(8))[0]
                if not 0 < length <= min(path.stat().st_size - 8, 32 * 1024 * 1024):
                    raise ValueError("invalid safetensors header length")
                raw = stream.read(length)
                header = json.loads(raw)
            self.headers.append({"file": path.name, "sha256": hashlib.sha256(raw).hexdigest()})
            for name, info in header.items():
                if name != "__metadata__":
                    if name in self.tensors:
                        raise ValueError("duplicate checkpoint tensor")
                    self.tensors[name] = (path, 8 + length, info)
        if not self.tensors:
            raise ValueError("empty checkpoint")

    def load(self, name):
        import numpy as np

        path, offset, info = self.tensors[name]
        dtype = {"F16": "<f2", "I16": "<i2", "I32": "<i4"}[info["dtype"]]
        begin, end = info["data_offsets"]
        if end - begin != int(np.prod(info["shape"])) * np.dtype(dtype).itemsize:
            raise ValueError("invalid tensor byte count")
        if not 0 <= begin <= end <= path.stat().st_size - offset:
            raise ValueError("tensor outside shard")
        return np.array(
            np.memmap(
                path, dtype=dtype, mode="r", offset=offset + begin, shape=tuple(info["shape"])
            )
        )

    def inventory(self):
        records = []
        for name, (_, _, info) in sorted(self.tensors.items()):
            if not name.endswith(".trellis") or not (
                re.match(r"model.language_model.layers.\d+\.", name) or name == "lm_head.trellis"
            ):
                continue
            prefix = name.removesuffix(".trellis")
            shape = info["shape"]
            records.append(
                {
                    "prefix": prefix,
                    "input": shape[0] * 16,
                    "widths": [shape[1] * 16],
                    "k": shape[2] // 16,
                    "cb": 2
                    if prefix + ".mul1" in self.tensors
                    else 1
                    if prefix + ".mcg" in self.tensors
                    else 0,
                    "dtype": info["dtype"],
                }
            )
        by_prefix = {item["prefix"]: item for item in records}
        bundles = []
        consumed = set()
        for item in records:
            prefix = item["prefix"]
            partner = None
            if prefix.endswith(".in_proj_qkv"):
                partner = prefix.removesuffix("in_proj_qkv") + "in_proj_z"
            if prefix.endswith(".gate_proj"):
                partner = prefix.removesuffix("gate_proj") + "up_proj"
            other = by_prefix.get(partner)
            if other and all(item[key] == other[key] for key in ("input", "k", "cb")):
                bundles.append(
                    {
                        **item,
                        "prefixes": [prefix, partner],
                        "widths": item["widths"] + other["widths"],
                    }
                )
                consumed.update((prefix, partner))
        bundles.extend(
            {**item, "prefixes": [item["prefix"]]}
            for item in records
            if item["prefix"] not in consumed
        )
        unique = {}
        for item in bundles:
            key = (item["input"], tuple(item["widths"]), item["k"], item["cb"])
            if key in unique:
                unique[key]["count"] += 1
            else:
                unique[key] = {**item, "count": 1}
        return list(unique.values())


def layout(rows: int, grouped: bool, output: int, input_dims: int, k: int) -> dict:
    """Mirror the current M5 dispatch for an inventory, not a proof of Rust.

    raises: ValueError
    post: __return__["mb"] * __return__["groups"] >= rows
    """
    if not 2 <= rows <= 8 or min(input_dims, output) <= 0 or input_dims % 128 or output % 128:
        raise ValueError("invalid small-M shape")
    tiles = output // 16
    wide = tiles >= 1024
    paired = rows in (2, 4, 6, 8) if grouped else rows in (2, 4, 5, 6, 7, 8)
    mb = 3 if rows == 3 and not grouped and wide else 2 if paired else 1
    nt = 2 if wide else 4 if tiles % 4 == 0 else 2 if tiles % 2 == 0 else 1
    if not grouped and rows >= 8 and tiles % 4 == 0:
        nt = 4
    if mb == 2:
        nt = min(nt, 2)
    if mb == 3:
        nt = 1
    splits = 1
    target = max(tiles // 2, 32) if tiles <= 64 else min(max(tiles, 32), 256)
    while input_dims // 16 // (splits * 2) >= target:
        splits *= 2
    return {
        "mb": mb,
        "nt": nt,
        "groups": (rows + mb - 1) // mb,
        "sg": 4 if grouped and k == 2 else 8,
        "splits": splits,
    }


def assert_exact(left, right):
    import numpy as np

    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.size == 0 or a.dtype != b.dtype:
        raise AssertionError("incomplete or different output shapes/dtypes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("non-finite output")
    if a.tobytes() != b.tobytes():
        raise AssertionError(
            f"different bits: {int(np.count_nonzero(a.view('u1') != b.view('u1')))} bytes"
        )


def qmv(mx, shape, rows, mb=None, nt=None):
    dims, widths, k, cb = (shape[key] for key in ("input", "widths", "k", "cb"))
    cols = sum(widths)
    dispatch = layout(max(2, rows), len(widths) > 1, cols, dims, k)
    if rows == 1:
        dispatch.update(mb=1, groups=1)
    mb = dispatch["mb"] if mb is None else mb
    nt = dispatch["nt"] if nt is None else nt
    sg, splits = dispatch["sg"], dispatch["splits"]
    defs = {
        "MLXL3_QMV_NT": nt,
        "MLXL3_QMV_MB": mb,
        "MLXL3_MATRIX_ROWS": rows,
        "MLXL3_QMV_SG": sg,
        "MLXL3_K_BITS": k,
        "MLXL3_K3_WINDOW_DECODE": 0,
        "MLXL3_BATCH_ROWS": 1,
        "GROUPS": len(widths),
        "K": k,
        "CB": cb,
        "PACKED_U32": k * 8,
        "INPUT_DIMS": dims,
        "TILES_K": dims // 16,
        "TILES_N": cols // 16,
        "N_SPLITS": splits,
        "LOCAL_OUTPUT_DIMS": cols,
        "OUTPUT_DIMS": cols,
        "MLXL3_FUSE_OUTPUT": 0,
        "IDENTITY_MAP": 1,
        "EXPERT_MAP": 0,
        "OUTPUT_TILES": cols // 16,
        "ROUTING_REPEAT": 1,
        "PROJECTION_STRIDE_TILES": 0,
    }
    mapped = len(widths) > 1
    filename = "_qmv_mapped_tile_kernel.metal" if mapped else "_qmv_tile_kernel.metal"
    source = (ROOT / "native/shaders" / filename).read_text()
    header = (
        f"inline float mlxl3_decode_codeword(uint x,int cb) {{x &= 0xffffu; {CODEBOOKS[cb]}}}\n"
    )
    header += "\n".join(f"#define {key} {value}" for key, value in defs.items()) + "\n"
    kernel = mx.fast.metal_kernel(
        name=f"smallm_{dims}_{cols}_{k}_{cb}_{rows}_{mb}_{nt}_{sg}_{splits}",
        input_names=["xhat", "trellis", "tile_map", "tile_sub"]
        if mapped
        else ["xhat", "trellis", "svh"],
        output_names=["yhat"],
        source=source,
        header=header,
    )

    def call(xhat, weights, tile_sub):
        return kernel(
            inputs=[xhat, weights, mx.array([0], mx.uint32), tile_sub]
            if mapped
            else [xhat, weights, mx.ones((cols,), mx.float16)],
            output_shapes=[(rows, splits, cols)],
            output_dtypes=[mx.float32],
            grid=(cols // 16 // nt * sg * 32, (rows + mb - 1) // mb, splits),
            threadgroup=(sg * 32, 1, 1),
        )[0]

    return call


def paired(mx, functions, iterations, steps=1):
    samples = [[], []]
    for _ in range(3):
        for fn in functions:
            mx.eval(fn(steps))
            mx.synchronize()
    for repeat in range(iterations):
        for index in (0, 1) if repeat % 2 == 0 else (1, 0):
            start = time.perf_counter_ns()
            mx.eval(functions[index](steps))
            mx.synchronize()
            samples[index].append((time.perf_counter_ns() - start) / 1e6 / steps)
    medians = [statistics.median(values) for values in samples]
    return {
        "medians_ms": medians,
        "samples_ms": samples,
        "paired_gain_pct": statistics.median([(a / b - 1) * 100 for a, b in zip(*samples)]),
        "steps": steps,
    }


def run(iterations, checkpoint, inventory_only=False, report=None):
    checkpoint = Checkpoint(checkpoint)
    shapes = checkpoint.inventory()
    report = {} if report is None else report
    report.update(headers=checkpoint.headers, inventory=[], checks=[], timings=[])
    for shape in shapes:
        for rows in (2, 3, 4, 6, 8):
            report["inventory"].append(
                {
                    **shape,
                    "m": rows,
                    **layout(
                        rows,
                        len(shape["widths"]) > 1,
                        sum(shape["widths"]),
                        shape["input"],
                        shape["k"],
                    ),
                    "median_ms": None,
                }
            )
    if inventory_only:
        return report
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(270603)
    for k in (1, 2, 3, 4):
        for cb in range(3):
            for dims, widths in ((512, [128, 256]), (2048, [128, 256])):
                shape = {"input": dims, "widths": widths, "k": k, "cb": cb}
                weights = mx.array(
                    rng.integers(0, 2**32, (dims // 16, sum(widths) // 16, k * 8), dtype=np.uint32)
                )
                xhat = mx.array(rng.normal(0, 0.2, (3, 2, dims)).astype(np.float16))
                sub = mx.array(np.repeat(np.arange(2, dtype=np.uint32), np.array(widths) // 16))
                outputs = [
                    qmv(mx, shape, 3, mb, nt)(xhat, weights, sub) for mb, nt in ((1, 4), (3, 1))
                ]
                mx.eval(*outputs)
                assert_exact(*outputs)
                report["checks"].append({**shape, "fp32_bit_exact": True})
    for shape in shapes:
        if len(shape["widths"]) != 2:
            continue
        dims, widths = shape["input"], shape["widths"]
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
        x = mx.array(rng.normal(0, 0.2, (3, dims)).astype(np.float16))
        sub = mx.array(np.repeat(np.arange(2, dtype=np.uint32), np.array(widths) // 16))
        kernels = [qmv(mx, shape, 3), qmv(mx, shape, 3, 3, 1)]

        def transform(value, suh=suh, dims=dims):
            return mx.hadamard_transform(
                (value[:, None, :] * suh).reshape(6, dims // 128, 128), scale=1 / 128**0.5
            ).reshape(3, 2, dims)

        xhat = transform(x)
        partials = [kernel(xhat, weights, sub) for kernel in kernels]
        mx.eval(*partials)
        assert_exact(*partials)

        def forward(
            value, kernel, weights=weights, sub=sub, widths=widths, svh=svh, transform=transform
        ):
            y = kernel(transform(value), weights, sub).sum(axis=1).astype(mx.float16)
            return (
                mx.hadamard_transform(
                    y.reshape(3, sum(widths) // 128, 128), scale=1 / 128**0.5
                ).reshape(3, sum(widths))
                * svh
            )

        def chain(steps, kernel, x=x, dims=dims, forward=forward):
            value = x
            for _ in range(steps):
                y = forward(value, kernel)
                value = (mx.tanh(y[:, :dims]) * 0.2).astype(mx.float16)
            return value

        assert_exact(chain(8, kernels[0]), chain(8, kernels[1]))
        isolated = paired(
            mx,
            [
                lambda _, kernel=kernel, xhat=xhat, weights=weights, sub=sub: kernel(
                    xhat, weights, sub
                )
                for kernel in kernels
            ],
            iterations,
        )
        dependent = paired(
            mx,
            [lambda steps, kernel=kernel, chain=chain: chain(steps, kernel) for kernel in kernels],
            iterations,
            8,
        )
        report["timings"].append(
            {
                **shape,
                "m": 3,
                "baseline": layout(3, True, sum(widths), dims, shape["k"]),
                "candidate": {"mb": 3, "nt": 1},
                "fp32_bit_exact": True,
                "dependent_fp16_bit_exact": True,
                "isolated": isolated,
                "dependent": dependent,
            }
        )
        print(
            json.dumps(
                {
                    "widths": widths,
                    "k": shape["k"],
                    "isolated": isolated["paired_gain_pct"],
                    "dependent": dependent["paired_gain_pct"],
                }
            ),
            flush=True,
        )
    report["timing_boundary"] = (
        "host+MLX+GPU synchronized; dependent is an eight-projection feedback proxy"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args(argv)
    if args.iterations <= 0 or not args.checkpoint.is_dir():
        parser.error("positive iterations and an existing checkpoint directory are required")
    if args.output.exists():
        parser.error("preserve the existing result; use a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "running", "parity": None}
    try:
        run(args.iterations, args.checkpoint, args.inventory_only, result)
        if not result["inventory"] or (
            not args.inventory_only and (not result["checks"] or not result["timings"])
        ):
            raise RuntimeError("empty comparison cannot validate parity")
        result.update(status="complete", parity=None if args.inventory_only else True)
    except BaseException as error:
        result.update(
            status="failed",
            parity=False,
            error={"type": type(error).__name__, "message": str(error)},
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        raise
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
