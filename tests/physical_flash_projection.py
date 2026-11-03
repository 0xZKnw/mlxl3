"""Compare a standalone direct-Metal CLI to the stock Flash projection on an Apple GPU."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np


def synthetic(root, rng):
    root.mkdir()
    cases = []
    tensors = {}
    storage = {}
    for k in range(1, 9):
        for mode in range(3):
            prefix = f"layer_k{k}_cb{mode}"
            width = (128, 640, 2560, 4096)[(k + mode) % 4]
            cols = 128 if mode != 2 else 256
            shape = (width // 16, cols // 16, 16 * k)
            data = rng.integers(0, 65536, shape, dtype=np.uint16)
            tensors[prefix + ".trellis"] = data if k % 2 else data.view(np.int16)
            legacy = k % 3 == 0
            for role, size in [
                (("su" if legacy else "suh"), width),
                (("sv" if legacy else "svh"), cols),
            ]:
                factors = rng.normal(0.8, 0.3, size).astype(np.float32 if mode == 0 else np.float16)
                tensors[prefix + "." + role] = (
                    mx.array(factors).astype(mx.bfloat16) if mode == 1 else factors
                )
            if mode:
                tensors[prefix + (".mul1" if mode == 2 else ".mcg")] = np.array([1], np.int32)
            if k % 2:
                tensors[prefix + ".bias"] = rng.normal(0, 0.17, cols).astype(np.float16)
            storage[prefix] = {"quant_format": "exl3"}
            cases.append(make_case(rng, root, prefix, width, 1 if k % 2 else 3))
    for expert, down in (("synthetic_expert", "layer_k8_cb0"), ("bad_expert", "layer_k1_cb0")):
        for projection, src in (
            ("gate_proj", "layer_k4_cb0"),
            ("up_proj", "layer_k3_cb1"),
            ("down_proj", down),
        ):
            dest = expert + "." + projection
            storage[dest] = {"quant_format": "exl3"}
            for name, val in list(tensors.items()):
                if name.startswith(src + "."):
                    tensors[dest + name[len(src) :]] = val
    for batch in (1, 3):
        cases.append(make_case(rng, root, "synthetic_expert", 128, batch, expert=True))
    tensors["four.trellis"] = np.zeros((512, 8, 16), dtype=np.uint16)
    tensors["four.suh"] = np.ones(8192, dtype=np.float16)
    tensors["four.svh"] = np.ones(128, dtype=np.float16)
    storage["four"] = {"quant_format": "exl3"}
    mx.save_safetensors(
        str(root / "model.safetensors"),
        {k: mx.array(v) for k, v in tensors.items()},
        metadata={"format": "pt"},
    )
    (root / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    (root / "tokenizer.json").write_text("{}")
    (root / "quantization_config.json").write_text(
        json.dumps({"quant_method": "exl3", "tensor_storage": storage})
    )
    return cases


def make_case(rng, root, prefix, width, batch, *, expert=False):
    x = rng.normal(0, 0.125, (batch, width)).astype(np.float16)
    return {
        "op": "metal-checkpoint-expert" if expert else "metal-checkpoint-linear",
        "path": str(root),
        "prefix": prefix,
        "x": x.view(np.uint16).ravel().tolist(),
    }


def verify(args, root, report):
    rng = np.random.default_rng(7503)
    cases = synthetic(root / "synthetic", rng)
    if args.checkpoint:
        ckpt = args.checkpoint.resolve()
        for label, width in (("gate_proj", 2560), ("down_proj", 640)):
            prefix = "model.language_model.layers.0.mlp.experts.0." + label
            cases.append(make_case(rng, ckpt, prefix, width, 3))
        for suffix in ("experts.0", "shared_expert"):
            for batch in (1, 3):
                cases.append(
                    make_case(
                        rng,
                        ckpt,
                        "model.language_model.layers.0.mlp." + suffix,
                        2560,
                        batch,
                        expert=True,
                    )
                )
    # The stock projection receives ordinary contiguous FP16 rows. Feed the same
    # values through a reversed, padded backing on the direct path.
    stock_cases = list(cases)
    for case in list(cases):
        if not (
            case["prefix"].endswith(("cb1", "experts.0", "shared_expert", "gate_proj", "down_proj"))
        ):
            continue
        x = np.asarray(case["x"], dtype=np.uint16).view(np.float16)
        rows = 3 if len(x) % 3 == 0 else 1
        width = len(x) // rows
        strided = x.reshape(rows, width)[::-1, ::-1]
        storage = np.pad(strided, ((0, 0), (1, 2)), constant_values=7)
        use_f32 = len(cases) % 2 == 0
        if use_f32:
            # Force the GPU to round the FP32 source before multiplication.
            storage = np.nextafter(storage.astype(np.float32), np.float32(1.0))
        gathered = storage[::-1, -3:0:-1].astype(np.float16)
        assert np.array_equal(gathered.view(np.uint16).ravel(), x.view(np.uint16))
        layout = {"shape": [rows, width], "strides": [-width - 3, -1], "offset": storage.size - 3}
        packet = {**case, "x_layout": layout}
        if use_f32:
            del packet["x"]
            packet["x_f32"] = storage.ravel().tolist()
        else:
            packet["x"] = storage.view(np.uint16).ravel().tolist()
        cases.append(packet)
        stock_cases.append(case)
    (root / "cases.json").write_text(json.dumps(stock_cases))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLXL3_", "DYLD_"))}
    env["MLXL3_PROJECTION_FIXTURES"] = str(root)
    stock = subprocess.run(
        [args.stock, "flash_projection_stock_oracle", "--exact", "--ignored", "--test-threads=1"],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if stock.returncode or "1 passed; 0 failed" not in stock.stdout:
        raise RuntimeError((stock.stdout + stock.stderr)[-2500:])
    expected = json.loads((root / "stock.json").read_bytes())
    assert len(expected) == len(cases)
    invalid = [
        {**cases[0], "x": []},
        {**cases[0], "prefix": "missing"},
        {**cases[0], "prefix": "four"},
        {**cases[0], "x": [1]},
    ]
    errors = [
        "invalid Flash projection batch",
        "missing projection trellis",
        "unsupported projection geometry/split count",
        "input is not a complete row",
    ]
    invalid.append({**cases[0], "op": "metal-checkpoint-expert", "prefix": "bad_expert"})
    errors.append("incompatible Flash expert projection geometry")
    stream = [*invalid, *cases]
    run = subprocess.run(
        [args.cli, "codec"],
        input="".join(json.dumps(v) + "\n" for v in stream),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env={**env, "DYLD_PRINT_LIBRARIES": "1"},
    )
    if run.returncode:
        raise RuntimeError(run.stderr[-1700:])
    libraries = [s.lower() for s in run.stderr.splitlines() if s.startswith("dyld[")]
    assert any("/metal.framework/" in s for s in libraries) and not any(
        key in s
        for s in libraries
        for key in ("/libmlx", "/libjaccl", "/mlx.framework", "/site-packages/mlx/")
    )
    replies = [json.loads(s) for s in run.stdout.splitlines()]
    assert len(replies) == len(stream)
    for message, reply in zip(errors, replies[: len(invalid)], strict=True):
        assert message in reply.get("error", ""), reply
    report.update(
        cases=[],
        values=0,
        dyld_images=len(libraries),
        mlx_loaded=False,
        invalid_recovered=len(invalid),
        lifecycle="three submissions, one output dropped, layer and Metal dropped before reading in reverse",
    )
    for case, ref, row in zip(cases, expected, replies[len(invalid) :], strict=True):
        actual = row["data"]
        assert len(ref) == len(actual) > 0 and all(
            type(v) is int and 0 <= v < 65536 and v & 0x7C00 != 0x7C00 for v in ref
        )
        if actual != ref:
            i = next(i for i, (a, b) in enumerate(zip(actual, ref, strict=True)) if a != b)
            raise AssertionError(
                f"{case['prefix']} at {i}: stock {ref[i]:04x}, direct {actual[i]:04x}, {sum(a != b for a, b in zip(actual, ref, strict=True))}/{len(ref)} differ"
            )
        report["values"] += len(ref)
        report["cases"].append(
            {
                "projection": case["prefix"],
                "outputs": len(ref),
                "input": "view-f32"
                if "x_f32" in case
                else "view-f16"
                if "x_layout" in case
                else "contiguous",
                "f16le_sha256": hashlib.sha256(np.asarray(ref, dtype="<u2").tobytes()).hexdigest(),
            }
        )
    report["status"] = "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", type=Path)
    parser.add_argument("stock", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    state = {
        "status": "started",
        "seed": 7503,
        "cli_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
        "stock_test_sha256": hashlib.sha256(args.stock.read_bytes()).hexdigest(),
    }
    with args.report.open("x") as output, tempfile.TemporaryDirectory() as directory:
        try:
            verify(args, Path(directory), state)
        except BaseException as exc:
            state.update(status="failed", error=str(exc))
            raise
        finally:
            json.dump(state, output, indent=2)
            output.write("\n")
