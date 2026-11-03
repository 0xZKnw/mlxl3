"""Compare native routed fused SwiGLU with stock SwitchGlu, using imposed routes."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from physical_moe_router import run
from safetensors.numpy import save_file


def synthetic(root):
    rng = np.random.default_rng(7599)
    cases = []
    for k in (1, 2, 3, 4, 5, 6, 8):
        for mode in range(3):
            folder = root / f"k{k}-cb{mode}"
            folder.mkdir()
            width = (128, 2048, 8192)[(k + mode) % 3]
            intermediate = 256 if mode == 1 else 128
            tensors, storage = {}, {}
            for expert in range(4):
                for name, source, target in (
                    ("gate_proj", width, intermediate),
                    ("up_proj", width, intermediate),
                    ("down_proj", intermediate, width),
                ):
                    prefix = f"experts.{expert}.{name}"
                    raw = rng.integers(
                        0, 65536, (source // 16, target // 16, k * 16), dtype=np.uint16
                    )
                    tensors[prefix + ".trellis"] = raw if k % 2 else raw.view(np.int16)
                    legacy = mode == 1
                    tensors[prefix + (".su" if legacy else ".suh")] = rng.normal(
                        0.7, 0.1, source
                    ).astype(np.float16)
                    tensors[prefix + (".sv" if legacy else ".svh")] = rng.normal(
                        0.7, 0.1, target
                    ).astype(np.float16)
                    if mode:
                        tensors[prefix + (".mcg" if mode == 1 else ".mul1")] = np.array(
                            [1], dtype=np.int32
                        )
                    storage[prefix] = {"quant_format": "exl3"}
            save_file(tensors, folder / "model.safetensors", metadata={"format": "pt"})
            (folder / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
            (folder / "tokenizer.json").write_text("{}")
            (folder / "quantization_config.json").write_text(
                json.dumps({"quant_method": "exl3", "tensor_storage": storage})
            )
            ids = [3, 0, 1, 3] if (k + mode) % 2 else [2]
            x = rng.normal(0, 0.0125, width).astype(np.float16)
            scores = np.asarray([0.125, 0.25, 0.375, 0.25] if len(ids) == 4 else [1], np.float16)
            cases.append(packet(folder, "experts", 4, x, ids, scores))
    cases.append({**cases[0], "x": [0] * len(cases[0]["x"])})
    return cases


def packet(model, prefix, experts, x, ids, scores):
    return {
        "op": "metal-checkpoint-switch",
        "path": str(model),
        "prefix": prefix,
        "key_heads": experts,
        "x": x.view(np.uint16).tolist(),
        "selected": ids,
        "beta": scores.view(np.uint16).tolist(),
    }


def verify(args, root, report):
    cases = synthetic(root)
    for model, layers, prefix, ids in (
        (
            args.qwen,
            (0, 36),
            "model.language_model.layers.{}.mlp.experts",
            [0, 17, 31, 64, 127, 191, 254, 255],
        ),
        (args.lfm, (2, 38), "model.layers.{}.feed_forward.experts", [0, 17, 31, 63]),
    ):
        if model is None:
            continue
        config = json.loads((model / "config.json").read_text())
        config = config.get("text_config", config)
        width, experts = config["hidden_size"], config["num_experts"]
        x = (((np.arange(width) * 29) % 193 - 96) / 779).astype(np.float16)
        scores = np.arange(1, len(ids) + 1, dtype=np.float32)
        scores = (scores / scores.sum()).astype(np.float16)
        for layer in layers:
            cases.append(packet(model.resolve(), prefix.format(layer), experts, x, ids, scores))
    refs, _ = run(args.stock, [{**p, "op": "mlx-checkpoint-switch"} for p in cases], False)
    first = cases[0]
    invalid = [
        {**first, "selected": []},
        {**first, "selected": [first["key_heads"]] * len(first["selected"])},
        {**first, "beta": []},
        {**first, "x": [0]},
        {**first, "x": first["x"] + [0]},
        {**first, "prefix": "missing"},
    ]
    actual, images = run(args.cli, invalid + cases + [cases[-1]], True)
    assert all(reply.get("error") for reply in actual[: len(invalid)])
    actual = actual[len(invalid) :]
    assert actual[-1] == actual[-2]
    report.update(cases=[], compared=0, unequal=0, mlx_loaded=False, dyld_images=images)
    for request, ref, native in zip(cases, refs, actual[:-1], strict=True):
        assert "data" in ref and "data" in native, (ref, native)
        expected, observed = ref["data"], native["data"]
        assert all(type(v) is int and 0 <= v < 65536 for v in expected + observed)
        a, b = np.asarray(observed, np.uint16), np.asarray(expected, np.uint16)
        assert a.size == b.size == len(request["x"]) > 0 and a.ndim == b.ndim == 1
        assert np.isfinite(a.view(np.float16)).all() and np.isfinite(b.view(np.float16)).all()
        assert np.any(b != 0) or not any(v & 0x7FFF for v in request["x"]), "empty numerical oracle"
        unequal = int(np.count_nonzero(a != b))
        report["compared"] += a.size
        report["unequal"] += unequal
        report["cases"].append(
            {
                "model": Path(request["path"]).name,
                "prefix": request["prefix"],
                "width": a.size,
                "ids": request["selected"],
                "reference_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                "unequal": unequal,
            }
        )
    report["invalid_request_recovery"] = len(invalid)
    assert report["compared"] > 0 and report["unequal"] == 0
    report["status"] = "verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "stock", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--lfm", type=Path)
    parser.add_argument("--portable", action="store_true")
    args = parser.parse_args()
    if args.portable:
        os.environ["MLXL3_DISABLE_TENSOR_OPS"] = "1"
    report = {
        "status": "started",
        "mode": "forced_four_groups" if args.portable else "native_hardware_policy",
        "direct_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
        "stock_sha256": hashlib.sha256(args.stock.read_bytes()).hexdigest(),
    }
    with args.report.open("x") as sink, tempfile.TemporaryDirectory(prefix="mlxl3-mapped-") as temp:
        try:
            verify(args, Path(temp), report)
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            json.dump(report, sink, indent=2)
            sink.write("\n")


if __name__ == "__main__":
    main()
