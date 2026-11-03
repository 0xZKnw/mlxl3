"""Compare direct Flash gate intermediates and live routes to independent stock matmul/softmax/top-k."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from physical_moe_router import run
from safetensors import safe_open
from safetensors.numpy import save_file


def verify(cli, stock, checkpoint, work, report):
    rng = np.random.default_rng(7583)
    synthetic = work / "synthetic"
    synthetic.mkdir()
    for name, obj in [
        ("config", {"model_type": "qwen4_exp"}),
        ("tokenizer", {}),
        (
            "quantization_config",
            {"quant_method": "exl3", "tensor_storage": {"fixture": {"quant_format": "exl3"}}},
        ),
    ]:
        (synthetic / (name + ".json")).write_text(json.dumps(obj))
    w = rng.normal(0, 0.025, (512, 2560)).astype(np.float16)
    exl3 = {
        "fixture.trellis": np.zeros((1, 1, 16), np.uint16),
        "fixture.suh": np.ones(16, np.float16),
        "fixture.svh": np.ones(16, np.float16),
    }
    save_file(
        {
            **exl3,
            "gate.weight": w,
            "bad.weight": np.zeros((2, 2560), np.float16),
            "bad_dtype.weight": w.astype(np.float32),
            "biased.weight": w,
            "biased.bias": np.zeros(512, np.float16),
        },
        str(synthetic / "model.safetensors"),
        metadata={"format": "pt"},
    )
    weights = [(synthetic, "gate", w)]
    names = json.loads((checkpoint / "model.safetensors.index.json").read_bytes())["weight_map"]
    for prefix in (
        "model.language_model.layers.0.mlp.gate",
        "model.language_model.layers.23.mlp.gate",
        "mtp.layers.0.mlp.gate",
    ):
        name = prefix + ".weight"
        with safe_open(checkpoint / names[name], framework="numpy") as f:
            weights.append((checkpoint, prefix, f.get_tensor(name)))
    # Reopening all five indexed shards for every layout is CPU-heavy in debug.
    # Keep an end-to-end read of each original gate, and replay its exact bytes for permutations.
    subset = work / "subset"
    subset.mkdir()
    for config in ("config.json", "quantization_config.json", "tokenizer.json"):
        (subset / config).write_bytes((synthetic / config).read_bytes())
    save_file(
        {**exl3, **{prefix + ".weight": w for _, prefix, w in weights[1:]}},
        str(subset / "model.safetensors"),
        metadata={"format": "pt"},
    )
    refs, requests, stock_requests = [], [], []
    for path, prefix, w in weights:
        assert w.dtype == np.float16 and w.shape == (512, 2560) and np.isfinite(w).all()
        weight = mx.array(w)
        for dtype in [np.float16, np.float32]:
            unsigned = np.uint16 if dtype == np.float16 else np.uint32
            for kind, batch in [("offset", 1), ("reverse", 3), ("tied", 1)]:
                data = rng.normal(0, 0.25, (2562, batch + 2)).astype(dtype)
                if kind == "tied":
                    data.fill(0)
                matrix = data[-2:0:-1, 1:-1].T
                req = {
                    "op": "metal-checkpoint-router",
                    "path": str(
                        subset
                        if path == checkpoint and (kind != "offset" or dtype != np.float16)
                        else path
                    ),
                    "prefix": prefix,
                    "x_layout": {
                        "shape": [batch, 2560],
                        "offset": (matrix.ctypes.data - data.ctypes.data) // data.itemsize,
                        "strides": [v // data.itemsize for v in matrix.strides],
                    },
                }
                if dtype == np.float16:
                    req["x"] = data.view(unsigned).ravel().tolist()
                else:
                    req["x_f32"] = data.ravel().tolist()
                requests.extend([{**req, "return_raw": True}, req])
                logits = []
                probs = []
                for row in matrix:
                    scores = mx.array(np.ascontiguousarray(row[None, :])) @ weight.T
                    probabilities = mx.softmax(scores, axis=-1, precise=True)
                    logits.append(np.asarray(scores))
                    probs.append(np.asarray(probabilities))
                logits, probs = np.concatenate(logits), np.concatenate(probs)
                assert (
                    logits.dtype == probs.dtype == dtype
                    and logits.shape == probs.shape == (batch, 512)
                    and np.isfinite(logits).all()
                    and np.isfinite(probs).all()
                )
                ref = {
                    "label": f"{prefix}-{dtype.__name__}-{kind}",
                    "weight": hashlib.sha256(w.tobytes()).hexdigest(),
                    "logits": logits.view(unsigned).ravel(),
                    "probabilities": probs.view(unsigned).ravel(),
                    "dtype": dtype,
                }
                stock_requests.append(
                    {
                        "op": "mlx-router",
                        "cols": 512,
                        "k": 10,
                        "normalize": dtype == np.float32,
                        ("data" if dtype == np.float16 else "x_f32"): (
                            probs.view(unsigned) if dtype == np.float16 else probs
                        )
                        .ravel()
                        .tolist(),
                    }
                )
                refs.append(ref)
    first = requests[0]
    invalid = [{**first, "prefix": name} for name in ["missing", "bad", "bad_dtype", "biased"]]
    invalid.append({**first, "x_layout": {**first["x_layout"], "shape": [1, 128]}})
    routing, _ = run(stock, stock_requests, False)
    replies, images = run(cli, [*invalid, *requests], True)
    for r, err in zip(
        replies,
        [
            "missing Flash gate",
            "requires unbiased",
            "requires unbiased",
            "requires unbiased",
            "invalid Flash gate input shape",
        ],
    ):
        assert err in r.get("error", ""), r
    report.update(
        cases=[],
        dyld_images=images,
        mlx_loaded=False,
        invalid_recovered=len(invalid),
        compared={k: 0 for k in ("logits", "probabilities", "indices", "scores")},
        mismatches={},
    )
    mismatches = []
    for i, (ref, route) in enumerate(zip(refs, routing, strict=True)):
        raw = replies[len(invalid) + 2 * i].get("data", {})
        live = replies[len(invalid) + 2 * i + 1].get("data", {})
        ids = route["data"]["indices"]
        bits = route["data"]["scores"]
        if ref["dtype"] == np.float16:
            scores = mx.array(np.array(bits, np.uint16).view(np.float16).reshape(-1, 10))
            # SsdMoe::route normalizes [1,10] even in a batch of independent AR rows.
            normalized = [
                scores[r : r + 1] / mx.sum(scores[r : r + 1], axis=1, keepdims=True)
                for r in range(scores.shape[0])
            ]
            bits = (
                np.concatenate([np.asarray(row) for row in normalized])
                .view(np.uint16)
                .ravel()
                .tolist()
            )
        expected = {
            "logits": ref["logits"],
            "probabilities": ref["probabilities"],
            "indices": ids,
            "scores": bits,
        }
        assert (
            raw.get("indices") == live.get("indices")
            and raw.get("scores") == live.get("scores")
            and live
        ), (raw, live)
        checked = {}
        for stage, oracle in expected.items():
            arr = np.asarray(raw[stage], dtype=np.uint32)
            want = np.asarray(oracle, dtype=np.uint32)
            assert (
                arr.size
                and arr.shape == want.shape
                and all(type(v) is int and 0 <= v <= 0xFFFFFFFF for v in raw[stage])
            )
            bad = np.flatnonzero(arr != want)
            if bad.size:
                j = int(bad[0])
                mismatches.append(
                    f"{ref['label']} {stage}[{j}]: {int(want[j]):x} vs {int(arr[j]):x}, {bad.size}/{arr.size} differ"
                )
                report["mismatches"][stage] = report["mismatches"].get(stage, 0) + int(bad.size)
            report["compared"][stage] += arr.size
            checked[stage] = hashlib.sha256(want.astype("<u4").tobytes()).hexdigest()
        report["cases"].append(
            {"name": ref["label"], "weight_sha256": ref["weight"], "oracle_sha256": checked}
        )
    if mismatches:
        raise AssertionError(" | ".join(mismatches[:9]))
    report["status"] = "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("cli", "stock", "checkpoint", "report"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    report = {
        "status": "started",
        "seed": 7583,
        "cli_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
        "stock_sha256": hashlib.sha256(args.stock.read_bytes()).hexdigest(),
    }
    with args.report.open("x") as f, tempfile.TemporaryDirectory() as work:
        try:
            verify(args.cli, args.stock, args.checkpoint, Path(work), report)
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            json.dump(report, f, indent=2)
            f.write("\n")
