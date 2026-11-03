"""Compare the native Qwen recurrent component with serial stock codec calls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from physical_moe_router import run


def verify(cli, stock, model, report):
    config = json.loads((model / "config.json").read_text())["text_config"]
    hidden = config["hidden_size"]
    kh = config["linear_num_key_heads"]
    vh = config["linear_num_value_heads"]
    conv = config["linear_conv_kernel_dim"]
    channels = (2 * kh + vh) * 128
    assert hidden > 0 and kh > 0 and vh > 0
    assert config["linear_key_head_dim"] == config["linear_value_head_dim"] == 128
    packets, reference, cases = [], [], []
    rows = np.stack(
        [
            ((np.arange(hidden) * 17 + i * 83) % 257 - 128).astype(np.float32) / np.float32(237)
            for i in range(4)
        ]
    ).astype(np.float16)
    for layer in (0, 36):
        for n in (1, 2, 4):
            data = rows[:n].view(np.uint16).flatten().tolist()
            packets.append(
                {
                    "op": "metal-qwen-gdn",
                    "path": str(model),
                    "prefix": f"model.language_model.layers.{layer}.linear_attn",
                    "x": data,
                }
            )
            reference.append(
                {
                    "op": "mlx-qwen-gdn-sequence",
                    "path": str(model),
                    "cols": hidden,
                    "key_heads": kh,
                    "value_heads": vh,
                    "value_dims": 128,
                    "conv_length": conv,
                    "layer": layer,
                    "x": data,
                }
            )
            cases.append((layer, n))
    refs, _ = run(stock, reference, False)
    invalid = [
        {**packets[0], "x": []},
        {**packets[0], "x": [1]},
        {**packets[0], "prefix": "missing"},
    ]
    replies, images = run(cli, invalid + packets + [packets[-1]], True)
    for reply in replies[: len(invalid)]:
        assert reply.get("error"), reply
    replies = replies[len(invalid) :]
    assert replies[-1] == replies[-2]
    report.update(
        cases=[],
        compared=0,
        mismatches=0,
        invalid_recovered=len(invalid),
        dyld_images=images,
        mlx_loaded=False,
    )
    for (layer, n), ref, reply in zip(cases, refs, replies[:-1], strict=True):
        expected, actual = ref.get("data"), reply.get("data")
        assert expected is not None, ref
        assert actual is not None, reply
        case = {"layer": layer, "positions": n, "tensors": {}}
        for key, count, dtype in [
            ("output", n * hidden, np.uint16),
            ("conv_state", (conv - 1) * channels, np.uint16),
            ("state", vh * 128 * 128, np.float32),
        ]:
            if dtype == np.uint16:
                assert all(type(v) is int and 0 <= v < 65536 for v in actual[key])
                assert all(type(v) is int and 0 <= v < 65536 for v in expected[key])
            else:
                assert all(type(v) in (int, float) for v in actual[key])
                assert all(type(v) in (int, float) for v in expected[key])
            a = np.asarray(actual[key], dtype=dtype)
            b = np.asarray(expected[key], dtype=dtype)
            assert a.size == b.size == count > 0 and a.ndim == b.ndim == 1
            floating = np.float16 if dtype == np.uint16 else np.float32
            assert np.isfinite(a.view(floating)).all() and np.isfinite(b.view(floating)).all()
            assert np.any(b != 0), "empty numerical oracle"
            words = np.uint16 if dtype == np.uint16 else np.uint32
            bad = int(np.count_nonzero(a.view(words) != b.view(words)))
            case["tensors"][key] = {
                "count": count,
                "reference_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                "mismatches": bad,
            }
            report["compared"] += count
            report["mismatches"] += bad
        report["cases"].append(case)
    assert report["compared"] > 0 and report["mismatches"] == 0
    report["status"] = "verified"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cli", type=Path)
    p.add_argument("stock", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("report", type=Path)
    args = p.parse_args()
    report = {
        "status": "started",
        "direct_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
        "stock_sha256": hashlib.sha256(args.stock.read_bytes()).hexdigest(),
    }
    with args.report.open("x") as sink:
        try:
            verify(args.cli, args.stock, args.model.resolve(), report)
        except BaseException as error:
            report.update(status="failed", error=repr(error))
            raise
        finally:
            json.dump(report, sink, indent=2)
            sink.write("\n")


if __name__ == "__main__":
    main()
