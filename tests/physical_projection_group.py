"""Qualify ordinary EXL3 groups against checkpoint-loaded stock Exl3Group."""

import argparse
import functools
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
from physical_moe_router import run
from safetensors.numpy import save_file


def save_checkpoint(folder, tensors, storage, bf16):
    save_file(tensors, folder / "model.safetensors", metadata={"format": "pt"})
    if bf16:
        file = folder / "model.safetensors"
        with file.open("rb") as source:
            size = struct.unpack("<Q", source.read(8))[0]
            header = json.loads(source.read(size))
            payload = source.read()
        for name in bf16:
            header[name]["dtype"] = "BF16"
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        file.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe"}))
    (folder / "tokenizer.json").write_text("{}")
    (folder / "quantization_config.json").write_text(
        json.dumps({"quant_method": "exl3", "tensor_storage": storage})
    )


def fixture(folder, rng, width, widths, k, mode):
    folder.mkdir()
    tensors, storage, bf16 = {}, {}, []
    for index, cols in enumerate(widths):
        prefix = f"p{index}"
        words = rng.integers(0, 65536, (width // 16, cols // 16, k * 16), dtype=np.uint16)
        tensors[prefix + ".trellis"] = words if k % 2 else words.view(np.int16)
        for suffix, n in (("suh", width), ("svh", cols), ("bias", cols)):
            if mode == 1:
                values = np.resize(np.array([0x3F00, 0x3F40, 0xBF00, 0x3F80], np.uint16), n)
                bf16.append(prefix + "." + suffix)
            else:
                values = rng.normal(0.7, 0.1, n).astype(np.float32 if mode == 0 else np.float16)
            tensors[prefix + "." + suffix] = values
        if mode:
            tensors[prefix + (".mcg" if mode == 1 else ".mul1")] = np.array([1], np.int32)
        storage[prefix] = {"quant_format": "exl3"}
    save_checkpoint(folder, tensors, storage, bf16)
    x = rng.normal(0, 0.1, width).astype(np.float16)
    return case(folder, [f"p{i}" for i in range(len(widths))], x, widths)


def case(model, prefixes, x, widths):
    native = {
        "op": "metal-checkpoint-group",
        "path": str(model),
        "prefixes": prefixes,
        "x": x.view(np.uint16).tolist(),
    }
    return {"native": native, "stock": {**native, "op": "mlx-checkpoint-group"}, "widths": widths}


@functools.cache
def header(file):
    with file.open("rb") as source:
        prefix = source.read(8)
        assert len(prefix) == 8
        size = struct.unpack("<Q", prefix)[0]
        assert 0 < size <= 100_000_000
        raw = source.read(size)
        assert len(raw) == size
        return json.loads(raw)


def shape(model, name):
    index = model / "model.safetensors.index.json"
    files = (
        [model / json.loads(index.read_text())["weight_map"][name]]
        if index.exists()
        else sorted(model.glob("*.safetensors"))
    )
    for file in files:
        if name in header(file):
            return header(file)[name]["shape"]
    raise KeyError(name)


def synthetic(root):
    rng = np.random.default_rng(75103)
    cases = []
    for k in range(1, 9):
        for mode in range(3):
            width = (128, 2048, 8192)[(k + mode) % 3]
            cases.append(fixture(root / f"k{k}-cb{mode}", rng, width, [1024, 128, 256], k, mode))
    for widths in ([10240, 6144], [17408, 17408]):
        cases.append(fixture(root / f"dense-{sum(widths)}", rng, 5120, widths, 2, 2))
    # Mixed codebooks retain the ordinary separate path, as does a single projection.
    mixed = fixture(root / "mixed", rng, 2048, [1024, 128], 2, 0)
    file = Path(mixed["native"]["path"]) / "model.safetensors"
    from safetensors.numpy import load_file

    tensors = load_file(file)
    tensors["p1.mul1"] = np.array([1], np.int32)
    save_file(tensors, file, metadata={"format": "pt"})
    cases.append(mixed)
    first = cases[0]
    cases.append(
        {
            "native": {**first["native"], "prefixes": ["p0"]},
            "stock": {**first["stock"], "prefixes": ["p0"]},
            "widths": [1024],
        }
    )
    cases.append(
        {
            "native": {**first["native"], "prefixes": ["p0", "p0"]},
            "stock": {**first["stock"], "prefixes": ["p0", "p0"]},
            "widths": [1024, 1024],
        }
    )
    # A negative-stride initialized backing exercises the shared view preparation.
    logical = np.asarray(first["native"]["x"], np.uint16).view(np.float16)
    backing = np.zeros(logical.size * 2 + 3, np.float16)
    backing[1 : logical.size * 2 + 1 : 2] = logical[::-1]
    cases.append(
        {
            "native": {
                **first["native"],
                "x": backing.view(np.uint16).tolist(),
                "x_layout": {
                    "shape": [1, logical.size],
                    "strides": [backing.size, -2],
                    "offset": logical.size * 2 - 1,
                },
            },
            "stock": first["stock"],
            "widths": first["widths"],
        }
    )
    return cases


def verify(args, root, report):
    cases = synthetic(root)
    for model, groups in (
        (
            args.qwen,
            [
                [f"model.language_model.layers.{i}.self_attn.{p}_proj" for p in ("q", "k", "v")]
                for i in (3, 39)
            ]
            + [
                [
                    f"model.language_model.layers.{i}.mlp.shared_expert.{p}_proj"
                    for p in ("gate", "up")
                ]
                for i in (0, 36)
            ],
        ),
        (
            args.lfm,
            [[f"model.layers.{i}.self_attn.{p}_proj" for p in ("q", "k", "v")] for i in (2, 38)]
            + [[f"model.layers.{i}.feed_forward.{p}" for p in ("w1", "w3")] for i in (0, 1)],
        ),
    ):
        if model is None:
            continue
        for prefixes in groups:
            shapes = [shape(model, p + ".trellis") for p in prefixes]
            width = shapes[0][0] * 16
            assert all(s[0] * 16 == width for s in shapes)
            x = (((np.arange(width) * 37) % 199 - 99) / 233).astype(np.float16)
            cases.append(case(model.resolve(), prefixes, x, [s[1] * 16 for s in shapes]))
    refs, _ = run(args.stock, [c["stock"] for c in cases], False)
    if args.previous:
        reference = refs[27]["data"]["outputs"][0]
        packet = cases[27]["native"]
        previous, _ = run(
            args.previous,
            [{**packet, "op": "metal-exl3-linear", "prefix": "p0"}],
            True,
        )
        assert previous[0].get("data") == reference, "old checkpoint bias control changed"
        report["previous_bias_control"] = {
            "status": "unchanged",
            "compared": len(reference),
            "sha256": hashlib.sha256(args.previous.read_bytes()).hexdigest(),
        }
    packet = cases[0]["native"]
    invalid = [
        {**packet, "prefixes": []},
        {**packet, "prefixes": ["missing"]},
        {**packet, "x": []},
        {**packet, "x": packet["x"] + [0]},
        {**packet, "prefixes": ["p0", "missing"]},
    ]
    native, images = run(
        args.cli, invalid + [c["native"] for c in cases] + [cases[-1]["native"]], True
    )
    assert all(p.get("error") for p in native[: len(invalid)])
    native = native[len(invalid) :]
    assert native[-1] == native[-2]
    report.update(cases=[], compared=0, unequal=0, mlx_loaded=False, dyld_images=images)
    for spec, expected, actual in zip(cases, refs, native[:-1], strict=True):
        assert "data" in expected and "data" in actual, (expected, actual)
        expected, actual = expected["data"], actual["data"]
        assert type(expected["grouped"]) is bool and type(actual["grouped"]) is bool
        assert expected["grouped"] == actual["grouped"]
        assert len(expected["outputs"]) == len(actual["outputs"]) == len(spec["widths"]) > 0
        record = {
            "model": Path(spec["native"]["path"]).name,
            "prefixes": spec["native"]["prefixes"],
            "grouped": actual["grouped"],
            "outputs": [],
        }
        for ref, values, width in zip(
            expected["outputs"], actual["outputs"], spec["widths"], strict=True
        ):
            assert all(type(v) is int and 0 <= v < 65536 for v in ref + values)
            a, b = np.asarray(values, np.uint16), np.asarray(ref, np.uint16)
            assert a.size == b.size == width > 0 and a.ndim == b.ndim == 1
            assert np.isfinite(a.view(np.float16)).all() and np.isfinite(b.view(np.float16)).all()
            assert np.any(b != 0), "empty numerical oracle"
            unequal = int(np.count_nonzero(a != b))
            report["compared"] += width
            report["unequal"] += unequal
            record["outputs"].append(
                {
                    "words": width,
                    "reference_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                    "unequal": unequal,
                }
            )
        report["cases"].append(record)
    assert report["compared"] > 0 and report["unequal"] == 0
    report["invalid_request_recovery"] = len(invalid)
    report["status"] = "verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "stock", "report"):
        parser.add_argument(name, type=Path)
    for name in ("qwen", "lfm", "previous"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--portable", action="store_true")
    args = parser.parse_args()
    if args.portable:
        os.environ["MLXL3_DISABLE_TENSOR_OPS"] = "1"
    report = {
        "status": "started",
        "direct_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
        "stock_sha256": hashlib.sha256(args.stock.read_bytes()).hexdigest(),
        "mode": "forced_four_groups" if args.portable else "native_hardware_policy",
    }
    with args.report.open("x") as sink, tempfile.TemporaryDirectory(prefix="mlxl3-groups-") as temp:
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
