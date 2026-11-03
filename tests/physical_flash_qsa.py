"""Compare dense direct QSA to the independent Flash evaluation capture, checking every row."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from physical_moe_router import run


def packet(model, x, layer, full=False, trace=False):
    half = x.dtype == np.float16
    rows, cols = x.shape
    backing = np.pad(x[:, ::-1], ((0, 0), (1, 1)), constant_values=6)
    return {
        "op": "metal-checkpoint-qsa-layer" if full else "metal-checkpoint-qsa",
        "prefix": f"model.language_model.layers.{layer}" + ("" if full else ".self_attn"),
        "path": str(model),
        "return_raw": trace,
        "x_layout": {"shape": [rows, cols], "offset": cols, "strides": [cols + 2, -1]},
        "x" if half else "x_f32": (backing.view(np.uint16) if half else backing).flatten().tolist(),
    }


def compare(words, expected, report, label):
    expected = np.ascontiguousarray(expected).flatten()
    dtype = expected.dtype
    bits = np.uint16 if dtype == np.float16 else np.uint32
    got = np.asarray(words, dtype=bits)
    assert got.size == expected.size > 0 and np.isfinite(expected).all()
    assert np.isfinite(got.view(dtype)).all()
    ref = expected.view(bits)
    bad = np.flatnonzero(got != ref)
    report["comparisons"] += ref.size
    report["mismatches"] += bad.size
    entry = {
        "count": ref.size,
        "expected_sha256": hashlib.sha256(ref.tobytes()).hexdigest(),
        "mismatches": bad.size,
    }
    if bad.size:
        i = int(bad[0])
        entry.update(first=i, actual=int(got[i]), expected=int(ref[i]))
    report["stages"][label] = entry


def verify(args, report):
    frozen = args.frozen
    d = json.loads((frozen / "reference.json").read_bytes())
    assert d["status"] == "complete" and d["mode"] == "eval"

    def read(desc):
        a = np.fromfile(frozen / desc["file"], dtype="<f4")
        assert a.size == desc["count"] == np.prod(desc["shape"]) and np.isfinite(a).all()
        return a

    for layer in (15, 35):
        steps = d["attention"][str(layer)]
        x = np.stack([read(step["input"]) for step in steps])
        assert x.shape == (5, 2560)
        p = packet(args.model, x, layer, trace=True)
        replies, images = run(
            args.cli, [{**p, "prefix": "missing"}, p, {**p, "return_raw": False}], True
        )
        assert "missing" in replies[0].get("error", ""), replies[0]
        normal = replies[2].get("data")
        traced = replies[1].get("data")
        assert traced and normal, replies[1].get("error") or replies[2].get("error")
        report["dyld_images"] = images
        for i, (step, row, ordinary) in enumerate(zip(steps, traced, normal, strict=True)):
            assert ordinary == {"values": row["values"], "trace": None}
            stages = {**row["trace"], "out": row["values"]}
            assert set(stages) == set(step["stages"]) - {"o_proj"}
            for name, values in stages.items():
                ref = read(step["stages"][name])
                compare(
                    values,
                    ref if name == "out" else ref.astype(np.float16),
                    report,
                    f"{layer}/{i}/{name}",
                )
            # The output projection emits half, then promotes once to match the input dtype.
            compare(
                row["values"],
                read(step["stages"]["o_proj"]).astype(np.float16).astype(np.float32),
                report,
                f"{layer}/{i}/o_proj",
            )


def verify_stock(args, states, report):
    for half in (False, True):
        dtype = np.float16 if half else np.float32
        for full, width, count in ((False, 2560, 512), (True, 10240, 4)):
            x = (
                (
                    (np.arange(width)[None, :] * 17 + np.arange(count)[:, None] * 83) % 257 - 128
                ).astype(np.float32)
                / np.float32(237)
            ).astype(dtype)
            p = packet(args.model, x, 3 if full else 15, full=full)
            replies, _ = run(args.cli, [p], True)
            rows = replies[0].get("data")
            assert rows and len(rows) == count, replies[0].get("error")
            expected = np.fromfile(
                states / f"rows-{str(half).lower()}-{str(full).lower()}.bin", dtype=dtype
            )
            assert expected.size == count * width and np.isfinite(expected).all()
            for i, (r, ref) in enumerate(zip(rows, expected.reshape(count, width), strict=True)):
                compare(r if full else r["values"], ref, report, f"stock/{half}/{full}/{i}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cli", "model", "frozen", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument(
        "--stock", type=Path, help="Stock MLX libtest used to generate dense outputs"
    )
    args = parser.parse_args()
    report = {
        "status": "incomplete",
        "comparisons": 0,
        "mismatches": 0,
        "stages": {},
        "cli_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
    }
    with args.output.open("x") as output:
        try:
            verify(args, report)
            if args.stock:
                import os
                import subprocess
                import tempfile

                with tempfile.TemporaryDirectory() as scratch:
                    name = "qwen4::direct_gdn_oracle::dense_qsa_across_simd_stripes_and_residual"
                    child = subprocess.run(
                        [args.stock, "--exact", name, "--ignored", "--test-threads=1"],
                        env={
                            **os.environ,
                            "MLXL3_QSA_MODEL": str(args.model),
                            "MLXL3_QSA_ORACLE_DIR": scratch,
                        },
                        capture_output=True,
                        text=True,
                        timeout=900,
                        check=True,
                    )
                    assert "1 passed" in child.stdout
                    verify_stock(args, Path(scratch), report)
            report["stock_qualified"] = bool(args.stock)
            assert report["mismatches"] == 0, {
                k: v["mismatches"] for k, v in report["stages"].items() if v["mismatches"]
            }
            report["status"] = "verified"
        except Exception as error:
            report.update(status="failed", error=str(error))
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
