"""Compare native Flash hyper-connections and complete MLP residuals with stock checkpoint rows."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from physical_moe_router import run


def verify(cli, stock, model, frozen, root, report):
    rng = np.random.default_rng(7585)
    cases, packets, captured = [], [], {}
    weights = "model.language_model"
    fixture = json.loads((frozen / "reference.json").read_bytes())
    assert fixture["status"] == "complete"

    def add(prefix, mlp, x):
        fp32 = x.dtype == np.float32
        dtype = np.uint32 if fp32 else np.uint16
        cases.append({"prefix": prefix, "mlp": mlp, "fp32": fp32, "words": x.view(dtype).tolist()})
        backing = np.pad(x[::-1], 1, constant_values=9)
        packets.append(
            {
                "op": "metal-checkpoint-mlp" if mlp else "metal-checkpoint-hc",
                "prefix": prefix,
                "path": str(model),
                "return_raw": True,
                "x_layout": {"shape": [1, 10240], "strides": [0, -1], "offset": 10240},
                ("x_f32" if fp32 else "x"): (backing if fp32 else backing.view(dtype)).tolist(),
            }
        )

    for prefix in (weights + ".layers.0", "mtp.layers.0"):
        for dtype in (np.float16, np.float32):
            add(prefix, True, rng.normal(0, 0.37, 10240).astype(dtype))

    def array(desc, dtype):
        assert desc["count"] == 10240
        a = np.fromfile(frozen / desc["file"], dtype="<f4")
        assert a.size == 10240
        return a.astype(dtype)

    for ref in fixture["cases"]:
        name = ref["weight"]
        if "hyper_connection" not in name or ref["rows"] != 1:
            continue
        dtype = np.float16 if ref["dtype"].endswith("16") else np.float32

        captured[len(cases)] = array(ref["output"], dtype).view(
            np.uint16 if dtype == np.float16 else np.uint32
        )
        add(name.removesuffix(".hc_norm.weight"), False, array(ref["input"], dtype))
    assert len(captured) == 4 and len(cases) == 8
    (root / "cases.json").write_text(json.dumps(cases))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLXL3_", "DYLD_"))}
    child = subprocess.run(
        [
            stock,
            "qwen4::hc_row_oracle::flash_hc_mlp_checkpoint_rows",
            "--exact",
            "--ignored",
            "--test-threads=1",
        ],
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
        env={**env, "MLXL3_HC_FIXTURES": str(root), "MLXL3_HC_MODEL": str(model)},
    )
    assert child.returncode == 0 and "1 passed; 0 failed" in child.stdout, (
        child.stdout + child.stderr
    )[-1800:]
    failures = []
    invalid = [
        {**packets[0], "x_layout": {"shape": [1, 10239], "strides": [0, 1], "offset": 0}},
        {**packets[0], "prefix": "absent"},
    ]
    replies, images = run(
        cli, [*invalid, *packets, *[{**p, "return_raw": False} for p in packets]], True
    )
    for err, reply in zip(("streams", "missing"), replies, strict=False):
        assert err in reply.get("error", ""), reply
    report.update(cases=[], dyld_images=images, invalid_recovered=len(invalid), mlx_loaded=False)
    for i, case in enumerate(cases):
        expected = json.loads((root / f"row-{i}.json").read_bytes())
        reply = replies[len(invalid) + i]
        normal = replies[len(invalid) + len(cases) + i]
        assert "data" in reply, reply
        direct = reply["data"]
        assert normal.get("data") == {**direct, "trace": None}, normal.get("error")

        def stages(row):
            return {**row["trace"], **{k: v for k, v in row.items() if k != "trace"}}

        nd, stock_row = stages(direct), stages(expected)
        assert (
            set(nd)
            == set(stock_row)
            == {"normalized", "down", "active", "up", "mixed", "gate", "indices", "output"}
        )
        counts = {
            "normalized": 10240,
            "down": 320,
            "active": 320,
            "up": 10240,
            "mixed": 2560,
            "gate": 0 if case["prefix"].endswith("hyper_connection_mixer") else 4,
            "indices": 10 if case["mlp"] else 0,
            "output": 10240 if case["mlp"] else 0,
        }
        words = np.uint32 if case["fp32"] else np.uint16
        dtype = np.float32 if case["fp32"] else np.float16
        record = {
            "prefix": case["prefix"],
            "dtype": dtype.__name__,
            "mismatches": {},
            "counts": counts,
            "oracle_sha256": {},
        }
        for key, count in counts.items():
            if not count:
                assert nd[key] is None and stock_row[key] is None
                continue
            assert len(nd[key]) == len(stock_row[key]) == count
            actual, ref = (
                np.asarray(a, dtype=np.uint32 if key == "indices" else words)
                for a in (nd[key], stock_row[key])
            )
            if key != "indices":
                assert np.isfinite(actual.view(dtype)).all() and np.isfinite(ref.view(dtype)).all()
            else:
                assert len(set(ref.tolist())) == 10 and all(j < 512 for j in ref)
            if key == "normalized" and i in captured:
                assert np.array_equal(ref, captured[i]), (
                    "stock disagrees with frozen upstream normalization"
                )
            record["oracle_sha256"][key] = hashlib.sha256(ref.tobytes()).hexdigest()
            bad = np.flatnonzero(actual != ref)
            if bad.size:
                at = int(bad[0])
                record["mismatches"][key] = {
                    "count": len(bad),
                    "first": at,
                    "expected": int(ref[at]),
                    "actual": int(actual[at]),
                }
                failures.append(f"{i}/{key} {len(bad)}/{count}")
        report["cases"].append(record)
    if failures:
        raise AssertionError(", ".join(failures))
    report["status"] = "passed"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("cli", "stock", "model", "frozen", "report"):
        p.add_argument(key, type=Path)
    args = p.parse_args()
    report = {"status": "started", "seed": 7585}
    with args.report.open("x") as output, tempfile.TemporaryDirectory() as d:
        try:
            report.update(
                cli_sha256=hashlib.sha256(args.cli.read_bytes()).hexdigest(),
                stock_sha256=hashlib.sha256(args.stock.read_bytes()).hexdigest(),
            )
            verify(args.cli, args.stock, args.model, args.frozen, Path(d), report)
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")
