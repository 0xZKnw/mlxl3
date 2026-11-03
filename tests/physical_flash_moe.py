"""Compare demand-loaded native MoE rows to the production SsdMoe on the same Apple GPU."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from physical_moe_router import run


def verify(cli, stock, model, root, report):
    rng = np.random.default_rng(7584)
    cases, requests = [], []
    for prefix in ("model.language_model.layers.0.mlp", "mtp.layers.0.mlp"):
        for fp32 in (False, True):
            dtype = np.float32 if fp32 else np.float16
            words = np.uint32 if fp32 else np.uint16
            a, b = rng.normal(0, 0.4, (2, 2560)).astype(dtype)
            data = np.stack([a, a, b, a])
            capacity = 20 if fp32 else 10
            cases.append(
                {
                    "prefix": prefix,
                    "fp32": fp32,
                    "rows": data.view(words).tolist(),
                    "slots": capacity,
                }
            )
            padded = np.pad(data[:, ::-1], ((0, 0), (1, 1)), constant_values=9)
            requests.append(
                {
                    "op": "metal-checkpoint-moe",
                    "return_raw": True,
                    "path": str(model),
                    "prefix": prefix,
                    "slots": capacity,
                    "x_layout": {"shape": [4, 2560], "offset": 2560, "strides": [2562, -1]},
                    ("x_f32" if fp32 else "x"): (padded if fp32 else padded.view(words))
                    .ravel()
                    .tolist(),
                }
            )
    (root / "cases.json").write_text(json.dumps(cases))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLXL3_", "DYLD_"))}
    child = subprocess.run(
        [
            stock,
            "ssd_moe::row_oracle::flash_moe_row_stock_oracle",
            "--exact",
            "--ignored",
            "--test-threads=1",
        ],
        env={**env, "MLXL3_MOE_FIXTURES": str(root), "MLXL3_MOE_CHECKPOINT": str(model)},
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    if child.returncode or "1 passed; 0 failed" not in child.stdout:
        raise AssertionError((child.stdout + child.stderr)[-2000:])
    oracle = json.loads((root / "expected.json").read_bytes())
    invalid = [{**requests[0], "slots": 9}, {**requests[0], "prefix": "missing"}]
    replies, images = run(
        cli, [*invalid, *requests, *[{**req, "return_raw": False} for req in requests]], True
    )
    for err, reply in zip(("slot", "missing"), replies[: len(invalid)], strict=True):
        assert err in reply.get("error", ""), reply
    report.update(
        cases=[],
        dyld_images=images,
        mlx_loaded=False,
        invalid_recovered=len(invalid),
        compared=0,
        mismatches=0,
    )
    problems = []
    traced = replies[len(invalid) : len(invalid) + len(requests)]
    ordinary = replies[len(invalid) + len(requests) :]
    for case, rows, reply, normal in zip(cases, oracle, traced, ordinary, strict=True):
        assert len(reply.get("data", [])) == len(rows) == len(case["rows"]) > 0, reply.get("error")
        assert normal.get("data") == [{**row, "switch": None} for row in reply["data"]], normal.get(
            "error"
        )
        seen = set()
        previous = set()
        for index, (ref, got) in enumerate(zip(rows, reply["data"], strict=True)):
            ids = ref["indices"]
            assert (
                len(set(ids)) == len(ids) == 10
                and all(type(v) is int and 0 <= v < 512 for v in ids)
                and got["indices"] == ids
            )
            hits = len(set(ids) & (previous if case["slots"] == 10 else seen))
            assert got["hits"] == hits and got["read_bytes"] == (10 - hits) * 1248012, (
                got["hits"],
                hits,
                got["read_bytes"],
            )
            previous = set(ids)
            seen |= previous
            utype = np.uint32 if case["fp32"] else np.uint16
            dtype = np.float32 if case["fp32"] else np.float16
            assert len(ref["values"]) == len(got["values"]) == 2560
            expected = np.array(ref["values"], dtype=utype)
            actual = np.array(got["values"], dtype=utype)
            assert np.isfinite(expected.view(dtype)).all() and np.isfinite(actual.view(dtype)).all()
            bad = np.flatnonzero(expected != actual)
            row = {
                "layer": case["prefix"],
                "dtype": dtype.__name__,
                "row": index,
                "hits": hits,
                "record_bytes": got["read_bytes"],
                "ids": ids,
                "oracle_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                "max_abs": float(
                    np.abs(expected.view(dtype).astype(np.float64) - actual.view(dtype)).max()
                ),
                "mismatches": len(bad),
                "switch": got.get("switch"),
                "stock_switch": ref.get("switch"),
            }
            if got["switch"] != ref["switch"]:
                problems.append(
                    f"{case['prefix']}/{dtype.__name__}/{index} switch {got['switch']} vs {ref['switch']}"
                )
            if bad.size:
                j = int(bad[0])
                problems.append(
                    f"{case['prefix']}/{dtype.__name__}/{index} column {j}: {int(expected[j]):x} vs {int(actual[j]):x}, {len(bad)}/2560"
                )
            report["compared"] += len(expected)
            report["mismatches"] += len(bad)
            report["cases"].append(row)
    if problems:
        raise AssertionError(" | ".join(problems[:6]))
    report["status"] = "passed"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("cli", "stock", "model", "report"):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    report = {
        "status": "started",
        "seed": 7584,
        "cli_sha256": hashlib.sha256(args.cli.read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(args.stock.read_bytes()).hexdigest(),
    }
    with args.report.open("x") as f, tempfile.TemporaryDirectory() as d:
        try:
            verify(args.cli, args.stock, args.model, Path(d), report)
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            json.dump(report, f, indent=2)
            f.write("\n")
