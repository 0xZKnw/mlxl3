"""Replay five independent Flash attention rows in order, checking all captured recurrent stages."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from physical_moe_router import run


def verify(cli, model, frozen, report, states):
    doc = json.loads((frozen / "reference.json").read_bytes())
    assert doc["status"] == "complete" and doc["mode"] == "eval"
    steps = doc["attention"]["2"]
    assert len(steps) == 5

    def array(desc):
        a = np.fromfile(frozen / desc["file"], dtype="<f4")
        assert a.size == desc["count"] == np.prod(desc["shape"]) and np.isfinite(a).all()
        return a

    rows = np.stack([array(step["input"]) for step in steps])
    assert rows.shape == (5, 2560)
    x = np.pad(rows[:, ::-1], ((0, 0), (1, 1)), constant_values=6)
    packet = {
        "op": "metal-checkpoint-gdn",
        "path": str(model),
        "prefix": "model.language_model.layers.2.linear_attn",
        "x_layout": {"shape": [5, 2560], "offset": 2560, "strides": [2562, -1]},
        "x_f32": x.flatten().tolist(),
    }
    replies, images = run(
        cli, [{**packet, "prefix": "missing"}, {**packet, "return_raw": True}, packet], True
    )
    assert "missing" in replies[0].get("error", ""), replies[0]
    traced = replies[1].get("data")
    ordinary = replies[2].get("data")
    assert traced is not None, replies[1].get("error")
    assert ordinary is not None, replies[2].get("error")
    assert len(traced) == len(ordinary) == len(steps)
    report.update(rows=[], dyld_images=images, mlx_loaded=False, comparisons=0, mismatches=0)
    problems = []
    for i, (step, row, normal) in enumerate(zip(steps, traced, ordinary, strict=True)):
        assert normal == {"values": row["values"], "trace": None}
        stages = {**row["trace"], "out": row["values"]}
        widths = {
            "qkv": 10240,
            "a": 48,
            "b": 48,
            "z": 6144,
            "conv": 10240,
            "q": 2048,
            "k": 2048,
            "v": 6144,
            "delta": 6144,
            "normed": 6144,
            "gated": 6144,
            "out": 2560,
            "history": 30720,
            "state": 48 * 128 * 128,
        }
        assert set(stages) == set(widths)
        result = {}
        for k, n in widths.items():
            is_float = k in ("normed", "out", "state")
            bits = np.uint32 if is_float else np.uint16
            dtype = np.float32 if is_float else np.float16
            assert len(stages[k]) == n
            value = np.asarray(stages[k], dtype=bits)
            assert np.isfinite(value.view(dtype)).all()
            if k in ("history", "state"):
                expected = np.fromfile(states / f"{i}-{k}.bin", dtype=bits)
            else:
                expected = array(step["stages"][k]).astype(dtype).view(bits)
            assert expected.size == n and np.isfinite(expected.view(dtype)).all()
            bad = np.flatnonzero(value != expected)
            report["comparisons"] += n
            report["mismatches"] += len(bad)
            result[k] = {
                "count": n,
                "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                "mismatches": len(bad),
            }
            if bad.size:
                j = int(bad[0])
                result[k].update(first=j, expected=int(expected[j]), actual=int(value[j]))
                problems.append(f"{i}/{k} {len(bad)}/{n}")
        report["rows"].append(result)
    if problems:
        raise AssertionError(", ".join(problems))


def verify_layer(cli, model, states, report):
    packets = []
    for half in (False, True):
        dtype = np.float16 if half else np.float32
        rows = np.stack(
            [
                (
                    ((np.arange(10240) * 17 + index * 83) % 257 - 128).astype(np.float32)
                    / np.float32(237)
                ).astype(dtype)
                for index in (0, 1, 0)
            ]
        )
        backing = np.pad(rows[:, ::-1], ((0, 0), (1, 1)), constant_values=8)
        packets.append(
            {
                "op": "metal-checkpoint-gdn-layer",
                "path": str(model),
                "prefix": "model.language_model.layers.0",
                "slots": 10,
                "x_layout": {"shape": [3, 10240], "offset": 10240, "strides": [10242, -1]},
                ("x" if half else "x_f32"): (backing.view(np.uint16) if half else backing)
                .flatten()
                .tolist(),
            }
        )
    replies, _ = run(
        cli,
        [
            {**packets[0], "prefix": "model.language_model.layers.1"},
            {**packets[0], "slots": 9},
            *packets,
        ],
        True,
    )
    for key, result in zip(("PLE", "slot"), replies[:2], strict=True):
        assert key in result.get("error", ""), result
    failures = []
    report["full_layer"] = []
    for half, response in zip((False, True), replies[2:], strict=True):
        assert len(response.get("data", [])) == 3, response.get("error")
        for i, row in enumerate(response["data"]):
            ref = np.fromfile(states / f"layer-{str(half).lower()}-{i}.bin", dtype="<u4")
            out = np.asarray(row, dtype=np.uint32)
            assert len(ref) == len(out) == 10240
            assert (
                np.isfinite(ref.view(np.float32)).all() and np.isfinite(out.view(np.float32)).all()
            )
            bad = np.flatnonzero(ref != out)
            report["full_layer"].append(
                {
                    "half_input": half,
                    "step": i,
                    "count": len(ref),
                    "mismatches": len(bad),
                    "expected_sha256": hashlib.sha256(ref.tobytes()).hexdigest(),
                }
            )
            if bad.size:
                j = int(bad[0])
                failures.append(f"{half}/{i} {len(bad)}/10240 at {j}: {out[j]:08x} vs {ref[j]:08x}")
    if failures:
        raise AssertionError(", ".join(failures))


def stock_states(stock, model, frozen, states):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLXL3_", "DYLD_"))}
    child = subprocess.run(
        [
            stock,
            "qwen4::direct_gdn_oracle::five_gdn_states_and_residual_layers",
            "--exact",
            "--ignored",
            "--test-threads=1",
        ],
        env={
            **env,
            "MLXL3_GDN_MODEL": str(model),
            "MLXL3_GDN_FROZEN": str(frozen),
            "MLXL3_GDN_ORACLE_DIR": str(states),
        },
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert child.returncode == 0 and "1 passed; 0 failed" in child.stdout, (
        child.stdout + child.stderr
    )[-2000:]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("cli", "stock", "model", "frozen", "report"):
        p.add_argument(arg, type=Path)
    args = p.parse_args()
    report = {"status": "started"}
    with args.report.open("x") as output, tempfile.TemporaryDirectory() as root:
        try:
            report["cli_sha256"] = hashlib.sha256(args.cli.read_bytes()).hexdigest()
            report["stock_sha256"] = hashlib.sha256(args.stock.read_bytes()).hexdigest()
            states = Path(root)
            stock_states(args.stock, args.model, args.frozen, states)
            verify(args.cli, args.model, args.frozen, report, states)
            verify_layer(args.cli, args.model, states, report)
            report["status"] = "passed"
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")
