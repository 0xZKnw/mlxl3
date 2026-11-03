"""Compare Flash PLE token-dependent residuals to the independent five-step eval capture."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from physical_moe_router import run


def rows(frozen):
    spec = json.loads((frozen / "reference.json").read_bytes())
    assert spec["mode"] == "eval" and spec["status"] == "complete"
    cases = spec["attention"]["1"]
    assert len(cases) == 5

    def read(desc):
        a = np.fromfile(frozen / desc["file"], dtype="<f4")
        assert a.size == desc["count"] == np.prod(desc["shape"]) and np.isfinite(a).all()
        return a

    return cases, read


def packet(model, x, tokens, prefix="model.language_model.layers.1.ple"):
    dtype = x.dtype
    padded = np.pad(x[:, ::-1], ((0, 0), (1, 1)), constant_values=9)
    return {
        "op": "metal-checkpoint-ple",
        "path": str(model),
        "prefix": prefix,
        "token_ids": tokens,
        "x_layout": {"shape": [len(x), 10240], "offset": 10240, "strides": [10242, -1]},
        ("x" if dtype == np.float16 else "x_f32"): (
            padded.view(np.uint16) if dtype == np.float16 else padded
        )
        .flatten()
        .tolist(),
    }


def verify(cli, model, frozen, report):
    cases, read = rows(frozen)
    x = np.stack([read(c["input"]) for c in cases])
    tokens = [c["token"] for c in cases]
    assert x.shape == (5, 10240)
    req = packet(model, x, tokens)
    replies, images = run(
        cli,
        [
            {**req, "token_ids": []},
            {**req, "token_ids": [248320] * 5},
            {**req, "prefix": "missing"},
            {**req, "return_raw": True},
            req,
        ],
        True,
    )
    for err, r in zip(("tokens", "tokens", "missing"), replies[:3], strict=True):
        assert err in r.get("error", ""), r
    traced = replies[3].get("data")
    normal = replies[4].get("data")
    assert traced is not None, replies[3].get("error")
    assert normal is not None, replies[4].get("error")
    assert len(traced) == len(normal) == len(tokens)
    report.update(
        compared=0, mismatches=0, rows=[], dyld_images=images, mlx_loaded=False, invalid_recovered=3
    )
    problems = []
    history = np.zeros((9, 10240), np.float32)
    for i, (c, r, n) in enumerate(zip(cases, traced, normal, strict=True)):
        assert n == {**r, "trace": None}
        assert 0 <= r["read_bytes"] <= 16 * 82 and r["read_bytes"] % 82 == 0
        if i == 0:
            assert r["read_bytes"] == 16 * 82
        history = np.concatenate([history[1:], read(c["stages"]["normed"])[None, :]])
        got = {**r["trace"], "out": r["values"]}
        assert set(got) == set(c["stages"]) | {"history"}
        record = {"token": tokens[i], "read_bytes": r["read_bytes"], "stages": {}}
        for k, data in got.items():
            dtype = np.float16 if k in ("embedding", "key") else np.float32
            bits = np.uint16 if dtype == np.float16 else np.uint32
            expected = (
                (history.flatten() if k == "history" else read(c["stages"][k]))
                .astype(dtype)
                .view(bits)
            )
            actual = np.asarray(data, dtype=bits)
            assert (
                len(actual) == len(expected) > 0
                and np.isfinite(actual.view(dtype)).all()
                and np.isfinite(expected.view(dtype)).all()
            )
            bad = np.flatnonzero(actual != expected)
            report["compared"] += len(actual)
            report["mismatches"] += len(bad)
            row = {
                "count": len(expected),
                "mismatches": len(bad),
                "oracle_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
            }
            if bad.size:
                j = int(bad[0])
                row.update(first=j, expected=int(expected[j]), actual=int(actual[j]))
                problems.append(f"{i}/{k} {len(bad)}/{len(actual)}")
            record["stages"][k] = row
        report["rows"].append(record)
    if problems:
        raise AssertionError(", ".join(problems))


def extended(cli, stock, model, frozen, root, report):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MLXL3_", "DYLD_"))}
    child = subprocess.run(
        [
            stock,
            "qwen4::direct_gdn_oracle::ple_history_and_complete_layer_on_twelve_steps",
            "--exact",
            "--ignored",
            "--test-threads=1",
        ],
        env={
            **env,
            "MLXL3_PLE_MODEL": str(model),
            "MLXL3_PLE_FROZEN": str(frozen),
            "MLXL3_PLE_ORACLE_DIR": str(root),
        },
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert child.returncode == 0 and "1 passed; 0 failed" in child.stdout, (
        child.stdout + child.stderr
    )[-2000:]
    cases, read = rows(frozen)
    order = (0, 1, 2, 3, 4, 0, 0, 4, 1, 2, 3, 4)
    tokens = [cases[i]["token"] for i in order]
    x = np.stack([read(cases[i]["input"]) for i in order])
    report["extended"] = []
    failures = []
    for half in (False, True):
        req = packet(model, x.astype(np.float16 if half else np.float32), tokens)
        both, _ = run(
            cli,
            [
                {**req, "return_raw": True},
                {
                    **req,
                    "op": "metal-checkpoint-ple-layer",
                    "prefix": "model.language_model.layers.1",
                },
            ],
            True,
        )
        ple, layer = (r.get("data") for r in both)
        assert ple is not None and layer is not None, [r.get("error") for r in both]
        assert len(ple) == len(layer) == len(order)
        reads = [r["read_bytes"] for r in ple]
        assert (
            reads[0] == 16 * 82
            and 0 in reads
            and all(0 <= v <= 16 * 82 and v % 82 == 0 for v in reads)
        )
        for i, (branch, residual) in enumerate(zip(ple, layer, strict=True)):
            record = {
                "half_input": half,
                "step": i,
                "token": tokens[i],
                "read_bytes": branch["read_bytes"],
                "stages": {},
            }
            for name, val, n in (
                ("history", branch["trace"]["history"], 92160),
                ("ple", branch["values"], 10240),
                ("layer", residual, 10240),
            ):
                expected = np.fromfile(root / f"{name}-{str(half).lower()}-{i}.bin", dtype="<u4")
                actual = np.asarray(val, dtype=np.uint32)
                assert (
                    len(actual) == len(expected) == n
                    and np.isfinite(expected.view(np.float32)).all()
                    and np.isfinite(actual.view(np.float32)).all()
                )
                bad = np.flatnonzero(expected != actual)
                record["stages"][name] = {
                    "count": n,
                    "mismatches": len(bad),
                    "oracle_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
                }
                if bad.size:
                    j = int(bad[0])
                    record["stages"][name].update(
                        first=j, expected=int(expected[j]), actual=int(actual[j])
                    )
                    failures.append(f"{half}/{i}/{name} {len(bad)}/{n}")
            report["extended"].append(record)
    if failures:
        raise AssertionError(", ".join(failures))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("cli", "stock", "model", "frozen", "report"):
        p.add_argument(k, type=Path)
    a = p.parse_args()
    report = {"status": "started"}
    with a.report.open("x") as output, tempfile.TemporaryDirectory() as root:
        try:
            report["cli_sha256"] = hashlib.sha256(a.cli.read_bytes()).hexdigest()
            report["stock_sha256"] = hashlib.sha256(a.stock.read_bytes()).hexdigest()
            verify(a.cli, a.model, a.frozen, report)
            extended(a.cli, a.stock, a.model, a.frozen, Path(root), report)
            report["status"] = "passed"
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")
