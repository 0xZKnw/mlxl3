"""Exercise the standalone Metal router, a frozen Flash CLI and a scalar stable sort on Apple Silicon."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def run(cli, packets, direct):
    child = subprocess.run(
        [cli, "codec"],
        input="".join(json.dumps(p, allow_nan=False) + "\n" for p in packets),
        env={**os.environ, "DYLD_PRINT_LIBRARIES": "1"},
        text=True,
        capture_output=True,
        timeout=150,
        check=False,
    )
    if child.returncode:
        raise RuntimeError(child.stderr[-2200:])
    images = [s.lower() for s in child.stderr.splitlines() if s.startswith("dyld[")]
    assert any("/metal.framework/" in s for s in images)
    assert any("/libmlx" in s for s in images) == (not direct)
    if direct:
        assert not any(
            key in s
            for s in images
            for key in ("/libjaccl", "/mlx.framework", "/site-packages/mlx/")
        )
    replies = [json.loads(s) for s in child.stdout.splitlines()]
    assert len(replies) == len(packets)
    return replies, len(images)


def verify(cli, stock, updated, report):
    rng = np.random.default_rng(7582)
    packets, stock_packets, updated_packets, refs = [], [], [], []
    for dtype in (np.float16, np.float32):
        word = np.uint16 if dtype == np.float16 else np.uint32
        for experts, k in [
            (1, 1),
            (17, 17),
            (38, 5),
            (64, 1),
            (256, 8),
            (512, 10),
            (1024, 32),
            (1024, 1024),
        ]:
            rows = 3
            base = rng.normal(0.7, 0.4, (experts + 2, rows + 2)).astype(dtype)
            for label in (
                ["mixed", "ties", "nonfinite"] if dtype == np.float16 else ["mixed", "ties"]
            ):
                if label == "ties":
                    base[...] = np.resize(
                        np.array([-0.0, 0.0, 1.0, 1.0, -2.0, 2.0], dtype=dtype), base.shape
                    )
                if label == "nonfinite":
                    base[...] = np.resize(
                        np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1.0, -1.0], dtype=dtype),
                        base.shape,
                    )
                x = base[-2:0:-1, 1:-1].T
                layout = {
                    "shape": list(x.shape),
                    "offset": (x.ctypes.data - base.ctypes.data) // x.itemsize,
                    "strides": [s // x.itemsize for s in x.strides],
                }
                for norm in (False, True):
                    packet = {"op": "metal-router", "x_layout": layout, "k": k, "normalize": norm}
                    if dtype == np.float16:
                        packet["x"] = base.view(word).ravel().tolist()
                        stock_packets.append(
                            {
                                "op": "mlx-router",
                                "data": np.ascontiguousarray(x).view(word).ravel().tolist(),
                                "cols": experts,
                                "k": k,
                                "normalize": norm,
                            }
                        )
                    else:
                        packet["x_f32"] = base.ravel().tolist()
                    updated_packets.append(
                        stock_packets[-1]
                        if dtype == np.float16
                        else {
                            "op": "mlx-router",
                            "x_f32": x.ravel().tolist(),
                            "cols": experts,
                            "k": k,
                            "normalize": norm,
                        }
                    )
                    ids, scores = [], []
                    for row in x:

                        def key(i, values=row):
                            v = float(values[i])
                            return np.isnan(v), 0.0 if np.isnan(v) else v, i

                        indices = sorted(range(experts), key=key)[-k:]
                        selected = row[indices]
                        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                            if norm:
                                total = dtype(0)
                                for value in selected:
                                    total = dtype(value + total)
                                selected = selected / total
                        ids.extend(indices)
                        scores.extend(selected.view(word).tolist())
                    packets.append(packet)
                    refs.append(
                        (
                            f"{dtype.__name__}-{label}-E{experts}-K{k}-N{int(norm)}",
                            ids,
                            scores,
                            dtype,
                        )
                    )
    v = {"shape": [1, 1], "offset": 0, "strides": [0, 0]}
    valid = {"op": "metal-router", "x": [0x3C00], "x_layout": v, "k": 1}
    invalid = []
    for shape, k in [
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1025], 1),
        ([1, 1], 0),
        ([1, 1], 2),
        ([2**31, 2], 1),
    ]:
        invalid.append({**valid, "x_layout": {**v, "shape": shape}, "k": k})
    replies, images = run(cli, [*invalid, *packets], True)
    for row in replies[: len(invalid)]:
        assert "invalid router geometry" in row.get("error", ""), row
    frozen, _ = run(stock, stock_packets, False)
    revised, _ = run(updated, [{"op": "mlx-router", "data": []}, *updated_packets], False)
    assert "invalid router rows" in revised[0].get("error", ""), revised[0]
    report.update(
        cases=[],
        invalid_recovered=len(invalid),
        finite_scores=0,
        nonfinite_scores=0,
        dyld_images=images,
        mlx_loaded=False,
    )
    cursor = 0
    for row, new, (label, ids, words, dtype) in zip(
        replies[len(invalid) :], revised[1:], refs, strict=True
    ):
        candidates = [("direct", row), ("updated", new)]
        if dtype == np.float16:
            candidates.append(("frozen", frozen[cursor]))
            cursor += 1
        unsigned = np.uint16 if dtype == np.float16 else np.uint32
        expected = np.array(words, dtype=unsigned)
        for source, candidate in candidates:
            got = candidate.get("data", {})
            if got.get("indices") != ids:
                raise AssertionError(f"{source} {label}: selected IDs differ: {str(got)[:130]}")
            scores = got["scores"]
            assert len(scores) == len(ids) > 0 and all(
                type(x) is int and 0 <= x <= np.iinfo(unsigned).max for x in scores
            )
            actual = np.array(scores, dtype=unsigned)
            mismatch = (actual != expected) & ~(
                np.isnan(actual.view(dtype)) & np.isnan(expected.view(dtype))
            )
            if np.any(mismatch):
                i = int(np.flatnonzero(mismatch)[0])
                raise AssertionError(
                    f"{source} {label} score[{i}] expected {int(expected[i]):x}, got {int(actual[i]):x}; {int(mismatch.sum())} mismatches"
                )
        finite = int(np.isfinite(expected.view(dtype)).sum())
        report["finite_scores"] += finite
        report["nonfinite_scores"] += len(ids) - finite
        report["cases"].append(
            {
                "case": label,
                "outputs": len(ids),
                "ids_sha256": hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest(),
            }
        )
    assert cursor == len(stock_packets)
    report["status"] = "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ["cli", "stock", "updated", "report"]:
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    report = {
        "status": "started",
        "seed": 7582,
        "binary_sha256": {
            key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
            for key in ["cli", "stock", "updated"]
        },
    }
    with args.report.open("x") as output:
        try:
            verify(args.cli, args.stock, args.updated, report)
        except BaseException as exc:
            report.update(status="failed", error=str(exc))
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")
