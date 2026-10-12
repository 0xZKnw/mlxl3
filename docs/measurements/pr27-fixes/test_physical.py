"""Untimed PR27 counterexample checks against the local fixes, using real MLX."""

import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BRIDGE = load("benchmark_smallm_bridge")
INVENTORY = load("benchmark_smallm_inventory")


def assert_reaped(path):
    pids = [int(value) for value in path.read_text().splitlines()]
    assert pids
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def engine(tmp_path, behavior):
    path = tmp_path / "engine"
    pids = tmp_path / "pids"
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        f"with open({str(pids)!r}, 'a') as f: f.write(str(os.getpid()) + '\\n')\n"
        "print(json.dumps({'type':'ready'}), flush=True)\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        " if r['type']=='shutdown': break\n"
        " count=r['max_tokens']\n"
        f" behavior={behavior!r}\n"
        " enabled=os.environ.get('MLXL3_EXPERIMENTAL_GROUPED_MB3')=='1'\n"
        " if behavior=='invalid': print('invalid JSON', flush=True); continue\n"
        " if behavior=='eof': sys.exit(0)\n"
        " if behavior=='silent': time.sleep(60); continue\n"
        " text='output' if behavior!='divergence' or not enabled else 'changed'\n"
        " stats={'generated_tokens':count,'decode_tps':8.0,'decode_seconds':count/8,\n"
        "        'mtp_accepted_tokens':count//2, 'mtp_proposed_tokens':count,\n"
        "        'mtp_blocks':count//2}\n"
        " if behavior=='counter_change' and enabled:\n"
        "  stats['mtp_accepted_tokens']-=1; stats['mtp_proposed_tokens']+=2; stats['mtp_blocks']+=1\n"
        " print(json.dumps({'type':'delta','request_id':r['request_id'],'text':text}),flush=True)\n"
        " print(json.dumps({'type':'complete','request_id':r['request_id'],'stats':stats,\n"
        "                   'token_hash':'stable', 'cache_context':'stable history'}),flush=True)\n"
    )
    path.chmod(0o700)
    return path, pids


def bridge_args(tmp_path, behavior, monkeypatch):
    path, pids = engine(tmp_path, behavior)
    monkeypatch.setattr(BRIDGE.subprocess, "check_output", lambda *a, **k: "test condition")
    original_until = BRIDGE.until
    monkeypatch.setattr(
        BRIDGE,
        "until",
        lambda p, k, d, r=None: original_until(p, k, min(d, time.monotonic() + 2), r),
    )
    args = [
        "--engine",
        str(path),
        "--model",
        str(tmp_path),
        "--head",
        str(tmp_path),
        "--output",
        str(tmp_path / "result.json"),
    ]
    return args, pids


def test_bridge_rejects_changed_mtp_work(tmp_path, monkeypatch):
    args, pids = bridge_args(tmp_path, "counter_change", monkeypatch)
    try:
        with pytest.raises(AssertionError, match="MTP counters changed"):
            BRIDGE.main(args)
    finally:
        assert_reaped(pids)
    report = json.loads((tmp_path / "result.json").read_text())
    counts = {
        tuple(
            row["stats"][key]
            for key in ("mtp_accepted_tokens", "mtp_proposed_tokens", "mtp_blocks")
        )
        for row in report["runs"]
        if "stats" in row
    }
    Path(__file__).with_name("counter-change-result.json").write_text(json.dumps(report, indent=2))
    assert report["status"] == "failed" and report["parity"] is False, (
        f"different MTP work {counts} accepted: parity={report['parity']}"
    )


@pytest.mark.parametrize("behavior", ["invalid", "eof", "silent", "divergence"])
def test_bridge_real_transport_failure_is_saved_and_reaped(tmp_path, monkeypatch, behavior):
    args, pids = bridge_args(tmp_path, behavior, monkeypatch)
    with pytest.raises((RuntimeError, TimeoutError, AssertionError)):
        BRIDGE.main(args)
    assert_reaped(pids)
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "failed" and report["parity"] is False


@pytest.mark.parametrize("scale", [np.nan, np.inf, 1], ids=["nan", "inf", "finite"])
def test_inventory_checks_dependent_outputs(tmp_path, monkeypatch, scale):
    # Tiny, disposable checkpoint: sixteen distinct shapes; no model is loaded.
    tensors = {}
    for index, (dims, cols) in enumerate(
        (d, c) for d in (128, 256, 384, 512) for c in (128, 256, 384, 512)
    ):
        prefix = f"model.language_model.layers.{index}.self_attn.q_proj"
        tensors[prefix + ".trellis"] = np.zeros((dims // 16, cols // 16, 16), dtype=np.int16)
        tensors[prefix + ".suh"] = np.ones(dims, dtype=np.float16)
        tensors[prefix + ".svh"] = np.full(cols, scale, dtype=np.float16)
    header, chunks, offset = {}, [], 0
    for name, value in tensors.items():
        raw = value.tobytes()
        header[name] = {
            "dtype": "I16" if value.dtype == np.int16 else "F16",
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    raw_header = json.dumps(header).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw_header)) + raw_header + b"".join(chunks)
    )
    observations = []

    def execute_without_timing(mx, functions, iterations, steps=1):
        for fn in functions:
            value = fn(steps)
            mx.eval(value)
            observations.append(
                {"steps": steps, "finite": bool(np.isfinite(np.asarray(value)).all())}
            )
        return {"medians_ms": [None, None], "review_only_no_timing": True}

    monkeypatch.setattr(INVENTORY, "paired", execute_without_timing)
    args = ["--checkpoint", str(tmp_path), "--output", str(tmp_path / "result.json")]
    finite = bool(np.isfinite(scale))
    if finite:
        INVENTORY.main(args)
    else:
        with pytest.raises(AssertionError, match="non-finite dependent output"):
            INVENTORY.main(args)
    report = json.loads((tmp_path / "result.json").read_text())
    summary = {
        "status": report["status"],
        "parity": report["parity"],
        "rows": len(report["rows"]),
        "dependent_observations": sum(row["steps"] == 8 for row in observations),
        "nonfinite_dependent_outputs": sum(
            row["steps"] == 8 and not row["finite"] for row in observations
        ),
        "timings_measured": False,
        "error": report.get("error"),
    }
    label = "finite" if finite else "nan" if np.isnan(scale) else "inf"
    Path(__file__).with_name(f"inventory-{label}-result.json").write_text(
        json.dumps(summary, indent=2)
    )
    assert report["status"] == ("complete" if finite else "failed"), summary
    assert report["parity"] is finite, summary
    assert summary["nonfinite_dependent_outputs"] == 0, summary
    assert summary["rows"] == (80 if finite else 0), summary
    assert summary["dependent_observations"] == (160 if finite else 0), summary
