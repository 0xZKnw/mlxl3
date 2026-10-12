"""Independent shape, safety, parity and CLI oracles for the small-M screen."""

import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
SPEC = importlib.util.spec_from_file_location("smallm", ROOT / "benchmarks/benchmark_smallm.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
BRIDGE_SPEC = importlib.util.spec_from_file_location(
    "smallm_bridge", ROOT / "benchmarks/benchmark_smallm_bridge.py"
)
BRIDGE = importlib.util.module_from_spec(BRIDGE_SPEC)
BRIDGE_SPEC.loader.exec_module(BRIDGE)


def load_screen(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GDN = load_screen("benchmark_smallm_gdn")
INVENTORY = load_screen("benchmark_smallm_inventory")
TENSOR = load_screen("benchmark_smallm_tensor")
SCREENS = [GDN, INVENTORY, TENSOR]


def screen_args(tmp_path):
    return ["--checkpoint", str(tmp_path), "--output", str(tmp_path / "result.json")]


def test_tensor_screen_requires_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("MLXL3_EXPERIMENTAL_SMALLM_TENSOR", raising=False)
    with pytest.raises(SystemExit) as error:
        TENSOR.main(screen_args(tmp_path))
    assert error.value.code == 2 and not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("screen", SCREENS)
@pytest.mark.parametrize("iterations", ["0", "-1", "invalid"])
def test_other_screens_reject_invalid_iterations(screen, iterations, tmp_path, monkeypatch):
    monkeypatch.setenv("MLXL3_EXPERIMENTAL_SMALLM_TENSOR", "1")
    with pytest.raises(SystemExit) as error:
        screen.main(screen_args(tmp_path) + ["--iterations", iterations])
    assert error.value.code == 2 and not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("screen", SCREENS)
def test_other_screens_preserve_existing_evidence(screen, tmp_path, monkeypatch):
    monkeypatch.setenv("MLXL3_EXPERIMENTAL_SMALLM_TENSOR", "1")
    output = tmp_path / "result.json"
    output.write_text("existing negative result")
    with pytest.raises(SystemExit) as error:
        screen.main(screen_args(tmp_path))
    assert error.value.code == 2 and output.read_text() == "existing negative result"


@pytest.mark.parametrize("screen", SCREENS)
def test_other_screens_cannot_validate_zero_cases(screen, tmp_path, monkeypatch):
    monkeypatch.setenv("MLXL3_EXPERIMENTAL_SMALLM_TENSOR", "1")
    monkeypatch.setattr(screen, "run", lambda *args: None)
    with pytest.raises(AssertionError, match="incomplete"):
        screen.main(screen_args(tmp_path))
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "failed" and report["parity"] is False


@pytest.mark.parametrize("screen", SCREENS)
@pytest.mark.parametrize("error", [AssertionError("different bits"), KeyboardInterrupt()])
def test_other_screens_preserve_failed_partial_data(screen, error, tmp_path, monkeypatch):
    monkeypatch.setenv("MLXL3_EXPERIMENTAL_SMALLM_TENSOR", "1")

    def failed(_, __, report):
        report["rows" if screen is INVENTORY else "checks"].append({"visited": True})
        raise error

    monkeypatch.setattr(screen, "run", failed)
    with pytest.raises(type(error)):
        screen.main(screen_args(tmp_path))
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert report["rows" if screen is INVENTORY else "checks"] == [{"visited": True}]
    if screen is not INVENTORY:
        assert report["timings"] == []


@pytest.mark.parametrize(
    "mutation",
    ["empty", "count", "nan", "negative", "hash", "bool_timing", "hash_type", "empty_context"],
)
def test_bridge_rejects_incomplete_output_and_bad_metrics(mutation):
    event = {
        "stats": {
            "generated_tokens": 256,
            "decode_tps": 8.0,
            "decode_seconds": 32.0,
            "mtp_accepted_tokens": 150,
            "mtp_proposed_tokens": 190,
            "mtp_blocks": 95,
        },
        "token_hash": "1234",
        "cache_context": "test",
    }
    text = "nonempty target output"
    if mutation == "empty":
        text = ""
    if mutation == "count":
        event["stats"]["generated_tokens"] = 255
    if mutation == "nan":
        event["stats"]["decode_tps"] = float("nan")
    if mutation == "negative":
        event["stats"]["decode_seconds"] = -1
    if mutation == "hash":
        event["token_hash"] = None
    if mutation == "bool_timing":
        event["stats"]["decode_tps"] = True
    if mutation == "hash_type":
        event["token_hash"] = True
    if mutation == "empty_context":
        event["cache_context"] = ""
    with pytest.raises(AssertionError):
        BRIDGE.fingerprint(event, text, 256)


def test_bridge_deadline_is_forwarded_and_process_error_cannot_pass():
    class Failed:
        def receive(self, deadline):
            assert deadline == 123
            return {"type": "error", "message": "interrupted"}

    with pytest.raises(RuntimeError, match="interrupted"):
        BRIDGE.until(Failed(), "complete", 123, "test")


def test_bridge_exception_preserves_partial_failed_report(tmp_path, monkeypatch):
    engine = tmp_path / "engine"
    engine.write_bytes(b"test fixture")
    output = tmp_path / "result.json"

    def failed(_, report):
        report["runs"].append({"status": "running"})
        raise TimeoutError("silent child")

    monkeypatch.setattr(BRIDGE, "run", failed)
    with pytest.raises(TimeoutError):
        BRIDGE.main(
            [
                "--engine",
                str(engine),
                "--model",
                str(tmp_path),
                "--head",
                str(tmp_path),
                "--output",
                str(output),
            ]
        )
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert report["runs"] == [{"status": "running"}]


@pytest.mark.parametrize(
    "grouped,rows,mb,nt,sg",
    [
        (True, 2, 2, 2, 4),
        (True, 3, 1, 2, 4),
        (True, 4, 2, 2, 4),
        (True, 6, 2, 2, 4),
        (True, 8, 2, 2, 4),
        (False, 3, 3, 1, 8),
        (False, 8, 2, 2, 8),
    ],
)
def test_inventory_matches_independent_native_geometry(grouped, rows, mb, nt, sg):
    actual = BENCH.layout(rows, grouped, 34816, 5120, 2)
    assert (actual["mb"], actual["nt"], actual["sg"], actual["splits"]) == (mb, nt, sg, 1)


def test_split_k_is_not_guessed_from_only_input_size():
    assert BENCH.layout(3, False, 1024, 5120, 3)["splits"] == 8
    assert BENCH.layout(3, False, 5120, 17408, 2)["splits"] == 4


@pytest.mark.parametrize(
    "left,right",
    [
        (np.array([], dtype=np.float16), np.array([], dtype=np.float16)),
        (np.array([np.inf]), np.array([np.inf])),
        (np.array([np.nan]), np.array([np.nan])),
        (np.array([1], dtype=np.float16), np.array([1], dtype=np.float32)),
        (np.array([0.0], dtype=np.float16), np.array([-0.0], dtype=np.float16)),
        (np.array([1], dtype=np.float16), np.array([2], dtype=np.float16)),
    ],
)
def test_empty_nonfinite_incomplete_or_different_bits_fail(left, right):
    with pytest.raises(AssertionError):
        BENCH.assert_exact(left, right)


def test_actual_checkpoint_byte_bounds(tmp_path):
    header = {"a": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]}}
    raw = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + b"\x00\x3c\x00\x40"
    )
    checkpoint = BENCH.Checkpoint(tmp_path)
    np.testing.assert_array_equal(checkpoint.load("a"), np.array([1, 2], dtype=np.float16))
    checkpoint.tensors["a"][2]["data_offsets"] = [4, 8]
    with pytest.raises(ValueError, match="outside shard"):
        checkpoint.load("a")


@pytest.mark.parametrize("error", [AssertionError("divergence"), KeyboardInterrupt()])
def test_failure_preserves_partial_evidence_and_invalidates_parity(tmp_path, monkeypatch, error):
    def run(_, __, ___, report):
        report.update(inventory=[{}], checks=[{"passed_fixture": True}], timings=[])
        raise error

    monkeypatch.setattr(BENCH, "run", run)
    output = tmp_path / "result.json"
    with pytest.raises(type(error)):
        BENCH.main(["--checkpoint", str(tmp_path), "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert report["checks"] == [{"passed_fixture": True}]


def test_cli_rejects_bad_iterations_and_preserves_results(tmp_path):
    output = tmp_path / "result.json"
    for count in ["0", "-1", "invalid"]:
        with pytest.raises(SystemExit) as error:
            BENCH.main(
                ["--checkpoint", str(tmp_path), "--iterations", count, "--output", str(output)]
            )
        assert error.value.code == 2 and not output.exists()
    output.write_text("existing evidence")
    with pytest.raises(SystemExit):
        BENCH.main(["--checkpoint", str(tmp_path), "--output", str(output)])
    assert output.read_text() == "existing evidence"


@pytest.mark.parametrize("accepted,proposed,blocks", [(150, 190, 95), (0, 1, 1), (256, 512, 256)])
def test_bridge_complete_nonempty_result_is_accepted(accepted, proposed, blocks):
    event = {
        "stats": {
            "generated_tokens": 256,
            "decode_tps": 8.0,
            "decode_seconds": 32.0,
            "mtp_accepted_tokens": accepted,
            "mtp_proposed_tokens": proposed,
            "mtp_blocks": blocks,
        },
        "token_hash": "1234",
        "cache_context": "test",
    }
    signature = BRIDGE.fingerprint(event, "target", 256)
    assert signature[:3] == ("1234", "test", 256)
    assert signature[-3:] == (accepted, proposed, blocks)


@pytest.mark.parametrize(
    "behavior",
    [None, "mtp_accepted_tokens", "mtp_proposed_tokens", "mtp_blocks", "invalid", "eof", "silent"],
)
def test_bridge_cli_checks_work_and_reaps_children(tmp_path, monkeypatch, behavior):
    engine = tmp_path / "engine"
    pids = tmp_path / "pids"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        f"with open({str(pids)!r}, 'a') as f: f.write(str(os.getpid()) + '\\n')\n"
        "print(json.dumps({'type':'ready'}), flush=True)\n"
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        " if r['type']=='shutdown': break\n"
        " count=r['max_tokens']\n"
        f" behavior={behavior!r}\n"
        " if behavior=='invalid': print('invalid JSON', flush=True); continue\n"
        " if behavior=='eof': sys.exit(0)\n"
        " if behavior=='silent': time.sleep(60); continue\n"
        " stats={'generated_tokens':count,'decode_tps':8.0,'decode_seconds':count/8,\n"
        "        'mtp_accepted_tokens':count//2,'mtp_proposed_tokens':count,'mtp_blocks':count//2}\n"
        " if behavior in stats and os.environ.get('MLXL3_EXPERIMENTAL_GROUPED_MB3')=='1':\n"
        "  stats[behavior]+=1\n"
        " print(json.dumps({'type':'delta','request_id':r['request_id'],'text':'target'}),flush=True)\n"
        " print(json.dumps({'type':'complete','request_id':r['request_id'],'stats':stats,\n"
        "                   'token_hash':'stable','cache_context':'stable history'}),flush=True)\n"
    )
    engine.chmod(0o700)
    monkeypatch.setattr(BRIDGE.subprocess, "check_output", lambda *args, **kwargs: "fixture")
    original_until = BRIDGE.until
    monkeypatch.setattr(
        BRIDGE,
        "until",
        lambda process, kind, deadline, request=None: original_until(
            process, kind, min(deadline, time.monotonic() + 3), request
        ),
    )
    output = tmp_path / "result.json"
    args = [
        "--engine",
        str(engine),
        "--model",
        str(tmp_path),
        "--head",
        str(tmp_path),
        "--output",
        str(output),
    ]
    try:
        if behavior is None:
            BRIDGE.main(args)
        else:
            with pytest.raises((AssertionError, RuntimeError, TimeoutError)):
                BRIDGE.main(args)
    finally:
        children = [int(pid) for pid in pids.read_text().splitlines()]
        assert children
        for pid in children:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    report = json.loads(output.read_text())
    assert report["status"] == ("complete" if behavior is None else "failed")
    assert report["parity"] is (behavior is None)
    if behavior is None:
        assert len(children) == len(report["runs"]) == 8
    elif behavior.startswith("mtp_"):
        assert len(children) == 2 and report["runs"][0]["parity"] is True
        assert report["runs"][1]["status"] != "complete"


@pytest.mark.parametrize("case", ["finite", "nan", "inf", "overflow", "late_overflow", "partial"])
def test_inventory_cli_validates_each_scaled_step_before_timing(tmp_path, monkeypatch, case):
    tensors = {}
    for index, (dims, cols) in enumerate(
        (d, c) for d in (128, 256, 384, 512) for c in (128, 256, 384, 512)
    ):
        prefix = f"model.language_model.layers.{index}.self_attn.q_proj"
        tensors[prefix + ".trellis"] = np.zeros((dims // 16, cols // 16, 16), np.int16)
        tensors[prefix + ".suh"] = np.ones(dims, np.float16)
        scale = (
            np.nan
            if case == "nan" or case == "partial" and index == 1
            else np.inf
            if case == "inf"
            else 1
        )
        tensors[prefix + ".svh"] = np.full(cols, scale, np.float16)
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
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + b"".join(chunks)
    )
    # CPU adapter for the validation boundary; it does not verify Metal arithmetic.
    mx = SimpleNamespace(
        array=np.array,
        float16=np.float16,
        uint32=np.uint32,
        concatenate=np.concatenate,
        hadamard_transform=lambda value, scale: value,
        eval=lambda *values: None,
        tile=np.tile,
        tanh=np.tanh,
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    def qmv(_, shape, rows, *geometry):
        calls = 0

        def project(*inputs):
            nonlocal calls
            calls += 1
            overflow = case == "overflow" or case == "late_overflow" and calls >= 4
            return np.full((rows, 1, sum(shape["widths"])), 1e10 if overflow else 1, np.float32)

        return project

    monkeypatch.setattr(INVENTORY, "qmv", qmv)
    timed = []

    def paired(_, functions, iterations, steps=1):
        timed.append(steps)
        if case not in ("finite", "partial"):
            pytest.fail("invalid scaled outputs reached timing")
        for function in functions:
            assert np.isfinite(function(steps)).all()
        return {"medians_ms": [None, None]}

    monkeypatch.setattr(INVENTORY, "paired", paired)
    output = tmp_path / "result.json"
    with np.errstate(over="ignore", invalid="ignore"):
        if case == "finite":
            INVENTORY.main(screen_args(tmp_path))
        else:
            with pytest.raises(AssertionError, match="non-finite dependent output"):
                INVENTORY.main(screen_args(tmp_path))
    report = json.loads(output.read_text())
    assert report["status"] == ("complete" if case == "finite" else "failed")
    assert report["parity"] is (case == "finite")
    rows = 80 if case == "finite" else 5 if case == "partial" else 0
    assert len(report["rows"]) == rows
    assert timed == [1, 8] * rows
