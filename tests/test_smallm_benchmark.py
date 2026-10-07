"""Independent shape, safety, parity and CLI oracles for the small-M screen."""

import importlib.util
import json
import struct
import sys
from pathlib import Path

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


def test_bridge_complete_nonempty_result_is_accepted():
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
    assert BRIDGE.fingerprint(event, "target", 256)[:3] == ("1234", "test", 256)
