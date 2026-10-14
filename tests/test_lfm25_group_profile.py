import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = (
    Path(__file__).parents[1]
    / "docs/measurements/quant-lfm25-research-20261013/profile-baseline.py"
)
spec = importlib.util.spec_from_file_location("lfm25_profile", PATH)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def result():
    pytest.importorskip("ponyexl3.ref.layer")
    from ponyexl3.ref.layer import EXL3Layer

    layer = EXL3Layer(
        "m",
        128,
        128,
        4,
        np.ones((8, 8, 64), np.uint16),
        suh=np.ones(128, np.float32),
        svh=np.ones(128, np.float32),
    )
    return SimpleNamespace(module_key="m", layer=layer, stats={"inner_mse": 0.01})


def test_complete_nonempty_profile_result_is_hashed():
    value = result()
    digest = profile.fingerprints([value], {"m": (128, 128)})
    assert set(digest) == {"m"} and len(digest["m"]) == 64
    value.layer.trellis[0, 0, 0] += 1
    assert profile.fingerprints([value], {"m": (128, 128)}) != digest


@pytest.mark.parametrize(
    "bad", ["empty", "missing", "duplicate", "nan_scale", "nan_error", "shape"]
)
def test_invalid_profile_cannot_claim_parity(bad):
    value = result()
    values, expected = [value], {"m": (128, 128)}
    if bad == "empty":
        values = []
    elif bad == "missing":
        expected["other"] = (128, 128)
    elif bad == "duplicate":
        values.append(value)
    elif bad == "nan_scale":
        value.layer.suh[0] = np.nan
    elif bad == "nan_error":
        value.stats["inner_mse"] = np.nan
    elif bad == "shape":
        expected["m"] = (128, 256)
    with pytest.raises(ValueError):
        profile.fingerprints(values, expected)


def test_zero_repetitions_stops_before_gpu_or_file(tmp_path):
    output = tmp_path / "profile.json"
    with pytest.raises(ValueError, match="Positive"):
        profile.run(output, 0)
    assert not output.exists()


@pytest.mark.parametrize("interrupted", [False, True])
def test_reference_error_or_cancel_leaves_failed_report(tmp_path, monkeypatch, interrupted):
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("ponyexl3.convert")
    import quantize_lfm26
    from ponyexl3.convert import metal_search, mlx_trellis

    monkeypatch.setattr(mx, "device_info", lambda: {"scope": "CPU failure test"})
    monkeypatch.setattr(quantize_lfm26, "patch_converter", lambda: None)
    monkeypatch.setattr(metal_search, "quantize_tiles_mlx", metal_search.quantize_tiles_mlx)
    monkeypatch.setattr(mlx_trellis, "pack_trellis_mlx", mlx_trellis.pack_trellis_mlx)
    source = tmp_path / "bad-reference.py"
    error = "KeyboardInterrupt" if interrupted else "RuntimeError"
    source.write_text(f"raise {error}('injected failure')")
    output = tmp_path / "report.json"
    with pytest.raises(KeyboardInterrupt if interrupted else RuntimeError):
        profile.run(output, 1, source)
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert "injected failure" in report["error"] and not report["runs"]


@pytest.mark.parametrize("failure", ["silent", "partial", "invalid", "error", "timeout", "cancel"])
def test_cli_supervisor_refuses_incomplete_or_interrupted_child(tmp_path, monkeypatch, failure):
    output, pid_file = tmp_path / "profile.json", tmp_path / "pid"
    real_run = subprocess.run
    scratch_dirs = []
    programs = {
        "silent": "pass",
        "partial": f'from pathlib import Path; Path({str(output)!r}).write_text(\'{{"status":"complete","parity":true,"runs":[]}}\')',
        "invalid": f"from pathlib import Path; Path({str(output)!r}).write_text('invalid JSON')",
        "error": "raise RuntimeError('child failed')",
        "timeout": f"import os,time; from pathlib import Path; Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)",
    }

    def child(command, *, timeout, check):
        assert timeout == 240 and check is True
        assert command[-2] == "--scratch-dir"
        scratch = Path(command[-1])
        scratch_dirs.append(scratch)
        (scratch / "discardable-weights").write_bytes(b"test data")
        if failure == "cancel":
            raise KeyboardInterrupt("injected cancellation")
        return real_run(
            [sys.executable, "-c", programs[failure]],
            timeout=1,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    monkeypatch.setattr(profile.subprocess, "run", child)
    # A previous successful report must not survive a failed new run.
    output.write_text('{"status":"complete","parity":true,"runs":[]}')
    with pytest.raises(
        (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyboardInterrupt)
    ):
        profile.main(["--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert report["supervisor_error"] and not output.with_suffix(".tmp").exists()
    assert scratch_dirs and all(not directory.exists() for directory in scratch_dirs)
    if failure == "timeout":
        assert pid_file.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_file.read_text()), 0)
