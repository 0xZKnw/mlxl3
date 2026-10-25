import copy
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


def complete_report(repeats=1, compared=False):
    rows = []
    for pair in range(-1, repeats):
        for group in ("gate_up", "down"):
            for variant in ("reference", "candidate") if compared else ("candidate",):
                rows.append(
                    {
                        "pair": pair,
                        "group": group,
                        "variant": variant,
                        "seconds": 1.0,
                        "stages": dict.fromkeys(profile.STAGES, 0.1),
                        "hashes": {
                            f"model.layers.2.feed_forward.experts.{e}.w{p}": "0" * 64
                            for e in range(4)
                            for p in ((1, 3) if group == "gate_up" else (2,))
                        },
                        "peak_mlx_bytes": 100,
                        "cache_mlx_bytes": 0,
                        "active_mlx_bytes": 1,
                        "cache_purges": 0,
                        "process_rss_bytes": 1,
                    }
                )
    return {"status": "complete", "parity": True, "runs": rows}


@pytest.mark.parametrize("compared", [False, True])
def test_complete_report_requires_each_pass_and_consistent_hashes(compared):
    report = complete_report(compared=compared)
    profile.validate_report(report, 1, compared)
    report["runs"][-1] = copy.deepcopy(report["runs"][0])
    with pytest.raises(ValueError, match="coverage"):
        profile.validate_report(report, 1, compared)


def test_extreme_repeat_count_is_rejected_before_building_passes():
    with pytest.raises(ValueError, match="Incomplete"):
        profile.validate_report(complete_report(), 1 << 128)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_report_json_rejects_nonfinite_values_even_outside_validated_fields(tmp_path, literal):
    output = tmp_path / "child.json"
    output.write_text('{"ignored": ' + literal + "}")
    with pytest.raises(ValueError, match="Non-finite"):
        profile.read_report(output)


@pytest.mark.parametrize("component", [None, "source", "hessian-source"])
def test_cli_supervisor_accepts_complete_valid_child_and_cleans_scratch(
    tmp_path, monkeypatch, component
):
    output = tmp_path / "profile.json"
    report = complete_report(compared=component is not None)
    real_run = subprocess.run
    scratch_dirs = []

    def child(command, *, timeout, check):
        assert timeout == 240 and check is True
        scratch = Path(command[-1])
        scratch_dirs.append(scratch)
        (scratch / "disposable-weights").write_bytes(b"test")
        return real_run(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path; Path({str(output)!r}).write_text({json.dumps(report)!r})",
            ],
            timeout=1,
            check=True,
        )

    monkeypatch.setattr(profile.subprocess, "run", child)
    args = ["--output", str(output), "--repeats", "1"]
    if component:
        args += ["--reference-" + component, str(tmp_path / "unused-reference.py")]
    assert profile.main(args) == report
    assert scratch_dirs and all(not folder.exists() for folder in scratch_dirs)


@pytest.mark.parametrize(
    "bad",
    [
        "hash",
        "missing_key",
        "diverged",
        "nan",
        "infinite",
        "negative_stage",
        "missing_stage",
        "memory",
        "pair",
        "group",
        "variant",
    ],
)
def test_complete_count_does_not_hide_invalid_measurements(bad):
    report = complete_report(compared=True)
    row = report["runs"][1]
    key = next(iter(row["hashes"]))
    if bad == "hash":
        row["hashes"][key] = "bad"
    elif bad == "missing_key":
        del row["hashes"][key]
    elif bad == "diverged":
        row["hashes"][key] = "1" * 64
    elif bad == "nan":
        row["seconds"] = float("nan")
    elif bad == "infinite":
        row["stages"][profile.STAGES[0]] = float("inf")
    elif bad == "negative_stage":
        row["stages"][profile.STAGES[0]] = -1.0
    elif bad == "missing_stage":
        del row["stages"][profile.STAGES[0]]
    elif bad == "memory":
        row["cache_purges"] = -1
    elif bad == "pair":
        row["pair"] = True
    elif bad == "group":
        row["group"] = []
    elif bad == "variant":
        row["variant"] = "unknown"
    with pytest.raises(ValueError):
        profile.validate_report(report, 1, True)


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
@pytest.mark.parametrize("component", ["metal", "hessian"])
def test_reference_error_or_cancel_leaves_failed_report(
    tmp_path, monkeypatch, interrupted, component
):
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
        options = {
            "reference_source" if component == "metal" else "reference_hessian_source": source
        }
        profile.run(output, 1, **options)
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert "injected failure" in report["error"] and not report["runs"]


@pytest.mark.parametrize(
    "failure",
    [
        "silent",
        "partial",
        "full_invalid",
        "full_nan",
        "full_overflow",
        "invalid",
        "error",
        "timeout",
        "cancel",
    ],
)
def test_cli_supervisor_refuses_incomplete_or_interrupted_child(tmp_path, monkeypatch, failure):
    output, pid_file = tmp_path / "profile.json", tmp_path / "pid"
    real_run = subprocess.run
    scratch_dirs = []
    programs = {
        "silent": "pass",
        "partial": f'from pathlib import Path; Path({str(output)!r}).write_text(\'{{"status":"complete","parity":true,"runs":[]}}\')',
        "full_invalid": f"import json; from pathlib import Path; Path({str(output)!r}).write_text(json.dumps(dict(status='complete',parity=True,runs=[{{}}]*8)))",
        "full_nan": f"import json; from pathlib import Path; Path({str(output)!r}).write_text(json.dumps(dict(status='complete',parity=True,runs=[dict(seconds=float('nan'))]*8)))",
        "full_overflow": f"import json; from pathlib import Path; Path({str(output)!r}).write_text(json.dumps(dict(status='complete',parity=True,runs=[dict(seconds=1.0)]*8)).replace('1.0','1e999'))",
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
