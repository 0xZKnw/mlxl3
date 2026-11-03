import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "physical_model_ops", Path(__file__).with_name("physical_model_ops.py")
)
physical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(physical)


def test_fixture_packing_preserves_every_negative_stride_value():
    values = np.arange(35, dtype=np.float16).reshape(5, 7)
    packed = physical.packed(values, strided=True)
    layout = packed["layout"]
    assert layout["strides"][1] < 0 and layout["offset"] > 0
    for row in range(5):
        for col in range(7):
            at = layout["offset"] + row * layout["strides"][0] + col * layout["strides"][1]
            assert packed["values"][at] == values[row, col]


@pytest.mark.parametrize(
    "fault", ["ok", "zero", "mismatch", "mlx", "fail", "timeout", "missing-rejection"]
)
def test_physical_report_cannot_validate_failed_or_missing_checks(tmp_path, monkeypatch, fault):
    binary = tmp_path / "binary"
    binary.write_text(
        "#!/usr/bin/env python3\nimport os,time\n"
        + ("time.sleep(60)\n" if fault == "timeout" else "")
        + "print('test result: ok. "
        + ("0" if fault == "zero" else "1")
        + " passed; 0 failed; 0 ignored;')\n"
        + "print('model-ops checked 20 finite words')\n"
        + "for i in range(20): print(f'model-op {i}: words=1, unequal="
        + ("1" if fault == "mismatch" else "0")
        + "')\n"
        + ("print('dyld: /unexpected/libmlx.dylib')\n" if fault == "mlx" else "")
        + ("raise SystemExit(2)\n" if fault == "fail" else "")
    )
    binary.chmod(0o755)
    original = subprocess.run
    original_popen = subprocess.Popen
    children = []

    def tracked_popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    def oracle_only(command, **kwargs):
        if "--oracle" in command:
            cases = [{"expected": [0]} for _ in range(20)]
            if fault == "missing-rejection":
                cases[0]["expected_error"] = "unqualified shape"
            Path(command[-1]).write_text(json.dumps(cases))
            return subprocess.CompletedProcess(command, 0)
        return original(command, **kwargs)

    monkeypatch.setattr(physical.subprocess, "run", oracle_only)
    monkeypatch.setattr(physical.subprocess, "Popen", tracked_popen)
    output = tmp_path / "report"
    if fault == "ok":
        physical.run(binary, output, timeout=1)
    else:
        with pytest.raises((RuntimeError, subprocess.SubprocessError)):
            physical.run(binary, output, timeout=0.2 if fault == "timeout" else 1)
    report = json.loads((output / "report.json").read_text())
    assert report["parity"] is (fault == "ok")
    assert report["status"] == ("validated" if fault == "ok" else "failed")
    assert len(children) == 1 and children[0].poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(children[0].pid, 0)


def test_invalid_arguments_fail_before_creating_a_report(tmp_path):
    for seconds in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            physical.run(Path(__file__), tmp_path / "unused", timeout=seconds)
    assert not (tmp_path / "unused").exists()
