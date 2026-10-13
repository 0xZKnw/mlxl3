import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks/bench_quantize_packing.py"
spec = importlib.util.spec_from_file_location("packing_benchmark", PATH)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((0, 64), dtype=np.uint16),
        np.zeros((1, 63), dtype=np.uint16),
        np.zeros((1, 64), dtype=np.float32),
        np.ones((1, 64), dtype=np.uint16),
    ],
)
def test_bad_results_cannot_be_reported_as_parity(value):
    with pytest.raises(ValueError):
        benchmark.checked_hash(value, np.zeros((1, 64), dtype=np.uint16))


def test_valid_nonempty_result_is_hashed():
    value = np.zeros((1, 64), dtype=np.uint16)
    assert len(benchmark.checked_hash(value, value.copy())) == 64


@pytest.mark.parametrize(
    "options",
    [
        {"tiles": ()},
        {"bits": ()},
        {"repeats": 0},
        {"iterations": 0},
        {"tiles": (0,)},
        {"bits": (0,)},
        {"bits": (9,)},
    ],
)
def test_invalid_matrix_never_creates_a_parity_report(tmp_path, options):
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="Benchmark needs"):
        benchmark.run(output, **options)
    assert not output.exists()


@pytest.mark.parametrize("option", ["--repeats", "--iterations"])
def test_invalid_benchmark_stops_before_gpu_or_report(tmp_path, option):
    output = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(PATH), "--output", str(output), option, "0"],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2 and "must be positive" in result.stderr
    assert not output.exists()


@pytest.mark.skipif(os.environ.get("MLXL3_SKIP_METAL_CHECK") == "1", reason="physical Apple GPU")
@pytest.mark.parametrize("interrupted", [False, True])
def test_failure_or_interrupt_never_leaves_a_valid_report(tmp_path, monkeypatch, interrupted):
    mx = pytest.importorskip("mlx.core")
    import mlxl3_quantizer

    def fail(*args):
        if interrupted:
            raise KeyboardInterrupt("injected cancellation")
        return mx.zeros((0, 32), dtype=mx.uint16)

    monkeypatch.setattr(mlxl3_quantizer, "pack_trellis_mlx", fail)
    output = tmp_path / "report.json"
    with pytest.raises(KeyboardInterrupt if interrupted else ValueError):
        benchmark.run(output, repeats=1, iterations=1, tiles=(1,), bits=(2,))
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert ("injected cancellation" if interrupted else "Invalid packing result") in report["error"]
    assert all(case["status"] != "complete" for case in report["cases"])
