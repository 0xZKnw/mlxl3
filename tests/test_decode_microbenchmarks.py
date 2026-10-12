"""CPU regressions for the actual GPU microfilter command entry points."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["benchmark_decode_tiles", "benchmark_gdn_norm_gate"])
def benchmark(request):
    name = request.param
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("iterations", ["0", "-1", "invalid"])
def test_bad_iterations_are_rejected_without_importing_gpu(benchmark, tmp_path, iterations):
    output = tmp_path / "result.json"
    run = subprocess.run(
        [sys.executable, benchmark.__file__, "--iterations", iterations, "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert run.returncode == 2 and not output.exists()
    assert "error:" in run.stderr and "mlx" not in run.stderr


def test_previous_result_is_preserved(benchmark, monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    output.write_text("previous evidence")
    monkeypatch.setattr(benchmark, "run", lambda _: pytest.fail("GPU must not run"))
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--output", str(output)])
    assert error.value.code == 2
    assert output.read_text() == "previous evidence"


@pytest.mark.parametrize("error", [AssertionError("different bits"), KeyboardInterrupt()])
def test_divergence_and_interruption_persist_failed_state(benchmark, monkeypatch, tmp_path, error):
    output = tmp_path / "result.json"

    def failed(_):
        raise error

    monkeypatch.setattr(benchmark, "run", failed)
    with pytest.raises(type(error)):
        benchmark.main(["--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert report["error"]["type"] == type(error).__name__
    assert "timings" not in report


def test_empty_results_cannot_certify_parity(benchmark, monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    monkeypatch.setattr(benchmark, "run", lambda _: {"checks": [], "timings": []})
    with pytest.raises(RuntimeError, match="empty comparison"):
        benchmark.main(["--output", str(output)])
    assert json.loads(output.read_text())["parity"] is False


def test_completed_comparison_is_saved(benchmark, monkeypatch, tmp_path):
    output = tmp_path / "result.json"
    fixture = {"checks": [{"bit_exact": True}], "timings": [{"samples_ms": [1, 2]}]}
    monkeypatch.setattr(benchmark, "run", lambda _: fixture)
    benchmark.main(["--iterations", "3", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["status"] == "complete" and report["parity"] is True
    assert report["checks"] == fixture["checks"] and report["timings"] == fixture["timings"]
