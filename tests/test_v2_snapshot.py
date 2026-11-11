"""Exercise the benchmark's real CLI with a disposable bridge."""

import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
from test_compare_native import write_bridge

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("behavior", "status", "completed"),
    [("valid", "complete", 4), ("error_on_third", "failed", 2), ("nonfinite_rate", "failed", 0)],
)
def test_snapshot_records_success_and_partial_failures(tmp_path, behavior, status, completed):
    bridge = tmp_path / "bridge"
    write_bridge(bridge, behavior)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("A deterministic prompt")
    output = tmp_path / "result.json"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/v2_snapshot.py"),
            str(bridge),
            str(tmp_path),
            str(prompt),
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    report = json.loads(output.read_text())
    assert (run.returncode == 0) == (status == "complete")
    assert report["status"] == status
    assert len(report["runs"]) == completed
    assert str(tmp_path) not in output.read_text()
    if status == "complete":
        assert report["summary"]["decode_tps"] == {"median": 2, "min": 2, "max": 2}
        assert all(run["stats"]["generated_tokens"] == 128 for run in report["runs"])


def test_published_snapshot_summaries_match_individual_runs():
    results = ROOT / "docs/measurements/v2-models-20261111"
    names = ("lfm3-original", "lfm3-heretic", "lfm3-vl", "qwen-27b", "qwen-35b-a3b", "flash-next")
    for name in names:
        raw = (results / f"{name}.json").read_text()
        assert "/Users/" not in raw and "/var/folders/" not in raw
        report = json.loads(raw)
        assert report["status"] == "complete" and len(report["runs"]) == 4
        assert (
            report["engine_sha256"]
            == "7f5ea49e6768ffddf0ad10e39dfb2cae48410789ce92842d4d1a42e7660338a0"
        )
        assert (
            report["prompt_sha256"]
            == "7447c24f4d66adca67c19034f192b182d5f8a4d680489adcaec41e5de7a16e13"
        )
        assert len({run["token_hash"] for run in report["runs"]}) == 1
        assert all(run["stats"]["generated_tokens"] == 128 for run in report["runs"])
        assert all(run["stats"]["cached_prompt_tokens"] == 0 for run in report["runs"])
        for metric, summary in report["summary"].items():
            values = [run["stats"][metric] for run in report["runs"][1:]]
            assert all(math.isfinite(value) and value > 0 for value in values)
            assert summary == {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
