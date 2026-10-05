"""Exercise persisted campaign state through the real benchmark CLI."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_bridge(path, behavior="valid", speed=2):
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"behavior, speed = {behavior!r}, {speed!r}\n"
        "if behavior == 'load_error':\n"
        "    print(json.dumps({'type': 'error', 'message': 'fixture loading failed'}), flush=True)\n"
        "    sys.exit(1)\n"
        "print(json.dumps({'type': 'ready'}), flush=True)\n"
        "count = 0\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if request['type'] == 'shutdown': break\n"
        "    count += 1\n"
        "    rid = request['request_id']\n"
        "    if behavior == 'error_on_third' and count == 3:\n"
        "        print(json.dumps({'type': 'error', 'message': 'fixture request failed'}), flush=True)\n"
        "        continue\n"
        "    stats = dict(generated_tokens=request['max_tokens'], cached_prompt_tokens=0,\n"
        "                 decode_tps=speed, decode_seconds=2, prefill_tps=speed,\n"
        "                 prefill_seconds=1, ttft_seconds=1, elapsed_seconds=3)\n"
        "    token_hash = 'different' if behavior == 'different' else 'fixture-hash'\n"
        "    print(json.dumps({'type': 'delta', 'request_id': rid, 'text': 'same text'}), flush=True)\n"
        "    print(json.dumps({'type': 'complete', 'request_id': rid,\n"
        "                      'token_hash': token_hash, 'stats': stats}), flush=True)\n"
    )
    path.chmod(0o755)


def run_campaign(tmp_path, behavior="valid", order="ABBA"):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_bridge(baseline)
    write_bridge(candidate, behavior, speed=4)
    # Disposable condition probes keep this process integration portable.
    for name in ["pmset", "sysctl"]:
        probe = tmp_path / name
        probe.write_text(f"#!{sys.executable}\nprint('synthetic condition probe')\n")
        probe.chmod(0o755)
    output = tmp_path / "results"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/compare_native.py"),
            str(tmp_path / "model"),
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
            "--tokens",
            "4",
            "--repeats",
            "2",
            "--order",
            order,
        ],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return run, output, json.loads((output / "results.json").read_text())


def test_divergent_candidate_is_saved_as_a_failed_pass(tmp_path):
    run, output, report = run_campaign(tmp_path, "different")
    assert run.returncode != 0
    assert report["parity"] is False, "a divergent artifact must not certify parity"
    assert report["status"] == "failed"
    assert len(report["passes"]) == 2
    failed = report["passes"][1]
    assert failed["label"] == "B" and failed["status"] == "failed"
    assert failed["prompts"]["short"]["warmup"]["token_hash"] == "different"
    assert "divergence" in failed["error"]["message"]
    assert json.loads((output / "1-B.json").read_text()) == failed
    assert "summary" not in report


@pytest.mark.parametrize("order", ["ABBA", "BAAB"])
def test_success_requires_every_pass_and_preserves_summary(tmp_path, order):
    run, output, report = run_campaign(tmp_path, order=order)
    assert run.returncode == 0, run.stderr
    assert report["status"] == "complete" and report["parity"] is True
    assert [p["label"] for p in report["passes"]] == list(order)
    for index, current in enumerate(report["passes"]):
        assert current["status"] == "complete"
        assert len(current["prompts"]["short"]["runs"]) == 2
        assert json.loads((output / f"{index}-{current['label']}.json").read_text()) == current
    assert report["summary"]["short"]["decode_tps"] == {"A": 2, "B": 4, "change_percent": 100}


def test_request_failure_preserves_previous_and_partial_runs(tmp_path):
    run, output, report = run_campaign(tmp_path, "error_on_third")
    assert run.returncode != 0
    assert report["status"] == "failed" and report["parity"] is None
    assert len(report["passes"]) == 2
    failed = report["passes"][1]
    assert failed["status"] == "failed"
    assert len(failed["prompts"]["short"]["runs"]) == 1
    assert failed["error"]["message"] == "fixture request failed"
    assert json.loads((output / "1-B.json").read_text()) == failed


def test_loading_failure_has_an_explicit_incomplete_report(tmp_path):
    run, output, report = run_campaign(tmp_path, "load_error", order="BAAB")
    assert run.returncode != 0
    assert report["status"] == "failed" and report["parity"] is None
    assert len(report["passes"]) == 1
    failed = report["passes"][0]
    assert failed["status"] == "failed" and failed["prompts"] == {}
    assert failed["error"]["message"] == "fixture loading failed"
    assert json.loads((output / "0-B.json").read_text()) == failed
