"""Exercise persisted campaign state through the real benchmark CLI."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def write_bridge(path, behavior="valid", speed=2, expected_mtp=None):
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
        f"behavior, speed = {behavior!r}, {speed!r}\n"
        f"expected_mtp = {expected_mtp!r}\n"
        "if behavior == 'load_error':\n"
        "    print(json.dumps({'type': 'error', 'message': 'fixture loading failed'}), flush=True)\n"
        "    sys.exit(1)\n"
        "if behavior == 'silent_load': time.sleep(60)\n"
        "if behavior == 'partial_ready':\n"
        "    os.write(1, b'{\"type\":'); time.sleep(60)\n"
        "if behavior == 'closed_stdin': os.close(0)\n"
        "print(json.dumps({'type': 'ready'}), flush=True)\n"
        "if behavior == 'closed_stdin': time.sleep(60)\n"
        "count = 0\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    if request['type'] == 'shutdown': break\n"
        "    if expected_mtp is not None:\n"
        "        assert request['mtp'] == (expected_mtp > 0)\n"
        "        assert request['mtp_depth'] == max(1, expected_mtp)\n"
        "        assert request['mtp_head_path'] == (str(Path(__file__).parent / 'head') if expected_mtp else '')\n"
        "    count += 1\n"
        "    rid = request['request_id']\n"
        "    if behavior == 'silent_request': time.sleep(60)\n"
        "    if behavior == 'eof': time.sleep(0.3); sys.exit(0)\n"
        "    if behavior == 'invalid': print('invalid JSON', flush=True); continue\n"
        "    if behavior == 'partial_complete':\n"
        '        os.write(1, b\'{"type":"complete"\'); time.sleep(60)\n'
        "    if behavior == 'flood':\n"
        "        while True:\n"
        "            print(json.dumps({'type': 'delta', 'request_id': rid, 'text': ''}), flush=True)\n"
        "            time.sleep(0.005)\n"
        "    if behavior == 'error_on_third' and count == 3:\n"
        "        print(json.dumps({'type': 'error', 'message': 'fixture request failed'}), flush=True)\n"
        "        continue\n"
        "    if behavior == 'cancelled':\n"
        "        print(json.dumps({'type': 'cancelled', 'request_id': rid}), flush=True)\n"
        "        continue\n"
        "    stats = dict(generated_tokens=request['max_tokens'], cached_prompt_tokens=0,\n"
        "                 decode_tps=speed, decode_seconds=2, prefill_tps=speed,\n"
        "                 prefill_seconds=1, ttft_seconds=1, elapsed_seconds=3)\n"
        "    token_hash = 'different' if behavior == 'different' else 'fixture-hash'\n"
        "    if behavior == 'empty_hash': token_hash = ''\n"
        "    if behavior == 'empty_tokens': stats['generated_tokens'] = 0\n"
        "    if behavior == 'oversize_tokens': stats['generated_tokens'] += 1\n"
        "    if behavior == 'bool_tokens': stats['generated_tokens'] = True\n"
        "    if behavior == 'wrong_id': rid = 'unrelated-request'\n"
        "    print(json.dumps({'type': 'delta', 'request_id': rid, 'text': 'same text'}), flush=True)\n"
        "    print(json.dumps({'type': 'complete', 'request_id': rid,\n"
        "                      'token_hash': token_hash, 'stats': stats}), flush=True)\n"
    )
    path.chmod(0o755)


def run_campaign(
    tmp_path, behavior="valid", order="ABBA", timeout=30, extra_args=(), expected_mtp=None
):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_bridge(baseline, expected_mtp=expected_mtp)
    write_bridge(candidate, behavior, speed=4, expected_mtp=expected_mtp)
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
            *extra_args,
        ],
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ.get("PATH", "")},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )
    return run, output, json.loads((output / "results.json").read_text())


@pytest.mark.parametrize("depth", range(4))
def test_explicit_mtp_modes_and_short_warmup_reach_both_engines(tmp_path, depth):
    arguments = ["--mtp-depth", str(depth), "--warmup-tokens", "2"]
    if depth:
        arguments += ["--mtp-head", str(tmp_path / "head")]
    run, _, report = run_campaign(tmp_path, extra_args=arguments, expected_mtp=depth)
    assert run.returncode == 0, run.stderr
    assert report["status"] == "complete" and report["parity"] is True
    assert report["protocol"]["mtp_depth"] == depth
    assert report["protocol"]["warmup_tokens"] == 2
    for current in report["passes"]:
        prompt = current["prompts"]["short"]
        assert prompt["warmup"]["stats"]["generated_tokens"] == 2
        assert [r["stats"]["generated_tokens"] for r in prompt["runs"]] == [4, 4]


@pytest.mark.parametrize(
    "behavior", ["empty_hash", "empty_tokens", "oversize_tokens", "bool_tokens"]
)
def test_empty_or_invalid_outputs_never_certify_a_campaign(tmp_path, behavior):
    run, _, report = run_campaign(tmp_path, behavior, order="BAAB")
    assert run.returncode != 0
    assert report["status"] == "failed" and report["parity"] is None
    assert "summary" not in report


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


def test_cancelled_generation_terminates_and_records_failure(tmp_path):
    run, _, report = run_campaign(tmp_path, "cancelled", timeout=3)
    assert run.returncode != 0
    assert report["status"] == "failed" and report["parity"] is None
    assert "cancelled" in report["error"]["message"]


@pytest.mark.parametrize(
    "behavior,message",
    [
        ("silent_load", "timed out"),
        ("partial_ready", "timed out"),
        ("silent_request", "timed out"),
        ("partial_complete", "timed out"),
        ("flood", "timed out"),
        ("wrong_id", "timed out"),
        ("eof", "complete JSON"),
        ("invalid", "invalid JSON"),
        ("closed_stdin", "closed stdin"),
    ],
)
def test_cli_protocol_failures_are_bounded_and_reaped(tmp_path, behavior, message):
    # Error diagnosis must allow child scheduling/teardown beyond the silence budget.
    load_timeout = "0.5" if behavior in ("silent_load", "partial_ready") else "3"
    request_timeout = "0.15" if message == "timed out" else "3"
    run, _, report = run_campaign(
        tmp_path,
        behavior,
        order="BAAB",
        timeout=10,
        extra_args=["--load-timeout", load_timeout, "--request-timeout", request_timeout],
    )
    assert run.returncode != 0
    assert report["status"] == "failed" and report["parity"] is None
    assert message in report["error"]["message"]
    pid = int((tmp_path / "candidate.pid").read_text())
    deadline = time.monotonic() + 2
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            os.kill(pid, signal.SIGKILL)
            pytest.fail("native benchmark left its child alive")
        time.sleep(0.01)
