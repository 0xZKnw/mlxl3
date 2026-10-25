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


def write_bridge(
    path, behavior="valid", speed=2, expected_mtp=None, expected_dflash=None, expected_head="head"
):
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))\n"
        "with Path(__file__).with_suffix('.pids').open('a') as pids: pids.write(str(os.getpid()) + '\\n')\n"
        f"behavior, speed = {behavior!r}, {speed!r}\n"
        f"expected_mtp = {expected_mtp!r}\n"
        f"expected_dflash = {expected_dflash!r}\n"
        "if behavior == 'load_error':\n"
        "    print(json.dumps({'type': 'error', 'message': 'fixture loading failed'}), flush=True)\n"
        "    sys.exit(1)\n"
        "if behavior == 'silent_load': time.sleep(60)\n"
        "if behavior == 'delayed_ready': time.sleep(3)\n"
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
        f"        assert request['mtp_head_path'] == (str(Path(__file__).parent / {expected_head!r}) if expected_mtp else '')\n"
        "    count += 1\n"
        "    if expected_dflash is not None:\n"
        "        assert request['dflash2'] == (expected_dflash > 0)\n"
        "        assert request['dflash_mode'] == max(1, expected_dflash)\n"
        "        assert request['dflash_draft_path'] == (str(Path(__file__).parent / 'draft') if expected_dflash else '')\n"
        "        assert not (request['mtp'] and request['dflash2'])\n"
        "    rid = request['request_id']\n"
        "    if behavior == 'silent_request': time.sleep(60)\n"
        "    if behavior == 'slow_request': time.sleep(3)\n"
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
        "    if behavior == 'nonfinite_rate': stats['decode_tps'] = float('nan')\n"
        "    if behavior == 'negative_time': stats['ttft_seconds'] = -1\n"
        "    if behavior == 'zero_time': stats['elapsed_seconds'] = 0\n"
        "    if behavior == 'wrong_id': rid = 'unrelated-request'\n"
        "    text = '' if behavior == 'empty_text' else 'same text'\n"
        "    history = 'different history' if behavior == 'different_history' else 'same text'\n"
        "    if behavior == 'empty_history': history = ''\n"
        "    if behavior == 'missing_history': history = None\n"
        "    if behavior == 'invalid_history': history = True\n"
        "    print(json.dumps({'type': 'delta', 'request_id': rid, 'text': text}), flush=True)\n"
        "    print(json.dumps({'type': 'complete', 'request_id': rid,\n"
        "                      'token_hash': token_hash, 'stats': stats, 'cache_context': history}), flush=True)\n"
        "if behavior == 'exit_after_eof': sys.exit(1)\n"
        "if behavior == 'stall_after_eof': time.sleep(60)\n"
    )
    path.chmod(0o755)


def run_campaign(
    tmp_path,
    behavior="valid",
    order="ABBA",
    timeout=30,
    extra_args=(),
    expected_mtp=None,
    expected_dflash=None,
    candidate_mtp=None,
    candidate_dflash=None,
    slow_probe_after=None,
    candidate_head="head",
):
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    write_bridge(baseline, expected_mtp=expected_mtp, expected_dflash=expected_dflash)
    write_bridge(
        candidate,
        behavior,
        speed=4,
        expected_mtp=expected_mtp if candidate_mtp is None else candidate_mtp,
        expected_dflash=expected_dflash if candidate_dflash is None else candidate_dflash,
        expected_head=candidate_head,
    )
    # Disposable condition probes keep this process integration portable.
    for name in ["pmset", "sysctl"]:
        probe = tmp_path / name
        probe.write_text(
            f"#!{sys.executable}\n"
            "import time\nfrom pathlib import Path\n"
            f"counter = Path({str(tmp_path / 'probe-count')!r})\n"
            "count = int(counter.read_text()) if counter.exists() else 0\n"
            "counter.write_text(str(count + 1))\n"
            f"if {slow_probe_after is not None!r} and count >= {slow_probe_after or 0}: time.sleep(10)\n"
            "print('synthetic condition probe')\n"
        )
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


def test_distinct_heads_reach_every_warmup_and_measured_request(tmp_path):
    run, _, report = run_campaign(
        tmp_path,
        extra_args=[
            "--mtp-depth",
            "1",
            "--mtp-head",
            str(tmp_path / "head"),
            "--candidate-mtp-head",
            str(tmp_path / "head-B"),
        ],
        expected_mtp=1,
        candidate_head="head-B",
    )
    assert run.returncode == 0, run.stderr
    assert report["status"] == "complete" and report["parity"] is True
    assert report["protocol"]["candidate_mtp_head"] == str(tmp_path / "head-B")
    for entry in report["passes"]:
        expected = "head-B" if entry["label"] == "B" else "head"
        assert entry["mtp_head"] == str(tmp_path / expected)


def test_candidate_head_without_mtp_never_launches_an_engine(tmp_path):
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/compare_native.py"),
            str(tmp_path),
            "--baseline",
            str(tmp_path / "missing-A"),
            "--candidate",
            str(tmp_path / "missing-B"),
            "--output",
            str(tmp_path / "results"),
            "--candidate-mtp-head",
            str(tmp_path / "head-B"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert run.returncode == 2 and "requires MTP in both arms" in run.stderr
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("mode", range(4))
def test_dflash_policy_reaches_every_warmup_and_sample(tmp_path, mode):
    arguments = ["--dflash-mode", str(mode)]
    if mode:
        arguments += ["--dflash-draft", str(tmp_path / "draft")]
    run, _, report = run_campaign(tmp_path, extra_args=arguments, expected_dflash=mode)
    assert run.returncode == 0, run.stderr
    assert report["status"] == "complete" and report["parity"] is True
    assert report["protocol"]["dflash_mode"] == mode
    assert len(report["passes"]) == 4


@pytest.mark.parametrize("depth,mode", [(1, 1), (2, 3), (3, 2), (0, 2), (1, 0), (0, 0)])
def test_direct_comparison_sends_only_the_selected_algorithm(tmp_path, depth, mode):
    run, _, report = run_campaign(
        tmp_path,
        extra_args=[
            "--compare-mtp-dflash",
            "--mtp-depth",
            str(depth),
            "--mtp-head",
            str(tmp_path / "head"),
            "--dflash-mode",
            str(mode),
            "--dflash-draft",
            str(tmp_path / "draft"),
        ],
        expected_mtp=depth,
        expected_dflash=0,
        candidate_mtp=0,
        candidate_dflash=mode,
    )
    assert run.returncode == 0, run.stderr
    assert report["status"] == "complete" and report["parity"] is True
    assert report["protocol"]["compare_mtp_dflash"] is True
    for current in report["passes"]:
        assert current["mtp_depth"] == (depth if current["label"] == "A" else 0)
        assert current["dflash_mode"] == (mode if current["label"] == "B" else 0)


@pytest.mark.parametrize(
    "arguments",
    [
        ["--dflash-mode", "1"],
        ["--dflash-mode", "4"],
        ["--dflash-mode", "1", "--dflash-draft", "draft", "--mtp-depth", "1", "--mtp-head", "head"],
    ],
)
def test_invalid_dflash_options_never_start_inference(tmp_path, arguments):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/compare_native.py"),
            "model",
            "--baseline",
            "absent",
            "--candidate",
            "absent",
            "--output",
            str(tmp_path),
            *arguments,
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "behavior",
    [
        "empty_hash",
        "empty_tokens",
        "oversize_tokens",
        "bool_tokens",
        "nonfinite_rate",
        "negative_time",
        "zero_time",
        "empty_text",
        "empty_history",
        "missing_history",
        "invalid_history",
    ],
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


def test_history_divergence_cannot_certify_identical_text_and_tokens(tmp_path):
    run, output, report = run_campaign(tmp_path, "different_history")
    assert run.returncode != 0
    assert report["status"] == "failed" and report["parity"] is False
    failed = report["passes"][1]
    assert failed["label"] == "B" and failed["status"] == "failed"
    reference = report["passes"][0]["prompts"]["short"]["warmup"]
    candidate = failed["prompts"]["short"]["warmup"]
    assert reference["token_hash"] == candidate["token_hash"]
    assert reference["text_sha256"] == candidate["text_sha256"]
    assert reference["history_sha256"] != candidate["history_sha256"]
    assert "summary" not in report
    assert json.loads((output / "1-B.json").read_text()) == failed


@pytest.mark.parametrize("order", ["ABBA", "BAAB"])
def test_success_requires_every_pass_and_preserves_summary(tmp_path, order):
    run, output, report = run_campaign(tmp_path, order=order)
    assert run.returncode == 0, run.stderr
    assert report["status"] == "complete" and report["parity"] is True
    assert [p["label"] for p in report["passes"]] == list(order)
    assert report["protocol"]["campaign_timeout"] is None
    for index, current in enumerate(report["passes"]):
        assert current["status"] == "complete"
        assert current["exit_code"] == 0
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


@pytest.mark.parametrize("order", ["ABBAAB", "BAABBA"])
def test_three_alternating_pairs_preserve_each_warmup_and_sample(tmp_path, order):
    run, output, report = run_campaign(
        tmp_path, order=order, extra_args=["--campaign-timeout", "15"]
    )
    assert run.returncode == 0, run.stderr
    assert report["status"] == "complete" and report["parity"] is True
    assert [item["label"] for item in report["passes"]] == list(order)
    assert report["protocol"]["campaign_timeout"] == 15
    for index, item in enumerate(report["passes"]):
        prompt = item["prompts"]["short"]
        assert prompt["warmup"]["stats"]["generated_tokens"] == 4
        assert len(prompt["runs"]) == 2
        assert item["status"] == "complete"
        assert item["exit_code"] == 0
        assert json.loads((output / f"{index}-{item['label']}.json").read_text()) == item
    assert report["summary"]["short"]["decode_tps"] == {"A": 2, "B": 4, "change_percent": 100}
    assert_campaign_children_reaped(tmp_path)


def assert_campaign_children_reaped(tmp_path):
    for binary in ("baseline", "candidate"):
        pid_file = (tmp_path / binary).with_suffix(".pids")
        for raw in pid_file.read_text().splitlines() if pid_file.exists() else []:
            with pytest.raises(ProcessLookupError):
                os.kill(int(raw), 0)


@pytest.mark.parametrize(
    "behavior,extra_args,stage",
    [
        ("delayed_ready", [], "load"),
        ("slow_request", [], "request"),
        ("flood", [], "flood"),
        ("valid", ["--settle-seconds", "4"], "settle"),
    ],
)
def test_campaign_deadline_bounds_every_wait_and_reaps_children(
    tmp_path, behavior, extra_args, stage
):
    started = time.monotonic()
    run, _, report = run_campaign(
        tmp_path,
        behavior,
        order="BAAB",
        timeout=8,
        extra_args=[
            "--campaign-timeout",
            "2",
            "--load-timeout",
            "5",
            "--request-timeout",
            "5",
            *extra_args,
        ],
    )
    assert time.monotonic() - started < 5
    assert run.returncode != 0
    assert report["status"] == "failed" and report["parity"] is None
    assert len(report["passes"]) == 1
    assert report["passes"][0]["status"] == "failed"
    assert report["error"]["type"] == "TimeoutError"
    assert "summary" not in report
    if stage in ("request", "flood"):
        assert "ready" in report["passes"][0]
        assert "warmup" not in report["passes"][0]["prompts"]["short"]
    if stage == "load":
        assert "ready" not in report["passes"][0]
    if stage == "settle":
        assert not (tmp_path / "candidate.pid").exists()
    assert_campaign_children_reaped(tmp_path)


def test_campaign_deadline_is_not_restarted_between_passes_or_condition_probes(tmp_path):
    started = time.monotonic()
    run, _, report = run_campaign(
        tmp_path,
        order="ABBAAB",
        timeout=8,
        extra_args=["--campaign-timeout", "3", "--load-timeout", "5", "--request-timeout", "5"],
        slow_probe_after=3,
    )
    assert time.monotonic() - started < 5
    assert run.returncode != 0
    assert report["status"] == "failed" and report["parity"] is None
    assert len(report["passes"]) == 2
    assert report["passes"][0]["status"] == "complete"
    assert report["passes"][1]["status"] == "failed"
    assert report["error"]["type"] == "TimeoutError"
    assert not (tmp_path / "candidate.pid").exists()
    assert_campaign_children_reaped(tmp_path)


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_invalid_campaign_timeout_does_not_start_a_bridge(tmp_path, duration):
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/compare_native.py"),
            "model",
            "--baseline",
            "absent",
            "--candidate",
            "absent",
            "--output",
            str(tmp_path / "results"),
            "--campaign-timeout",
            duration,
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert run.returncode == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("behavior", ["exit_after_eof", "stall_after_eof"])
def test_valid_responses_do_not_hide_an_abnormal_bridge_exit(tmp_path, behavior):
    run, _, report = run_campaign(tmp_path, behavior, order="BAAB", timeout=10)
    assert run.returncode != 0
    assert report["status"] == "failed" and report["parity"] is None
    assert "summary" not in report
    assert len(report["passes"]) == 1
    failed = report["passes"][0]
    assert failed["status"] == "failed"
    assert failed["exit_code"] == (1 if behavior == "exit_after_eof" else -signal.SIGTERM)
    assert len(failed["prompts"]["short"]["runs"]) == 2
    assert "native bridge exited abnormally" in failed["error"]["message"]
    assert_campaign_children_reaped(tmp_path)
