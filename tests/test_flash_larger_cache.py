"""Larger-cache evidence is bounded, exact, fresh and never partially qualified."""

import copy
import hashlib
import json
import math
import os
import queue
import runpy
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/measurements/flash-next-ssd-impl/opt-50-larger-cache/bench.txt"
API = runpy.run_path(str(SOURCE))
LIMIT = API["MAX_FOOTPRINT"]
GOLDEN = {"token_hash": "1234567890abcdef", "cache_context": "known prefix continued"}


def fixture(mode="baseline"):
    rows = []
    for repeat in (0, 1):
        rows.append(
            {
                "repeat": repeat,
                "mode": mode,
                "prompt": "MAIN",
                "token_hash": GOLDEN["token_hash"],
                "text": GOLDEN["cache_context"],
                "stats": {
                    "generated_tokens": 128,
                    "decode_tokens": 127,
                    "decode_seconds": 12.7,
                    "decode_tps": 10.0,
                    "ttft_seconds": 1.0,
                    "prefill_tps": 1.0,
                    "memory": {"process_lifetime_peak_bytes": LIMIT},
                    "mtp_proposed_tokens": 70 if mode == "mtp" else None,
                    "mtp_accepted_tokens": 57 if mode == "mtp" else None,
                    "mtp_blocks": 70 if mode == "mtp" else None,
                },
            }
        )
    item = {
        "engine": "/engine",
        "model": "/model",
        "binary_sha256": "1" * 64,
        "cache_gib": 6,
        "mode": mode,
        "prompt": "MAIN",
        "label": "C",
        "index": 1,
    }
    data = {
        "status": "validated",
        "tokens": 128,
        "repeats": 1,
        "expert_cache_gib": 6,
        "warmup": rows[:1],
        "runs": rows[1:],
        **{k: item[k] for k in ("engine", "model", "binary_sha256")},
    }
    stderr = "\n".join(
        f"Flash Next: loading layer {i}/48, 107 exact expert slots" for i in range(1, 49)
    )
    return data, item, stderr


@pytest.mark.parametrize("mode", ["baseline", "mtp"])
def test_report_at_exact_12gb_boundary(mode, monkeypatch):
    monkeypatch.chdir(ROOT)
    d, item, log = fixture(mode)
    before = copy.deepcopy(d)
    result = API["check_report"](d, item, log, GOLDEN)
    assert result["peak_bytes"] == LIMIT and result["tps"] == 10 and d == before


@pytest.mark.parametrize(
    "fault",
    [
        "status",
        "empty",
        "extra_repeat",
        "cache",
        "engine",
        "binary",
        "hash",
        "text",
        "peak",
        "nan",
        "timing",
        "warm_drift",
        "counts",
        "banks",
        "missing_bank",
    ],
)
def test_report_refuses_bad_evidence(fault, monkeypatch):
    monkeypatch.chdir(ROOT)
    d, item, log = fixture("mtp")
    if fault == "status":
        d["status"] = "incomplete"
    elif fault == "empty":
        d["runs"] = []
    elif fault == "extra_repeat":
        d["repeats"] = 2
    elif fault == "cache":
        d["expert_cache_gib"] = 3
    elif fault in ("engine", "binary"):
        d["engine" if fault == "engine" else "binary_sha256"] = "wrong"
    elif fault == "hash":
        d["warmup"][0]["token_hash"] = "wrong"
    elif fault == "text":
        d["runs"][0]["text"] += "wrong"
    elif fault == "peak":
        d["runs"][0]["stats"]["memory"]["process_lifetime_peak_bytes"] += 1
    elif fault == "nan":
        d["runs"][0]["stats"]["decode_tps"] = math.nan
    elif fault == "timing":
        d["runs"][0]["stats"]["decode_seconds"] = 1.0
    elif fault == "warm_drift":
        d["warmup"][0]["stats"].update(decode_tps=7.0, decode_seconds=127 / 7)
    elif fault == "counts":
        d["runs"][0]["stats"]["mtp_accepted_tokens"] = 56
    elif fault == "banks":
        log = log.replace("107 exact", "89 exact")
    else:
        log = log.split("\n", 1)[1]
    with pytest.raises((AssertionError, ValueError)):
        API["check_report"](d, item, log, GOLDEN)


@pytest.mark.parametrize(
    "text,valid",
    [
        (f" {LIMIT}  peak memory footprint\n", True),
        ("", False),
        ("0 peak memory footprint", False),
        (f"{LIMIT + 1} peak memory footprint", False),
        ("1 peak memory footprint\n2 peak memory footprint", False),
    ],
)
def test_native_footprint_is_required_and_bounded(text, valid):
    if valid:
        assert API["native_footprint"](text) == LIMIT
    else:
        with pytest.raises(AssertionError):
            API["native_footprint"](text)


def records():
    return [
        {
            "mode": "baseline",
            "prompt": prompt,
            "index": i,
            "label": label,
            "status": "validated",
            "swapouts_bytes": 0,
            "tps": 10.3 if label == "C" else 10.0,
            "peak_bytes": LIMIT,
            "cache_gib": 6 if label == "C" else 3,
            "token_hash": GOLDEN["token_hash"],
            "text": GOLDEN["cache_context"],
            "mtp_counts": [None] * 3,
        }
        for prompt in ("MAIN", "HOLD")
        for i, label in enumerate("HCCH")
    ]


@pytest.mark.parametrize(
    "fault",
    [None, "missing", "duplicate", "swap", "peak", "nan", "cache", "divergence", "counts", "mode"],
)
def test_qualification_needs_complete_exact_fixed_matrix(fault):
    r = records()
    if fault == "missing":
        r.pop()
    elif fault == "duplicate":
        r[-1] = r[0]
    elif fault == "swap":
        r[-1]["swapouts_bytes"] = 1
    elif fault == "peak":
        r[-1]["peak_bytes"] += 1
    elif fault == "nan":
        r[-1]["tps"] = math.nan
    elif fault == "cache":
        r[5]["cache_gib"] = 5
    elif fault == "divergence":
        r[-1]["text"] += "wrong"
    elif fault == "counts":
        r[-1]["mtp_counts"] = [1, 0, 1]
    elif fault == "mode":
        r[-1]["mode"] = "mtp"
    if fault:
        with pytest.raises(AssertionError):
            API["qualify"](r, "baseline")
    else:
        assert API["qualify"](r, "baseline")["confirmed_gain"]


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("regression", "rejected_gain"),
        ("control_drift", "inconclusive_drift"),
        ("gain_spread", "inconclusive_drift"),
    ],
)
def test_noise_and_negative_results_stay_visible(fault, expected):
    r = records()
    if fault == "regression":
        for row in r:
            if row["label"] == "C":
                row["tps"] = 9.9
    elif fault == "control_drift":
        r[3]["tps"] = 10.6
    else:
        r[1]["tps"] = 11.0
    result = API["qualify"](r, "baseline")
    assert result["status"] == expected and not result["confirmed_gain"]


@pytest.mark.parametrize("fault", ["ok", "hash", "memory", "silent", "crash", "interrupt"])
def test_actual_driver_reaps_bridge_on_invalid_output_or_interrupt(tmp_path, monkeypatch, fault):
    monkeypatch.chdir(ROOT)
    engine = tmp_path / "engine"
    pidfile = tmp_path / "pid"
    engine.write_text(
        "#!/usr/bin/env python3\n"
        + f"fault={fault!r}\npidfile={str(pidfile)!r}\n"
        + """
import json,os,sys,time
open(pidfile,'w').write(str(os.getpid()))
print(json.dumps({'type':'ready','pid':os.getpid()}),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['type']=='shutdown':break
 if fault=='silent':time.sleep(5)
 if fault=='crash':sys.exit(3)
 print(json.dumps({'type':'generation_mode','request_id':r['request_id'],'mtp_active':False,'dflash_active':False}),flush=True)
 print(json.dumps({'type':'complete','request_id':r['request_id'],'token_hash':'wrong' if fault=='hash' else '1234567890abcdef','cache_context':'known prefix continued','stats':{'generated_tokens':128,'decode_tokens':127,'decode_tps':10.,'decode_seconds':12.7,'ttft_seconds':1.,'prefill_tps':1.,'memory':{'process_lifetime_peak_bytes':12000000001 if fault=='memory' else 12000000000}}}),flush=True)
"""
    )
    engine.chmod(0o755)
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps(GOLDEN))
    if fault == "silent":
        get = queue.Queue.get
        monkeypatch.setattr(queue.Queue, "get", lambda self, **kwargs: get(self, timeout=0.2))
    if fault == "interrupt":

        def interrupt(*args):
            raise KeyboardInterrupt

        monkeypatch.setitem(API["generate"].__globals__, "check_event", interrupt)
    args = (engine, tmp_path, tmp_path / "output", "baseline", "MAIN", 6, golden)
    if fault == "ok":
        first = API["generate"](*args)
        second = API["generate"](
            engine, tmp_path, tmp_path / "output2", "baseline", "MAIN", 6, golden
        )
        assert first["status"] == second["status"] == "validated"
        assert first["runtime"]["pid"] != second["runtime"]["pid"]
        with pytest.raises(ProcessLookupError):
            os.kill(first["runtime"]["pid"], 0)
    else:
        with pytest.raises((AssertionError, RuntimeError, TimeoutError, KeyboardInterrupt)):
            API["generate"](*args)
        report = json.loads((tmp_path / "output/report.json").read_text())
        assert report["status"] == "failed" and "median_decode_tps" not in report
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_campaign_marks_missing_input_failed_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setitem(API["main"].__globals__, "BASE", tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        API["main"]("baseline")
    result = json.loads((tmp_path / "baseline-result.json").read_text())
    assert result["status"] == "failed_or_interrupted" and not result["confirmed_gain"]
    with pytest.raises(AssertionError, match="overwrite"):
        API["main"]("baseline")


@pytest.mark.parametrize("fault", ["interrupt", "timeout", "schedule"])
def test_campaign_keeps_failed_supervision_and_refuses_bad_order(tmp_path, monkeypatch, fault):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(API["main"].__globals__, "BASE", tmp_path)
    matrix = records()
    for i, row in enumerate(matrix):
        row["plan"] = f"arm-{i}.json"
    if fault == "schedule":
        matrix.reverse()
    (tmp_path / "baseline-matrix.json").write_text(json.dumps(matrix))
    (tmp_path / "quality-summary.json").write_text(
        json.dumps({"status": "validated_quality_not_speed", "cache_gib": 6})
    )
    frozen = tmp_path / "frozen"
    frozen.write_text("frozen")
    (tmp_path / "speed-identity.json").write_text(
        json.dumps({str(frozen): hashlib.sha256(frozen.read_bytes()).hexdigest()})
    )
    called = []

    def supervise(path):
        called.append(path)
        path.with_name(path.stem + "-result.json").write_text(
            json.dumps({"commands": [{"status": "failed_or_interrupted", "reaped": True}]})
        )
        if fault == "interrupt":
            raise KeyboardInterrupt
        raise subprocess.TimeoutExpired("fake", 1)

    monkeypatch.setattr(
        runpy,
        "run_path",
        lambda path: {"main": supervise} if "run-supervisor" in str(path) else {"snapshot": dict},
    )
    with pytest.raises((AssertionError, KeyboardInterrupt, subprocess.TimeoutExpired)):
        API["main"]("baseline")
    output = json.loads((tmp_path / "baseline-result.json").read_text())
    assert output["status"] == "failed_or_interrupted" and not output["confirmed_gain"]
    if fault == "schedule":
        assert not called and not output["records"]
    else:
        row = output["records"][0]
        assert (
            len(called) == 1
            and row["status"] == "failed_or_interrupted"
            and row["process"]["reaped"]
        )
