"""Negative oracles for the idle-memory driver, independent of MLX arithmetic."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "idle_memory", Path(__file__).parents[1] / "benchmarks/benchmark_idle_memory.py"
)
idle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(idle)


def completion():
    return {
        "type": "complete",
        "token_hash": "123456",
        "cache_context": "Useful text",
        "stats": {
            "generated_tokens": 32,
            "prompt_tokens": 600,
            "cached_prompt_tokens": 256,
            "evaluated_prompt_tokens": 344,
            "context_used": 632,
            "context_limit": 32768,
            "decode_tps": 20.0,
            "prefill_tps": 100.0,
            "ttft_seconds": 1.0,
        },
    }


@pytest.mark.parametrize(
    "key,value",
    [
        ("generated_tokens", 0),
        ("generated_tokens", True),
        ("generated_tokens", 33),
        ("prompt_tokens", 0),
        ("context_used", 631),
        ("cached_prompt_tokens", 0),
        ("context_limit", 10),
        ("decode_tps", float("nan")),
        ("prefill_tps", float("inf")),
        ("ttft_seconds", 0),
    ],
)
def test_quality_rejects_empty_invalid_or_inconsistent_outputs(key, value):
    event = completion()
    event["stats"][key] = value
    with pytest.raises(ValueError):
        idle.quality(event)


@pytest.mark.parametrize("key", ["token_hash", "cache_context"])
def test_quality_requires_both_oracles(key):
    event = completion()
    event[key] = ""
    with pytest.raises(ValueError, match="oracle"):
        idle.quality(event)


def test_quality_keeps_context_and_token_identity():
    result = idle.quality(completion())
    assert result["stats"]["generated_tokens"] == 32
    assert result["stats"]["cached_prompt_tokens"] == 256
    assert result["text_sha256"] != "Useful text"


def runs():
    names = [
        "short-normal",
        "short-mtp",
        "cache-cold",
        "cache-warm",
        "invalid-depth",
        "cache-after-error",
        "long-mtp",
        "cancel",
        "recovery",
    ]
    rows = [
        {
            "case": name,
            "quality": None if name in ("cancel", "invalid-depth") else idle.quality(completion()),
            "idle_bytes": 1024,
        }
        for name in names
    ]
    run = {"records": rows, "status": "complete", "reaped": True}
    return [run, deepcopy(run)]


def test_complete_comparison_and_mutations():
    idle.validate_runs(runs())
    actual = runs()
    actual[1]["records"][3]["quality"]["stats"]["cached_prompt_tokens"] = 0
    with pytest.raises(ValueError, match="changed"):
        idle.validate_runs(actual)


@pytest.mark.parametrize(
    "mutation", ["empty", "partial", "not_reaped", "zero_ram", "bool_ram", "ids"]
)
def test_validation_rejects_failed_or_vacuous_comparisons(mutation):
    actual = runs()
    if mutation == "empty":
        actual = []
    elif mutation == "partial":
        actual[1]["records"].pop()
    elif mutation == "not_reaped":
        actual[1]["reaped"] = False
    elif mutation in ("zero_ram", "bool_ram"):
        actual[1]["records"][0]["idle_bytes"] = 0 if mutation == "zero_ram" else True
    else:
        actual[1]["records"][0]["quality"]["token_hash"] = "different"
    with pytest.raises(ValueError):
        idle.validate_runs(actual)


def test_exchange_rejects_mismatched_request():
    class Child:
        def send(self, _request, _deadline):
            pass

        def receive(self, _deadline):
            return {"type": "complete", "request_id": "other"}

    with pytest.raises(ValueError, match="identity"):
        idle.exchange(Child(), {"request_id": "ours"})


def test_failed_engine_is_reaped(tmp_path):
    engine = tmp_path / "engine"
    engine.write_text(
        '#!/bin/sh\nprintf \'fixture stderr\\n\' >&2\nprintf \'%s\\n\' \'{"type":"error","message":"fixture failure"}\'\n'
    )
    engine.chmod(0o755)
    result = {}
    with pytest.raises(ValueError, match="fixture failure"):
        idle.run(engine, "unused", "unused", measure=lambda _pid: 1024, result=result)
    assert result["stderr"] == "fixture stderr\n"


def test_main_preserves_partial_failure_and_restores_handlers(tmp_path, monkeypatch):
    output = tmp_path / "failed.json"
    monkeypatch.setattr(
        idle.sys,
        "argv",
        [
            "memory",
            "--baseline",
            "stock",
            "--candidate",
            "new",
            "--model",
            "model",
            "--head",
            "head",
            "--output",
            str(output),
        ],
    )
    before = idle.signal.getsignal(idle.signal.SIGTERM)

    def failed_run(_engine, _model, _head, *, result, progress, timeout, **_kwargs):
        assert timeout == 180
        result["records"] = [{"case": "partial", "idle_bytes": 1024}]
        progress()
        raise TimeoutError("fixture timeout")

    monkeypatch.setattr(idle, "run", failed_run)
    with pytest.raises(TimeoutError):
        idle.main()
    actual = json.loads(output.read_text())
    assert actual["status"] == "failed" and actual["parity"] is False
    assert actual["runs"][0]["records"] == [{"case": "partial", "idle_bytes": 1024}]
    assert idle.signal.getsignal(idle.signal.SIGTERM) == before


def saver_runs():
    actual = runs()
    for run in actual:
        run["records"] = [row for row in run["records"] if row["case"] != "long-mtp"]
    actual[1]["memory_saver"] = True
    for row in actual[1]["records"]:
        row["idle_memory"] = {"mlx_cache_bytes": 0}
        if row["quality"] is not None:
            row["quality"]["stats"].update(cached_prompt_tokens=0, evaluated_prompt_tokens=600)
    return actual


def test_saver_compares_outputs_without_claiming_identical_cache_reuse():
    actual = saver_runs()
    before = deepcopy(actual)
    idle.validate_runs(actual, short_only=True)
    idle.validate_runs(list(reversed(actual)), short_only=True)
    assert actual == before


@pytest.mark.parametrize("mutation", ["cache", "allocator", "bool", "ids", "mtp", "duplicate"])
def test_saver_validation_still_rejects_retention_and_divergence(mutation):
    actual = saver_runs()
    row = actual[1]["records"][0]
    if mutation == "cache":
        row["quality"]["stats"]["cached_prompt_tokens"] = 256
    elif mutation == "allocator":
        row["idle_memory"]["mlx_cache_bytes"] = 1
    elif mutation == "bool":
        row["idle_memory"]["mlx_cache_bytes"] = False
    elif mutation == "ids":
        row["quality"]["token_hash"] = "different"
    elif mutation == "mtp":
        row["quality"]["stats"]["mtp_blocks"] = 1
    else:
        actual[0]["records"][1] = deepcopy(actual[0]["records"][0])
    with pytest.raises(ValueError):
        idle.validate_runs(actual, short_only=True)


def test_saver_runs_real_protocol_and_cleans_child(tmp_path):
    import sys

    wire = tmp_path / "wire.jsonl"
    engine = tmp_path / "engine"
    engine.write_text(
        f"#!{sys.executable}\n"
        "import json, signal, sys, time\n"
        f"wire = {str(wire)!r}\n"
        "cancelled = False\n"
        "def cancel(*_):\n    global cancelled\n    cancelled = True\n"
        "signal.signal(signal.SIGUSR1, cancel)\n"
        "def emit(value):\n    print(json.dumps(value), flush=True)\n"
        "emit({'type':'ready', 'memory_saver_supported':True})\n"
        "for line in sys.stdin:\n"
        "    request = json.loads(line)\n"
        "    with open(wire, 'a') as log:\n        log.write(line)\n"
        "    kind, rid = request['type'], request['request_id']\n"
        "    if kind == 'shutdown':\n        break\n"
        "    if kind == 'ping':\n"
        "        emit({'type':'pong','request_id':rid,'memory':{'mlx_cache_bytes':0}})\n"
        "    elif request['mtp_depth'] == 4:\n"
        "        emit({'type':'error','request_id':rid})\n"
        "    elif request['max_tokens'] == 256:\n"
        "        emit({'type':'delta','request_id':rid,'text':'token'})\n"
        "        end = time.monotonic()+2\n"
        "        while not cancelled and time.monotonic()<end:\n            time.sleep(.001)\n"
        "        emit({'type':'cancelled' if cancelled else 'error','request_id':rid})\n"
        "    else:\n"
        f"        event = {completion()!r}\n"
        "        event['request_id'] = rid\n"
        "        event['stats'].update(cached_prompt_tokens=0,evaluated_prompt_tokens=600)\n"
        "        emit(event)\n"
    )
    engine.chmod(0o755)
    result = idle.run(
        engine,
        "unused",
        "unused",
        measure=lambda _pid: 1024,
        memory_saver=True,
        short_only=True,
        timeout=3,
    )
    assert result["status"] == "complete" and result["reaped"] is True
    assert len(result["records"]) == 8
    requests = [json.loads(line) for line in wire.read_text().splitlines()]
    generations = [row for row in requests if row["type"] == "generate"]
    assert len(generations) == 8 and all(row["memory_saver"] is True for row in generations)
    assert requests[-1]["type"] == "shutdown"


@pytest.mark.parametrize(
    "mutation", [None, "empty", "depth", "tokens", "hash", "nan", "bool", "cache", "counters"]
)
def test_physical_tune_check_rejects_invalid_grids_and_cache(mutation):
    spec = importlib.util.spec_from_file_location(
        "transitions",
        Path(__file__).parents[1] / "docs/measurements/memory-saver/check-transitions.py",
    )
    transitions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transitions)
    event = {
        "type": "mtp_tune_complete",
        "rows": [
            {
                "depth": depth,
                "decode_tokens": 190,
                "decode_tps": 50.0,
                "decode_seconds": 3.8,
                "token_hashes": ["a", "b"],
                "accepted_tokens": 0,
                "proposed_tokens": 0,
            }
            for depth in range(4)
        ],
    }
    pong = {"type": "pong", "memory": {"mlx_cache_bytes": 0}}
    if mutation == "empty":
        event["rows"] = []
    elif mutation == "depth":
        event["rows"][3]["depth"] = 2
    elif mutation == "tokens":
        event["rows"][0]["decode_tokens"] = 0
    elif mutation == "hash":
        event["rows"][1]["token_hashes"][0] = "different"
    elif mutation == "nan":
        event["rows"][0]["decode_tps"] = float("nan")
    elif mutation == "bool":
        pong["memory"]["mlx_cache_bytes"] = False
    elif mutation == "cache":
        pong["memory"]["mlx_cache_bytes"] = 1
    elif mutation == "counters":
        event["rows"][1]["accepted_tokens"] = 1
    if mutation is None:
        transitions.validate_tune_result(event, pong)
    else:
        with pytest.raises(ValueError):
            transitions.validate_tune_result(event, pong)


@pytest.mark.parametrize("silent", [False, True])
def test_transition_check_saves_failure_and_reaps_real_child(tmp_path, monkeypatch, silent):
    import os
    import sys

    spec = importlib.util.spec_from_file_location(
        "transitions_failure",
        Path(__file__).parents[1] / "docs/measurements/memory-saver/check-transitions.py",
    )
    transitions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transitions)
    engine = tmp_path / "build/memory-saver/candidate-rs"
    engine.parent.mkdir(parents=True)
    engine.write_text(
        f"#!{sys.executable}\nimport json, os, sys, time\n"
        "print('pid=' + str(os.getpid()), file=sys.stderr, flush=True)\n"
        + (
            "print(json.dumps({'type':'ready','memory_saver_supported':True}), flush=True)\n"
            if silent
            else "print(json.dumps({'type':'error','message':'fixture failure'}), flush=True)\n"
        )
        + "time.sleep(30)\n"
    )
    engine.chmod(0o755)
    output = tmp_path / "failed.json"
    monkeypatch.setattr(transitions, "ROOT", tmp_path)
    monkeypatch.setattr(
        transitions.sys, "argv", ["check", str(output), "unused-model", "unused-head"]
    )
    if silent:
        original = transitions.idle.exchange
        monkeypatch.setattr(
            transitions.idle,
            "exchange",
            lambda child, request: original(child, request, timeout=0.1),
        )
    before = transitions.signal.getsignal(transitions.signal.SIGTERM)
    with pytest.raises(TimeoutError if silent else ValueError):
        transitions.main()
    actual = json.loads(output.read_text())
    assert actual["status"] == "failed" and actual["parity"] is False
    assert actual["reaped"] is True and actual["requests"] == []
    assert transitions.signal.getsignal(transitions.signal.SIGTERM) == before
    pid = int(actual["stderr"].split("pid=", 1)[1].splitlines()[0])
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
