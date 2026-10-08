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

    def failed_run(_engine, _model, _head, *, result, progress, timeout):
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
