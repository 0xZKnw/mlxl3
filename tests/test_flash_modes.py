"""Mode/cost oracles and the complete matrix with disposable bridge children."""

import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

modes = runpy.run_path(str(Path(__file__).parents[1] / "scripts/benchmark_flash_modes.py"))


def entry(name, prompt="MAIN", tokens=128):
    dflash = name not in ("shipped-off", "shipped-mtp1")
    width = {
        "tuned-b2": 1,
        "tuned-b3": 2,
        "tuned-b4": 3,
        "tuned-precap": 3,
        "shipped-precap": 3,
    }.get(name, 7)
    hist = [0] * 8
    hist[width] = 70
    return {
        "prompt": prompt,
        "token_hash": prompt,
        "text": "nonempty " + prompt,
        "generation_mode": {
            "mtp_active": name == "shipped-mtp1",
            "dflash_active": dflash,
            "dflash_proposals": width if dflash else 0,
        },
        "stats": {
            "generated_tokens": tokens,
            "decode_tokens": tokens - 1,
            "decode_seconds": (tokens - 1) / 16,
            "decode_tps": 16,
            "ttft_seconds": 0.5,
            "prefill_tps": 100,
            "context_limit": 4096,
            "prompt_tokens": 64,
            "elapsed_seconds": 9,
            "end_to_end_ttft_seconds": 1,
            "mtp_blocks": 0,
            "mtp_proposed_tokens": 0,
            "mtp_accepted_tokens": 0,
            "dflash_blocks": 70,
            "dflash_proposed_tokens": width * 70,
            "dflash_accepted_tokens": 56,
            "dflash_block_seconds": 7,
            "dflash_profile": {
                "drafted_rows": (width + 1) * 70,
                "verified_rows": (width + 1) * 70,
                "drafted_widths": hist,
                "verified_widths": hist.copy(),
                "expert_demand_bytes": 1_248_012 * 100,
                "draft_seconds": 0.5,
                "verification_seconds": 6,
                "commit_cache_seconds": 0.1,
            },
        },
    }


def child(name):
    return {
        "status": "validated",
        "reaped": True,
        "reader_joined": True,
        "warmup": [entry(name, p) for p in ("MAIN", "HOLD")],
        "runs": [entry(name, p) for p in ("MAIN", "HOLD")],
    }


def arms():
    return [
        {"index": i, "profile": n, "report": child(n)}
        for i, n in enumerate((*modes["PROFILES"], *reversed(modes["PROFILES"])))
    ]


def test_complete_matrix_checks_target_on_both_orders_and_every_prompt():
    data = arms()
    result = modes["summarize"](data, 128, 1, 15)
    assert result["shipped-off"]["MAIN"]["target_reached"]
    assert (
        result["tuned-b2"]["HOLD"]["measurements"][0]["verified_rows_per_decode_token"] == 140 / 127
    )
    data[-1]["report"]["runs"][0]["stats"].update(decode_tps=14.9, decode_seconds=127 / 14.9)
    result = modes["summarize"](data, 128, 1, 15)
    assert not result["shipped-off"]["MAIN"]["target_reached"]
    assert not result["shipped-off"]["MAIN"]["stable_within_3_percent"]
    assert result["shipped-off"]["HOLD"]["target_reached"]


@pytest.mark.parametrize(
    "bad",
    [
        "partial",
        "duplicate",
        "status",
        "unreaped",
        "reader",
        "missing",
        "foreign",
        "warm_diverge",
        "diverge",
    ],
)
def test_partial_or_divergent_reports_cannot_qualify(bad):
    data = arms()
    report = data[-1]["report"]
    if bad == "partial":
        data.pop()
    elif bad == "duplicate":
        data[-1] = copy.deepcopy(data[0])
    elif bad == "status":
        report["status"] = "failed"
    elif bad == "unreaped":
        report["reaped"] = False
    elif bad == "reader":
        report["reader_joined"] = False
    elif bad == "missing":
        report["runs"].pop()
    elif bad == "foreign":
        report["runs"][0]["prompt"] = "OTHER"
    elif bad == "warm_diverge":
        report["warmup"][0]["token_hash"] = "different"
    else:
        report["runs"][0]["text"] = "different"
    with pytest.raises(ValueError):
        modes["summarize"](data, 128, 1, 15)


@pytest.mark.parametrize(
    "bad",
    [
        "nan",
        "bool",
        "negative",
        "width",
        "hist_len",
        "hist_bool",
        "hist_count",
        "rows",
        "bytes",
        "time",
        "adaptive",
        "empty",
    ],
)
def test_fixed_cost_oracle_rejects_invalid_or_ignored_settings(bad):
    data = entry("tuned-b2")
    stats = data["stats"]
    profile = stats["dflash_profile"]
    if bad == "nan":
        stats["decode_tps"] = float("nan")
    elif bad == "bool":
        profile["expert_demand_bytes"] = True
    elif bad == "negative":
        profile["draft_seconds"] = -1
    elif bad == "width":
        data["generation_mode"]["dflash_proposals"] = 3
    elif bad == "hist_len":
        profile["drafted_widths"] = []
    elif bad == "hist_bool":
        profile["drafted_widths"][0] = False
    elif bad == "hist_count":
        profile["drafted_widths"][1] -= 1
    elif bad == "rows":
        profile["verified_rows"] += 1
    elif bad == "bytes":
        profile["expert_demand_bytes"] += 1
    elif bad == "time":
        stats["dflash_block_seconds"] = 1
    elif bad == "adaptive":
        profile["verified_widths"] = [35, 35, 0, 0, 0, 0, 0, 0]
    else:
        stats["dflash_blocks"] = 0
    with pytest.raises(ValueError):
        modes["entry_metrics"](data, "tuned-b2", 128)


def test_clean_environment_and_distinct_proposal_count(monkeypatch):
    monkeypatch.setenv("MLXL3_FLASH_DRAFT_WIDTH", "7")
    monkeypatch.setenv("MLXL3_SSD_CACHE_GIB", "12")
    monkeypatch.setenv("MLXL3_UNRELATED_EXPERIMENT", "1")
    env, options = modes["configuration"]("shipped-auto")
    assert not any(k.startswith("MLXL3_") for k in env) and options["auto_cache"]
    for name, width in (("tuned-b2", "1"), ("tuned-b3", "2"), ("tuned-b4", "3")):
        env, options = modes["configuration"](name)
        assert env["MLXL3_FLASH_DRAFT_WIDTH"] == width
        assert env["MLXL3_FLASH_DRAFT_ADAPTIVE"] == "0" and not options["auto_cache"]
        assert "MLXL3_UNRELATED_EXPERIMENT" not in env
    with pytest.raises(ValueError):
        modes["configuration"]("b2")


@pytest.mark.parametrize("prompt", [0, -1, True, 3968, 4096])
def test_context_clipping_is_rejected(prompt):
    data = entry("tuned-b2")
    data["stats"]["prompt_tokens"] = prompt
    with pytest.raises(ValueError):
        modes["entry_metrics"](data, "tuned-b2", 128)


@pytest.mark.parametrize("memory", [[], {"bytes": float("nan")}, {"bytes": True}, {"bytes": -1}])
def test_invalid_memory_cannot_break_failure_report_serialization(memory):
    data = entry("shipped-off")
    data["stats"]["memory"] = memory
    with pytest.raises(ValueError):
        modes["entry_metrics"](data, "shipped-off", 128)


def test_zero_and_unavailable_memory_are_preserved():
    data = entry("shipped-off")
    data["stats"]["memory"] = {"bytes": 0, "optional": None}
    assert modes["entry_metrics"](data, "shipped-off", 128)["memory"] == data["stats"]["memory"]


@pytest.mark.parametrize("fault", ["ok", "diverge", "interrupt", "power", "changed_binary", "nan"])
def test_persistent_campaign_status_and_no_false_parity(tmp_path, monkeypatch, fault):
    engine = tmp_path / "engine"
    engine.write_bytes(b"engine")
    output = tmp_path / "results"
    run = modes["run"]
    monkeypatch.setitem(
        run.__globals__,
        "conditions",
        lambda: {"pmset batt": "AC Power" if fault != "power" else "Battery Power"},
    )
    calls = []

    def runner(*args, **kwargs):
        name = (*modes["PROFILES"], *reversed(modes["PROFILES"]))[len(calls)]
        calls.append(name)
        if fault == "interrupt":
            raise KeyboardInterrupt()
        report = child(name)
        report["binary_sha256"] = hashlib.sha256(engine.read_bytes()).hexdigest()
        if fault == "diverge" and len(calls) == 2:
            report["runs"][0]["token_hash"] = "bad"
        if fault == "changed_binary":
            engine.write_bytes(b"changed")
        if fault == "nan":
            report["runs"][0]["stats"]["decode_tps"] = float("nan")
        return report

    if fault == "ok":
        result = run(engine, tmp_path, tmp_path, output, runner=runner)
        assert (
            len(calls) == 2 * len(modes["PROFILES"])
            and result["status"] == "completed"
            and result["parity"]
        )
    else:
        with pytest.raises((ValueError, RuntimeError, KeyboardInterrupt)):
            run(engine, tmp_path, tmp_path, output, runner=runner)
        result = json.loads((output / "campaign.json").read_text())
        assert not result["parity"] and "summary" not in result
        assert result["status"] == ("interrupted" if fault == "interrupt" else "failed")
        assert len(calls) <= 2


def test_real_bridge_matrix_uses_all_profiles_and_reaps_children(tmp_path, monkeypatch):
    engine = tmp_path / "engine"
    # The child supplies deterministic protocol data, never a model or user registry.
    engine.write_text(
        "#!/usr/bin/env python3\n"
        + """
import hashlib, json, os, sys
print('{"type":"ready"}', flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request['type'] == 'shutdown': break
    width = int(os.environ.get('MLXL3_FLASH_DRAFT_WIDTH', '3' if os.environ.get('MLXL3_FLASH_DRAFT_PRECAP') == '1' else '7'))
    mtp, df = request['mtp'], request['dflash2']
    hist = [0]*8
    hist[width] = 70
    rid = request['request_id']
    text = request['messages'][0]['content']
    print(json.dumps(dict(type='generation_mode', request_id=rid, mtp_active=mtp, dflash_active=df, dflash_proposals=width if df else 0)), flush=True)
    stats = dict(generated_tokens=128, decode_tokens=127, decode_seconds=127/16, decode_tps=16, ttft_seconds=.5, prefill_tps=100,
        context_limit=4096, prompt_tokens=64, elapsed_seconds=9, end_to_end_ttft_seconds=1,
        mtp_proposed_tokens=0, mtp_accepted_tokens=0, mtp_blocks=0,
        dflash_blocks=70, dflash_proposed_tokens=70*width, dflash_accepted_tokens=56, dflash_block_seconds=7,
        dflash_profile=dict(drafted_rows=(width+1)*70, verified_rows=(width+1)*70, drafted_widths=hist, verified_widths=hist,
            expert_demand_bytes=1248012*100, draft_seconds=.5, verification_seconds=6, commit_cache_seconds=.1))
    print(json.dumps(dict(type='complete', request_id=rid, token_hash=hashlib.sha256(text.encode()).hexdigest(), cache_context=text, stats=stats)), flush=True)
"""
    )
    engine.chmod(0o755)
    monkeypatch.setitem(modes["run"].__globals__, "conditions", lambda: {"pmset batt": "AC Power"})
    report = modes["run"](engine, tmp_path, tmp_path, tmp_path / "out")
    assert report["status"] == "completed" and report["parity"]
    assert all(a["report"]["reaped"] and a["report"]["reader_joined"] for a in report["arms"])
    assert "MLXL3_FLASH_DRAFT_WIDTH" not in report["arms"][2]["report"]["environment"]
    assert report["arms"][5]["report"]["environment"]["MLXL3_FLASH_DRAFT_WIDTH"] == "1"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, True])
def test_invalid_target_cannot_be_a_performance_success(value):
    with pytest.raises(ValueError):
        modes["summarize"](arms(), 128, 1, value)
