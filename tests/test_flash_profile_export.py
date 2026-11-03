"""Exported MTP blocks conserve real target rows, offset commits and I/O."""

import copy
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "scripts/summarize_flash_profile.py"))


def fixture():
    events = []

    def add(name, parent, start, end, data=None):
        index = len(events)
        events.append(
            {"name": name, "parent": parent, "start_ns": start, "end_ns": end, "data": data}
        )
        return index

    add("decode", None, 0, 100000000)
    first = {
        "step": 1,
        "token": 11,
        "offset": 4,
        "seconds": 0.01,
        "elapsed_seconds": 0.01,
        "mtp_proposed": 1,
        "mtp_accepted": 1,
        "mtp_blocks": 1,
    }
    queued = {
        "step": 2,
        "token": 12,
        "offset": 4,
        "seconds": 0.0003,
        "elapsed_seconds": 0.011,
        "mtp_proposed": 0,
        "mtp_accepted": 0,
        "mtp_blocks": 0,
    }
    step = add("output_step", 0, 1, 10000000, first)
    add("mtp_head_proposal", step, 100, 1000)
    verify = add("verify_pair", step, 2000, 9000000)
    for layer in range(48):
        begin = 3000 + layer * 1000
        l = add(f"layer_{layer}", verify, begin, begin + 900)
        add("route_readback_inclusive", l, begin + 1, begin + 10)
        mapping = [[i, not (layer == 0 and i == 0)] for i in range(20)]
        m = add(
            "moe_forward_rows",
            l,
            begin + 20,
            begin + 800,
            {
                "rows": 2,
                "capacity": 53,
                "ids": list(range(20)),
                "slots_hit": mapping,
                "hot_rows": 19 if layer == 0 else 20,
                "cold_rows": 1 if layer == 0 else 0,
            },
        )
        if layer == 0:
            add(
                "disk_read_batch",
                m,
                begin + 30,
                begin + 40,
                {
                    "ok": True,
                    "jobs": [[0, 0]],
                    "workers": 4,
                    "timings_job_worker_start_end_bytes_failed": [[0, 0, 0, 7, 1248012, 0]],
                },
            )
    add("commit_pair", step, 9000010, 9000100)
    add("output_step", 0, 10000001, 11000000, queued)
    summary = {
        "mtp": True,
        "prompt_ids": [1, 2],
        "tokens": [10, 11, 12],
        "steps": [first, queued],
        "mtp_proposed": 1,
        "mtp_accepted": 1,
        "mtp_blocks": 1,
        "decode_seconds": 0.1,
    }
    return events, summary


def test_acceptance_queue_real_rows_and_bytes_have_independent_expected_totals():
    events, summary = fixture()
    rows, stats = API["output_steps"](events, summary)
    assert [r["classification"] for r in rows] == ["accepted", "queued"]
    assert rows[0]["target_rows"] == 2 and rows[0]["target_moe_calls"] == 48
    assert rows[0]["hits"] == 959 and rows[0]["misses"] == 1
    assert rows[0]["returned_bytes"] == 1248012 and rows[0]["ssd_jobs"] == 1
    assert rows[0]["route_readback_ns"] == 48 * 9 and rows[0]["ssd_batch_wall_ns"] == 10
    assert rows[1]["target_rows"] == rows[1]["returned_bytes"] == 0
    assert API["clocks"](events, summary, "decode", stats) == 100000000


def draft_fixture(width, accepted, fallback):
    events, steps = [], []

    def add(name, parent, start, end, data=None):
        index = len(events)
        events.append(
            {"name": name, "parent": parent, "start_ns": start, "end_ns": end, "data": data}
        )
        return index

    count = 1 + accepted + int(fallback)
    add("decode", None, 0, (count + 1) * 10000000)
    offset = 2
    for i in range(count):
        block = i == 0
        terminal = fallback and i == count - 1
        delta = accepted + 1 if block else int(terminal)
        offset += delta
        computed = width + 1 if block else int(terminal)
        data = {
            "step": i + 1,
            "token": 11 + i,
            "offset": offset,
            "target_rows": delta,
            "seconds": 0.01,
            "elapsed_seconds": 0.01 * (i + 1),
            "mtp_proposed": 0,
            "mtp_accepted": 0,
            "mtp_blocks": 0,
            "dflash": {
                "proposed": width if block else 0,
                "accepted": accepted if block else 0,
                "blocks": int(block),
                "verified_rows": width + 1 if block else 0,
                "drafted_rows": width + 1 if block else 0,
                "expert_demand_bytes": 0,
            },
        }
        steps.append(data)
        start = i * 10000000
        step = add("output_step", 0, start + 1, start + 9999999, data)
        parent = step
        if block:
            add("dflash_proposal_inclusive", step, start + 100, start + 1000)
            parent = add("dflash_verification_inclusive", step, start + 2000, start + 9000000)
        if computed:
            for layer in range(48):
                begin = start + 3000 + layer * 1000
                layer_id = add(f"layer_{layer}", parent, begin, begin + 900)
                add(
                    "moe_forward_rows",
                    layer_id,
                    begin + 20,
                    begin + 800,
                    {
                        "rows": computed,
                        "capacity": 53,
                        "ids": [j % 10 for j in range(10 * computed)],
                        "slots_hit": [[j % 10, True] for j in range(10 * computed)],
                        "hot_rows": 10 * computed,
                        "cold_rows": 0,
                    },
                )
        if block:
            add("dflash_commit_cache_build", step, start + 9000010, start + 9000100)
    summary = {
        "mtp": False,
        "dflash": True,
        "prompt_ids": [1, 2],
        "tokens": [10] + [r["token"] for r in steps],
        "steps": steps,
        "mtp_proposed": 0,
        "mtp_accepted": 0,
        "mtp_blocks": 0,
        "decode_seconds": (count + 1) * 0.01,
        "dflash_stats": [width, accepted, 1, 0.01],
        "dflash_profile": {
            "verified_rows": width + 1,
            "drafted_rows": width + 1,
            "expert_demand_bytes": 0,
        },
    }
    return events, summary


@pytest.mark.parametrize(
    "width,accepted,fallback",
    [(w, a, f) for w in range(1, 8) for a in range(w + 1) for f in (False, True)],
)
def test_draft_all_supported_widths_partial_acceptance_queue_and_terminal_fallback(
    width, accepted, fallback
):
    events, summary = draft_fixture(width, accepted, fallback)
    rows, stats = API["output_steps"](events, summary)
    assert rows[0]["classification"] == (
        "draft_accepted" if accepted == width else "draft_rejected"
    )
    assert rows[0]["target_rows"] == width + 1 and rows[0]["committed_rows"] == accepted + 1
    assert rows[0]["target_rows_sum"] == 48 * (width + 1)
    assert sum(r["classification"] == "queued" for r in rows) == accepted
    assert sum(r["classification"] == "fallback" for r in rows) == int(fallback)
    assert sum(r["target_rows"] for r in rows) == width + 1 + int(fallback)
    assert sum(r["committed_rows"] for r in rows) == accepted + 1 + int(fallback)
    assert API["clocks"](events, summary, "decode", stats) == (len(rows) + 1) * 10000000


@pytest.mark.parametrize(
    "fault",
    ["span", "rows", "committed", "accepted", "bytes", "aggregate", "queued", "mode", "bool"],
)
def test_draft_export_rejects_impossible_or_incomplete_verification(fault):
    events, summary = draft_fixture(7, 3, True)
    first = summary["steps"][0]
    if fault == "span":
        next(e for e in events if e["name"] == "dflash_verification_inclusive")["name"] = "missing"
    elif fault == "rows":
        first["dflash"]["verified_rows"] = 4
    elif fault == "committed":
        first["target_rows"] = 8
    elif fault == "accepted":
        first["dflash"]["accepted"] = 8
    elif fault == "bytes":
        first["dflash"]["expert_demand_bytes"] = 1
    elif fault == "aggregate":
        summary["dflash_profile"]["verified_rows"] = 4
    elif fault == "queued":
        summary["steps"][1]["offset"] += 1
    elif fault == "mode":
        summary["mtp"] = True
    elif fault == "bool":
        first["dflash"]["blocks"] = True
    with pytest.raises(AssertionError):
        API["output_steps"](events, summary)


def test_manifest_rejects_missing_or_duplicate_cases_without_claiming_parity(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    case = {"output": str(base / "case/profile")}
    for index, matrix in enumerate([{"cases": [case]}, {"cases": [case, case]}, {"cases": []}]):
        output = tmp_path / str(index)
        with pytest.raises(AssertionError):
            API["summarize"](base, output, matrix)
        result = json.loads((output / "result.json").read_text())
        assert result["status"] == "failed_or_interrupted" and result["parity"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "verify",
        "commit",
        "head",
        "target_row",
        "offset",
        "queued_offset",
        "acceptance",
        "proposal_count",
        "steps",
        "empty",
        "bool",
        "wrong_seconds",
    ],
)
def test_export_rejects_incomplete_or_impossible_blocks(fault):
    events, summary = fixture()
    if fault in ["verify", "commit", "head"]:
        target = {"verify": "verify_pair", "commit": "commit_pair", "head": "mtp_head_proposal"}[
            fault
        ]
        next(e for e in events if e["name"] == target)["name"] = "missing"
    elif fault == "target_row":
        e = next(
            e for e in events if e["name"] == "moe_forward_rows" and e["data"]["hot_rows"] == 20
        )
        d = e["data"]
        d.update(rows=1, ids=d["ids"][:10], slots_hit=d["slots_hit"][:10], hot_rows=10)
    elif fault == "offset":
        summary["steps"][0]["offset"] = 3
    elif fault == "queued_offset":
        summary["steps"][1]["offset"] = 5
    elif fault == "acceptance":
        summary["steps"][0]["mtp_accepted"] = 2
    elif fault == "proposal_count":
        summary["mtp_proposed"] = 2
    elif fault == "steps":
        summary["steps"].pop()
    elif fault == "empty":
        events = []
    elif fault == "bool":
        summary["steps"][0]["mtp_proposed"] = True
    elif fault == "wrong_seconds":
        summary["decode_seconds"] = 100
    with pytest.raises((AssertionError, IndexError)):
        _rows, stats = API["output_steps"](events, summary)
        API["clocks"](events, summary, "decode", stats)


def test_clock_and_host_self_conservation_detect_corrupt_timing_units():
    events, summary = fixture()
    _, stats = API["output_steps"](events, summary)
    wrong = copy.deepcopy(events)
    for e in wrong:
        e["start_ns"] *= 1000
        e["end_ns"] *= 1000
    with pytest.raises(AssertionError):
        API["clocks"](wrong, summary, "decode", stats)
    stats["components"][0]["host_self_ns"]["total"] += 1
    with pytest.raises(AssertionError):
        API["clocks"](events, summary, "decode", stats)


def test_prefill_includes_post_graph_greedy_readback_without_losing_or_double_counting_time():
    events, summary = fixture()
    events[0]["name"] = "prefill"
    events.append(
        {
            "name": "host_readback",
            "parent": None,
            "start_ns": 100000010,
            "end_ns": 100000020,
            "data": {
                "dtype": "UInt32",
                "shape": [1],
                "sample": {
                    "wait_ns": 8,
                    "copy_ns": 1,
                    "contiguous_build_ns": 0,
                    "bytes": 4,
                    "ready_before": False,
                    "direct": True,
                    "total_ns": 9,
                },
            },
        }
    )
    summary["prefill_seconds"] = 0.100000020
    stats = API["API"]["validate_events"](events)
    assert API["clocks"](events, summary, "prefill", stats) == 100000020
    assert sum(c["host_self_ns"]["total"] for c in stats["components"]) == 100000010
    events[-1]["name"] = "unaccounted_work"
    with pytest.raises(AssertionError):
        API["clocks"](events, summary, "prefill", stats)
    events[-1]["name"] = "host_readback"
    events[-1]["start_ns"] = 99999999
    with pytest.raises(AssertionError):
        API["clocks"](events, summary, "prefill", stats)


def test_empty_export_is_explicit_failure_and_never_overwrites(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    out = tmp_path / "output"
    with pytest.raises(AssertionError):
        API["summarize"](base, out)
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "failed_or_interrupted" and result["parity"] is False
    old = (out / "result.json").read_bytes()
    with pytest.raises(AssertionError):
        API["summarize"](base, out)
    assert (out / "result.json").read_bytes() == old


@pytest.mark.parametrize("wall", [float("nan"), float("inf"), 0, -1, True])
def test_nonfinite_or_invalid_wall_is_rejected(wall):
    events, summary = fixture()
    _, stats = API["output_steps"](events, summary)
    summary["decode_seconds"] = wall
    with pytest.raises(AssertionError):
        API["clocks"](events, summary, "decode", stats)
