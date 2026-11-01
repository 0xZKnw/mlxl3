"""Validate and summarize Cold readiness host spans; these are not GPU timings."""

import copy
import hashlib
import json
import math
import runpy
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
COMMON = ["begin", "wait", "finish"]
SPLIT = ["first_build", "snapshot_before_submit", "first_submit", "snapshot_after_submit"]


def batches(events):
    assert type(events) is list and events, "empty trace"
    children = defaultdict(list)
    calculations = defaultdict(list)
    for i, event in enumerate(events):
        start, end, parent = (event[k] for k in ("start_ns", "end_ns", "parent"))
        assert type(start) is int and type(end) is int and 0 <= start <= end, "interval"
        if parent is not None:
            assert type(parent) is int and 0 <= parent < i, "parent index"
            owner = events[parent]
            assert owner["start_ns"] <= start <= end <= owner["end_ns"], "containment"
        if event["name"].startswith("cold_ready_") and event["name"] != "cold_ready_batch":
            assert parent is not None and events[parent]["name"] == "cold_ready_batch", "orphan"
            children[parent].append(event)
        if event["name"] == "expert_calculate_build":
            calculations[parent].append(event)
    rows = []
    for i, event in enumerate(events):
        if event["name"] != "cold_ready_batch":
            continue
        d = event["data"]
        assert d["status"] == "finished" and d["io_ok"] is True, "incomplete batch"
        for key in ("jobs", "cold_rows", "ready_initial", "w1_rows", "w2_rows"):
            assert type(d[key]) is int and 0 <= d[key] <= 20, "counts"
        assert 2 < d["jobs"] <= d["cold_rows"] and type(d["split"]) is bool, "geometry"
        assert d["w1_rows"] + d["w2_rows"] == d["cold_rows"], "row conservation"
        names = [e["name"].removeprefix("cold_ready_") for e in children[i]]
        expected = ["begin", "wait", *SPLIT, "finish"] if d["split"] else COMMON
        assert names == expected, "missing, duplicate or reordered stages"
        assert all(a["end_ns"] <= b["start_ns"] for a, b in zip(children[i], children[i][1:])), (
            "overlap"
        )
        spans = dict(zip(names, children[i]))
        row = dict(d)
        row["ns"] = {name: e["end_ns"] - e["start_ns"] for name, e in spans.items()}
        row["ns"]["batch"] = event["end_ns"] - event["start_ns"]
        if d["split"]:
            assert 2 <= d["ready_initial"] < d["jobs"] and 0 < d["w1_rows"] < d["cold_rows"], (
                "split"
            )
            previous = d["ready_initial"]
            for suffix in ("before", "after"):
                snapshot = spans[f"snapshot_{suffix}_submit"]["data"]
                assert snapshot["status"] == "ok" and snapshot["jobs"] == d["jobs"], "snapshot"
                ready = snapshot["ready"]
                assert type(ready) is int and previous <= ready <= d["jobs"], "monotonic readiness"
                previous = row[f"ready_{suffix}"] = ready
        else:
            assert d["ready_initial"] == d["jobs"] and d["w1_rows"] == 0, "fallback"
        if event["parent"] is not None:
            tail = [e for e in calculations[event["parent"]] if e["start_ns"] >= event["end_ns"]]
            assert len(tail) == 1 and tail[0]["data"]["rows"] == d["w2_rows"], "second graph"
            row["w2_build_ns"] = tail[0]["end_ns"] - tail[0]["start_ns"]
        rows.append(row)
    assert rows, "no Cold batches"
    return rows


def summarize(rows, units):
    assert rows and type(units) is int and units > 0
    splits = [r for r in rows if r["split"]]
    durations = defaultdict(list)
    for row in rows:
        for name, ns in row["ns"].items():
            durations[name].append(ns)
    return {
        "units": units,
        "batches": len(rows),
        "split": len(splits),
        "all_ready_initial": len(rows) - len(splits),
        "all_ready_before_submit": sum(r["ready_before"] == r["jobs"] for r in splits),
        "partial_before_complete_after": sum(
            r["ready_before"] < r["jobs"] == r["ready_after"] for r in splits
        ),
        "partial_after_submit": sum(r["ready_after"] < r["jobs"] for r in splits),
        "wave_rows": dict(Counter(f"{r['w1_rows']}/{r['w2_rows']}" for r in splits)),
        "placement_estimates_ms_per_unit": {
            "observed_w2_build": sum(r.get("w2_build_ns", 0) for r in splits) / units / 1e6,
            "w2_overlap_min_build_and_finish": sum(
                min(r.get("w2_build_ns", 0), r["ns"]["finish"]) for r in splits
            )
            / units
            / 1e6,
            "submit_in_already_complete_batches": sum(
                r["ns"]["first_submit"] for r in splits if r["ready_before"] == r["jobs"]
            )
            / units
            / 1e6,
        },
        "host_times": {
            name: {
                "count": len(values),
                "sum_ms_per_unit": sum(values) / units / 1e6,
                "mean_us": statistics.mean(values) / 1e3,
                "median_us": statistics.median(values) / 1e3,
                "p95_us": sorted(values)[math.ceil(0.95 * len(values)) - 1] / 1e3,
                "max_us": max(values) / 1e3,
            }
            for name, values in durations.items()
        },
    }


def self_check():
    d = {
        "status": "finished",
        "io_ok": True,
        "jobs": 3,
        "cold_rows": 4,
        "ready_initial": 2,
        "w1_rows": 2,
        "w2_rows": 2,
        "split": True,
    }
    events = [{"name": "cold_ready_batch", "parent": None, "start_ns": 0, "end_ns": 100, "data": d}]
    for i, name in enumerate(["begin", "wait", *SPLIT, "finish"]):
        data = {"status": "ok", "ready": 3, "jobs": 3} if name.startswith("snapshot") else None
        events.append(
            {
                "name": f"cold_ready_{name}",
                "parent": 0,
                "start_ns": i * 10,
                "end_ns": i * 10 + 5,
                "data": data,
            }
        )
    result = summarize(batches(events), 2)
    assert (
        result["all_ready_before_submit"] == 1
        and result["host_times"]["batch"]["sum_ms_per_unit"] == 0.00005
    )
    for before, after, key in (
        (2, 3, "partial_before_complete_after"),
        (2, 2, "partial_after_submit"),
    ):
        e = copy.deepcopy(events)
        e[4]["data"]["ready"], e[6]["data"]["ready"] = before, after
        assert summarize(batches(e), 1)[key] == 1
    fallback = copy.deepcopy(events)
    fallback[0]["data"].update(split=False, ready_initial=3, w1_rows=0, w2_rows=4)
    fallback = [fallback[i] for i in (0, 1, 2, 7)]
    assert summarize(batches(fallback), 1)["all_ready_initial"] == 1
    bad_cases = [[], events[:-1], events + [events[-1]], events[1:]]
    nested = [
        {"name": "moe_forward_rows", "parent": None, "start_ns": 0, "end_ns": 120, "data": None},
        *copy.deepcopy(events),
    ]
    for event in nested[1:]:
        event["parent"] = 0 if event["parent"] is None else event["parent"] + 1
    nested.append(
        {
            "name": "expert_calculate_build",
            "parent": 0,
            "start_ns": 105,
            "end_ns": 115,
            "data": {"rows": 2},
        }
    )
    estimate = summarize(batches(nested), 1)["placement_estimates_ms_per_unit"]
    assert estimate["observed_w2_build"] == 0.00001
    assert estimate["w2_overlap_min_build_and_finish"] == 0.000005
    bad_cases.append(nested[:-1])
    wrong_shape = copy.deepcopy(nested)
    wrong_shape[-1]["data"]["rows"] = 3
    bad_cases.append(wrong_shape)
    for index, key, value in (
        (0, "status", "incomplete"),
        (0, "io_ok", False),
        (0, "w2_rows", 1),
        (0, "jobs", True),
        (4, "ready", 1),
        (6, "ready", 2),
        (4, "status", "diagnostic_error"),
    ):
        e = copy.deepcopy(events)
        e[index]["data"][key] = value
        bad_cases.append(e)
    for index, key, value in (
        (1, "start_ns", -1),
        (1, "end_ns", 101),
        (1, "parent", 1),
        (2, "start_ns", 4),
        (1, "start_ns", float("nan")),
    ):
        e = copy.deepcopy(events)
        e[index][key] = value
        bad_cases.append(e)
    for bad in bad_cases:
        try:
            batches(bad)
        except (AssertionError, KeyError):
            pass
        else:
            raise AssertionError("invalid trace accepted")
    global BASE
    previous = BASE
    try:
        with tempfile.TemporaryDirectory() as folder:
            BASE = Path(folder)
            (BASE / "diagnostic-ar-main").mkdir()
            (BASE / "diagnostic-ar-main/complete.json").write_text('{"status":"incomplete"}')
            try:
                main()
            except AssertionError:
                pass
            else:
                raise AssertionError("incomplete run accepted")
            result = json.loads((BASE / "diagnostic-analysis.json").read_text())
            assert result["status"] == "failed_analysis" and "parity" not in result
            before = (BASE / "diagnostic-analysis.json").read_bytes()
            try:
                main()
            except AssertionError:
                pass
            else:
                raise AssertionError("old report overwritten")
            assert (BASE / "diagnostic-analysis.json").read_bytes() == before
    finally:
        BASE = previous
    print(f"5 positive cases; {len(bad_cases)} malformed traces rejected")
    print("Incomplete run recorded as failed; prior result preserved")


def main():
    output = BASE / "diagnostic-analysis.json"
    assert not output.exists(), "preserve prior analysis"
    report = {"status": "incomplete", "files_sha256": {}}

    def read(path):
        raw = path.read_bytes()
        report["files_sha256"][str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    try:
        complete = read(BASE / "diagnostic-ar-main/complete.json")
        assert (
            complete["status"] == "complete_diagnostic_internal_full_parity_external_golden_pending"
        )
        assert (
            complete["kind"] == "natural"
            and complete["mtp"] is False
            and complete["dflash"] is False
        )
        phases = complete["summaries"]
        assert [s["phase"] for s in phases] == [
            "warmup",
            "control_before",
            "trace_1",
            "trace_2",
            "trace_3",
            "control_after",
        ]
        manifest = read(BASE / "manifest.json")
        golden = read(Path(manifest["external_golden_files_for_root_validation"]["MAIN"]))
        pool = runpy.run_path(str(BASE.parent / "issue58-20261029/pool-speed-driver.py"))
        all_rows = defaultdict(list)
        report["traces"] = {}
        for s in phases:
            assert (
                s["full_logits_target_head_states_byte_exact"] is True
                and s["logit_count"] == 248320
            )
            assert s["tokens"] == golden["tokens"] and len(s["tokens"]) == 128
            assert pool["token_hash"](s["tokens"]) == s["token_hash"]
            assert s["prompt"] == "MAIN" and len(s["prompt_ids"]) == 87 and len(s["steps"]) == 127
            pool["check_golden"](
                {
                    "token_hash": s["token_hash"],
                    "cache_context": s["text"],
                    "stats": {"memory": s["after"]["memory"]},
                },
                golden,
                128,
            )
            if s["phase"].startswith("trace_"):
                assert (
                    s["traced"] is True
                    and s["fenced"] is False
                    and s["test_stage_retention"] is False
                )
                for part, units in (("prefill", 87), ("decode", 127)):
                    name = f"{s['phase']}-{part}"
                    events = read(BASE / f"diagnostic-ar-main/{name}.json")
                    assert len(events) == s[f"{part}_event_count"]
                    rows = batches(events)
                    report["traces"][name] = summarize(rows, units)
                    all_rows[part].extend(rows)
        report["pooled"] = {
            part: summarize(rows, 3 * (87 if part == "prefill" else 127))
            for part, rows in all_rows.items()
        }
        report["status"] = "validated_host_diagnostic_external_golden_not_gpu_time_or_gain"
        report["parity"] = (
            "six phases full internal logits/states; external 128 token IDs/hash/text"
        )
        report["limits"] = (
            "Instrumentation perturbs scheduling. Nested host intervals are not additive GPU times. Completion after async_eval does not exclude GPU overlap."
        )
    except BaseException as error:
        report.update(status="failed_analysis", error=repr(error))
        raise
    finally:
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["pooled"], indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    else:
        assert len(sys.argv) == 1
        main()
