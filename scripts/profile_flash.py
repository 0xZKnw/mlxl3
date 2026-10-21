"""Validate and summarize Flash53 diagnostic traces; never infer exclusive GPU time."""

import csv
import gzip
import hashlib
import json
import math
import re
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

EXPERT_BYTES = 1_248_012
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/measurements/flash-next-ssd-impl/opt-53-detailed-profile"


def integer(value):
    assert type(value) is int and value >= 0, "invalid unsigned integer"
    return value


def distribution(values):
    assert values and all(math.isfinite(v) and v >= 0 for v in values)
    ordered = sorted(values)
    return {
        "count": len(values),
        "total": sum(values),
        "min": ordered[0],
        "p50": ordered[math.ceil(len(values) * 0.50) - 1],
        "p90": ordered[math.ceil(len(values) * 0.90) - 1],
        "p95": ordered[math.ceil(len(values) * 0.95) - 1],
        "p99": ordered[math.ceil(len(values) * 0.99) - 1],
        "max": ordered[-1],
    }


def validate_events(events):
    assert isinstance(events, list) and events, "empty trace"
    children = [0] * len(events)
    paths, layers, ends = [], [], {}
    groups = defaultdict(lambda: {"wall": [], "self": []})
    routes, batches, jobs = [], [], []
    for i, event in enumerate(events):
        name = event["name"]
        assert isinstance(name, str) and name
        begin, end = integer(event["start_ns"]), integer(event["end_ns"])
        assert end >= begin
        parent = event["parent"]
        layer = None
        if parent is not None:
            assert type(parent) is int and 0 <= parent < i
            p = events[parent]
            assert p["start_ns"] <= begin <= end <= p["end_ns"], "invalid nesting"
            assert begin >= ends.get(parent, p["start_ns"]), "overlapping siblings"
            ends[parent] = end
            children[parent] += end - begin
            layer = layers[parent]
        if name.startswith("layer_"):
            layer = int(name.removeprefix("layer_"))
            assert 0 <= layer < 48
        path = (paths[parent] + "/" if parent is not None else "") + name
        paths.append(path)
        layers.append(layer)
        data = event["data"]
        if name == "moe_forward_rows":
            assert isinstance(data, dict)
            rows, cap = integer(data["rows"]), integer(data["capacity"])
            ids, slots = data["ids"], data["slots_hit"]
            assert 1 <= rows <= 8 and 10 <= cap <= 512
            assert len(ids) == len(slots) == 10 * rows
            assert all(type(n) is int and 0 <= n < 512 for n in ids)
            assert all(type(n) is int and 0 <= n < cap and type(hit) is bool for n, hit in slots)
            hot = sum(hit for _, hit in slots)
            assert data["hot_rows"] == hot and data["cold_rows"] == len(ids) - hot
            assert all(len(set(ids[j : j + 10])) == 10 for j in range(0, len(ids), 10))
            routes.append(
                {
                    "layer": layer,
                    "rows": rows,
                    "hot": hot,
                    "cold": len(ids) - hot,
                    "all_ready": hot == len(ids),
                    "unique": len(set(ids)),
                    "pairs": len(ids) - len(set(ids)),
                    "capacity": cap,
                }
            )
        if name == "disk_read_batch":
            assert data["ok"] is True
            timings, mapping = data["timings_job_worker_start_end_bytes_failed"], data["jobs"]
            workers = integer(data["workers"])
            assert 1 <= workers <= 4 and 1 <= len(mapping) == len(timings) <= 512
            assert len({slot for _, slot in mapping}) == len(mapping)
            worker_end = {}
            for index, row in enumerate(timings):
                assert len(row) == 6
                job, worker, start, stop, count, failed = map(integer, row)
                assert job == index and worker == index % min(workers, len(mapping))
                assert start <= stop and start >= worker_end.get(worker, 0)
                assert count == EXPERT_BYTES and failed == 0
                worker_end[worker] = stop
                jobs.append(
                    {
                        "layer": layer,
                        "worker": worker,
                        "start_ns": start,
                        "end_ns": stop,
                        "duration_ns": stop - start,
                        "bytes": count,
                    }
                )
            batches.append(
                {
                    "layer": layer,
                    "jobs": len(mapping),
                    "last_completion_ns": max(t[3] for t in timings),
                    "worker_finish_spread_ns": max(worker_end.values()) - min(worker_end.values()),
                    "wall_ns": end - begin,
                }
            )
    for i, event in enumerate(events):
        duration = event["end_ns"] - event["start_ns"]
        assert children[i] <= duration
        key = (layers[i], event["name"])
        groups[key]["wall"].append(duration)
        groups[key]["self"].append(duration - children[i])
    assert routes, "trace has no actual routed experts"
    components = [
        {
            "layer": layer,
            "component": name,
            "wall_ns": distribution(v["wall"]),
            "host_self_ns": distribution(v["self"]),
        }
        for (layer, name), v in groups.items()
    ]
    return {
        "event_count": len(events),
        "components": components,
        "routes": routes,
        "disk_batches": batches,
        "disk_jobs": jobs,
        "limits": "nested host wall/self intervals; not exclusive GPU; per-job read intervals overlap across workers",
    }


def vm_counts(raw):
    assert isinstance(raw, str) and re.search(r"page size of ([1-9][0-9]*) bytes", raw)
    keys = ["Compressions", "Decompressions", "Swapins", "Swapouts", "Pages occupied by compressor"]
    result = {}
    for key in keys:
        values = re.findall(rf"^{re.escape(key)}:\s+(\d+)\.\s*$", raw, re.MULTILINE)
        assert len(values) == 1, f"missing/duplicate counter: {key}"
        result[key] = int(values[0])
    return result


def validate_summary(summary, kind, golden, expected_tokens=None):
    budget = {"quality": 17, "fenced": 33, "natural": 128, "verifier": 64}[kind]
    assert summary["full_logits_target_head_states_byte_exact"] is True
    assert summary["logit_count"] == 248320
    tokens = summary["tokens"]
    assert len(tokens) == budget and all(type(n) is int and 0 <= n < 248320 for n in tokens)
    hash_value = 0xCBF29CE484222325
    for token in tokens:
        for byte in token.to_bytes(4, "little"):
            hash_value = ((hash_value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    assert summary["token_hash"] == f"{hash_value:016x}"
    assert summary["text"] and golden["cache_context"].startswith(summary["text"])
    if kind == "natural":
        assert (
            summary["token_hash"] == golden["token_hash"]
            and summary["text"] == golden["cache_context"]
        )
    if expected_tokens is not None:
        assert tokens == expected_tokens[:budget]
    assert len(summary["steps"]) == budget - 1
    for i, row in enumerate(summary["steps"], 1):
        assert row["step"] == i and row["token"] == tokens[i]
        assert math.isfinite(row["seconds"]) and row["seconds"] >= 0
    for key in ("prefill_seconds", "decode_seconds", "diagnostic_tps"):
        assert math.isfinite(summary[key]) and summary[key] > 0
    assert math.isclose(
        summary["diagnostic_tps"], (budget - 1) / summary["decode_seconds"], rel_tol=1e-12
    )
    counts = [summary[k] for k in ("before_counts", "prefill_counts", "after_counts")]
    assert all(len(rows) == 48 for rows in counts)
    for before, prefill, after in zip(*counts, strict=True):
        assert len(before) == len(prefill) == len(after) == 3
        assert all(
            0 <= integer(a) <= integer(b) <= integer(c)
            for a, b, c in zip(before, prefill, after, strict=True)
        )
    vm = []
    for key in ("before", "prefill_boundary", "after"):
        snap = summary[key]
        memory = snap["memory"]
        assert (
            0
            < memory["process_footprint_bytes"]
            <= memory["process_lifetime_peak_bytes"]
            <= 12_000_000_000
        )
        vm.append(vm_counts(snap["vm_stat"]))
    for i in (0, 1):
        assert all(
            vm[i + 1][key] >= vm[i][key]
            for key in ("Compressions", "Decompressions", "Swapins", "Swapouts")
        )
    return {
        "prefill_vm_counter_pages_delta": {k: vm[1][k] - vm[0][k] for k in vm[0]},
        "decode_vm_counter_pages_delta": {k: vm[2][k] - vm[1][k] for k in vm[0]},
        "limits": "global VM pages/events, not model-exclusive compressed bytes; compressor occupancy is a gauge",
    }


def load_events(path):
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


def analyze(directory, kind, golden, expected_tokens=None):
    complete = json.loads((directory / "complete.json").read_text())
    assert complete["status"] == "complete_diagnostic_internal_full_parity_external_golden_pending"
    expected_phases = {
        "verifier": [
            "sequential_oracle",
            "warmup",
            "control_before",
            "trace_1",
            "trace_2",
            "control_after",
        ],
        "natural": ["warmup", "control_before", "trace_1", "trace_2", "trace_3", "control_after"],
        "fenced": ["warmup", "oracle", "fenced_1", "fenced_2"],
        "quality": ["warmup", "oracle", "trace", "fenced"],
    }[kind]
    assert [s["phase"] for s in complete["summaries"]] == expected_phases
    report = {
        "status": "incomplete",
        "kind": kind,
        "phases": [],
        "load_seconds": complete["load_seconds"],
    }
    for summary in complete["summaries"]:
        item = {
            "phase": summary["phase"],
            "vm": validate_summary(summary, kind, golden, expected_tokens),
            "diagnostic_tps": summary["diagnostic_tps"],
            "prefill_seconds": summary["prefill_seconds"],
            "decode_seconds": summary["decode_seconds"],
            "parts": {},
        }
        if summary["traced"]:
            for part in ("prefill", "decode"):
                events = load_events(directory / f"{summary['phase']}-{part}.json")
                assert len(events) == summary[f"{part}_event_count"]
                stats = validate_events(events)
                layer_routes = [r for r in stats["routes"] if r["layer"] is not None]
                assert {r["layer"] for r in layer_routes} == set(range(48))
                before, after = (
                    ("before_counts", "prefill_counts")
                    if part == "prefill"
                    else ("prefill_counts", "after_counts")
                )
                expected_bytes = sum(
                    b[2] - a[2] for a, b in zip(summary[before], summary[after], strict=True)
                )
                assert (
                    sum(j["bytes"] for j in stats["disk_jobs"] if j["layer"] is not None)
                    == expected_bytes
                )
                for layer in range(48):
                    routes = [r for r in layer_routes if r["layer"] == layer]
                    assert (
                        sum(r["hot"] for r in routes)
                        == summary[after][layer][0] - summary[before][layer][0]
                    )
                    assert (
                        sum(r["cold"] for r in routes)
                        == summary[after][layer][1] - summary[before][layer][1]
                    )
                item["parts"][part] = stats
        report["phases"].append(item)
    report["status"] = "validated_diagnostic_external_golden_full_internal_parity"
    output = directory / "analysis.json"
    assert not output.exists(), "refusing to overwrite analysis"
    output.write_text(json.dumps(report, indent=2) + "\n")
    with (directory / "components.csv").open("x", newline="") as stream:
        rows = []
        for phase in report["phases"]:
            for part, stats in phase["parts"].items():
                for component in stats["components"]:
                    rows.append(
                        {
                            "phase": phase["phase"],
                            "part": part,
                            "layer": component["layer"],
                            "component": component["component"],
                            **component["wall_ns"],
                            "host_self_total_ns": component["host_self_ns"]["total"],
                        }
                    )
        assert rows
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report


def compress_raw(directory):
    for path in directory.glob("*-prefill.json"):
        compress(path)
    for path in directory.glob("*-decode.json"):
        compress(path)


def compress(path):
    destination = Path(str(path) + ".gz")
    assert not destination.exists()
    with (
        path.open("rb") as source,
        destination.open("xb") as target,
        gzip.GzipFile(filename="", fileobj=target, mode="wb", mtime=0) as zipped,
    ):
        for block in iter(lambda: source.read(1 << 20), b""):
            zipped.write(block)
    with gzip.open(destination, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    path.unlink()


def cumulative_monitor(monitor, baseline):
    assert set(baseline) == {"Swapins", "Swapouts"}
    assert all(type(n) is int and n >= 0 for n in baseline.values())
    baseline = baseline.copy()

    def guarded(child, record, parse_vm):
        before = record["before"]["vm_bytes"]
        saved = {key: before[key] for key in baseline}
        assert all(baseline[key] <= saved[key] for key in baseline), "future VM baseline"
        record["live_guard_basis"] = {"scope": "cumulative_campaign", **baseline}
        before.update(baseline)
        try:
            return monitor(child, record, parse_vm)
        finally:
            before.update(saved)

    return guarded


def cumulative_preflight(record, baseline, limit):
    assert set(baseline) == {"Swapins", "Swapouts"}
    assert all(type(n) is int and n >= 0 for n in baseline.values())
    assert type(limit) is int and limit >= 0
    guard = {
        "status": "failed",
        "scope": "cumulative_campaign",
        "baseline": baseline.copy(),
        "limit_bytes": limit,
        "deltas": {},
    }
    record["preflight_guard"] = guard
    for key, start in baseline.items():
        current = record["before"]["vm_bytes"][key]
        assert type(current) is int and current >= 0, "invalid VM counter before launch"
        delta = current - start
        guard["deltas"][key] = delta
        assert 0 <= delta <= limit, f"campaign {key} limit exceeded before launch"
    guard["status"] = "passed"


def run_sampler(plan, baseline=None):
    sampler_path = (
        ROOT / "docs/measurements/flash-next-ssd-impl/opt-49-variance-diagnostic/sample-system.txt"
    )
    if baseline is None:
        return runpy.run_path(str(sampler_path))["main"](plan)
    # Isolated namespaces keep historical scripts and the global runpy module intact.
    supervisor_path = (
        ROOT / "docs/measurements/flash-next-ssd-impl/opt-45-audit-prerequisites/run-supervisor.txt"
    )
    monitor_path = (
        ROOT
        / "docs/measurements/flash-next-ssd-impl/opt-43-mtp-route-head/speed-heads/bench-supervisor.txt"
    )
    cumulative_monitor(lambda *_: None, baseline)

    def load(path):
        namespace = runpy.run_path(str(path))
        selected = Path(path).resolve()
        if selected == monitor_path.resolve():
            namespace["monitor"] = cumulative_monitor(namespace["monitor"], baseline)
            namespace["before_launch"] = lambda record: cumulative_preflight(
                record, baseline, namespace["LIMIT"]
            )
        if selected in {sampler_path.resolve(), supervisor_path.resolve()}:
            namespace["main"].__globals__["runpy"] = SimpleNamespace(run_path=load)
        return namespace

    return load(sampler_path)["main"](plan)


def main(plan, baseline=None):
    data = json.loads(plan.read_text())
    result_path = plan.parent / "dataset-result.json"
    assert not result_path.exists()
    result = {
        "status": "incomplete",
        "parity": False,
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
    }
    try:
        assert data and any(item.get("profile_directory") for item in data), (
            "no profile cases selected"
        )
        run_sampler(plan, baseline)
        for item in data:
            if not item.get("profile_directory"):
                continue
            directory = Path(item["profile_directory"])
            golden = json.loads(Path(item["golden"]).read_text())
            oracle = None
            if item.get("natural_oracle"):
                oracle = json.loads(Path(item["natural_oracle"]).read_text())["summaries"][0][
                    "tokens"
                ]
            analysis = analyze(directory, item["profile_kind"], golden, oracle)
            result["phases"] = [
                {k: v for k, v in p.items() if k != "parts"} for p in analysis["phases"]
            ]
            compress_raw(directory)
        result.update(status="complete_validated_diagnostic", parity=True)
    except BaseException as error:
        result.update(status="failed_or_interrupted", parity=False, error=repr(error))
        raise
    finally:
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        with (ROOT / "opti.md").open("a") as stream:
            stream.write(
                f"\n- **53 dataset {plan.parent.name} : {result['status']}**, parité={result['parity']} ; {result_path}.\n"
            )


if __name__ == "__main__":
    if sys.argv[1] == "--control":
        baseline = json.loads(Path(sys.argv[3]).read_text())["original_vm_baseline"]
        run_sampler(Path(sys.argv[2]), baseline)
    else:
        baseline = (
            json.loads(Path(sys.argv[2]).read_text())["original_vm_baseline"]
            if len(sys.argv) == 3
            else None
        )
        main(Path(sys.argv[1]), baseline)
