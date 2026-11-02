"""Validate and summarize Flash53 diagnostic traces; never infer exclusive GPU time."""

import csv
import gzip
import hashlib
import json
import math
import re
import runpy
import subprocess
import sys
import time
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


def readback_sample(event):
    data = event["data"]
    sample = data["sample"]
    fields = ("wait_ns", "copy_ns", "contiguous_build_ns", "bytes", "total_ns")
    assert all(integer(sample[k]) < 1 << 64 for k in fields)
    assert type(sample["ready_before"]) is bool and type(sample["direct"]) is bool
    assert sum(sample[k] for k in fields[:3]) <= sample["total_ns"]
    assert not sample["direct"] or sample["contiguous_build_ns"] == 0
    sizes = {
        "Bool": 1,
        "UInt8": 1,
        "Int8": 1,
        "UInt16": 2,
        "Int16": 2,
        "UInt32": 4,
        "Int32": 4,
        "UInt64": 8,
        "Int64": 8,
        "Float16": 2,
        "BFloat16": 2,
        "Float32": 4,
        "Float64": 8,
        "Complex64": 8,
    }
    assert data["dtype"] in sizes and isinstance(data["shape"], list)
    count = math.prod(integer(n) for n in data["shape"])
    assert count * sizes[data["dtype"]] == sample["bytes"]
    assert sample["total_ns"] <= event["end_ns"] - event["start_ns"]
    return {**sample, "dtype": data["dtype"], "shape": data["shape"]}


def cold_schedule_diagnostic(data, route=None, batch_wall_ns=None):
    fields = ("route_rows", "priority", "cold_occurrences", "multiplicities")
    present = [key in data for key in fields]
    has_ready_indices = "ready_job_indices" in data
    if not any(present) and not has_ready_indices:
        return {"status": "unavailable_legacy", "missing_fields": list(fields)}
    assert all(present), "incomplete Cold schedule diagnostic"
    assert data["status"] == "finished" and data["io_ok"] is True
    assert isinstance(route, dict), "Cold schedule has no routed parent"
    rows = integer(data["route_rows"])
    assert 1 <= rows <= 4 and rows == route["rows"]
    assert type(data["priority"]) is bool and (not data["priority"] or rows == 2)
    ids, slots = route["ids"], route["slots_hit"]
    assert len(ids) == len(slots) == rows * 10
    occurrences = data["cold_occurrences"]
    mapping, multiplicities = data["jobs_mapping"], data["multiplicities"]
    assert all(isinstance(v, list) for v in (occurrences, mapping, multiplicities))
    assert len(mapping) == len(multiplicities) == integer(data["jobs"]) > 2
    assert len(occurrences) == integer(data["cold_rows"]) <= rows * 10
    has_resident_count = "resident_gpu_rows" in route
    if has_resident_count:
        residents = integer(route["resident_gpu_rows"])
        assert residents <= rows * 10
        assert len(occurrences) == rows * 10 - residents, "Cold/Hot parent count mismatch"
    seen, counts, first = set(), defaultdict(int), {}
    for item in occurrences:
        assert isinstance(item, list) and len(item) == 2
        occurrence, slot = map(integer, item)
        assert occurrence < rows * 10 and slot < route["capacity"]
        assert occurrence not in seen, "duplicate Cold occurrence"
        assert slot == slots[occurrence][0], "Cold occurrence uses a different routed slot"
        assert not seen or occurrence > max(seen), "Cold occurrences are not in route order"
        seen.add(occurrence)
        counts[slot] += 1
        first.setdefault(slot, occurrence)
    assert all(hit or i in seen for i, (_, hit) in enumerate(slots)), "missing Cold miss"
    assert len({expert for expert, _ in mapping}) == len(mapping)
    assert len({slot for _, slot in mapping}) == len(mapping)
    assert {slot for _, slot in mapping} == {slot for slot, hit in slots if not hit}
    for (expert, slot), multiplicity in zip(mapping, multiplicities, strict=True):
        assert integer(expert) < 512 and integer(slot) < route["capacity"]
        assert 1 <= integer(multiplicity) == counts[slot] <= rows
        assert all(ids[i] == expert for i, s in occurrences if s == slot)
    expected = sorted(
        mapping,
        key=lambda job: (
            (-counts[job[1]], first[job[1]]) if data["priority"] else (0, first[job[1]])
        ),
    )
    assert mapping == expected, "Cold job launch order does not match policy"
    timings = data["timings_job_worker_start_end_bytes_failed"]
    assert len(timings) == len(mapping), "incomplete Cold job timings"
    if batch_wall_ns is not None:
        integer(batch_wall_ns)
    for index, timing in enumerate(timings):
        assert len(timing) == 6
        job, _, begin, end, count, failed = map(integer, timing)
        assert job == index and begin <= end and count == EXPERT_BYTES and failed == 0
        if batch_wall_ns is not None:
            assert end <= batch_wall_ns, "Cold read timing exceeds batch span"
    completed = sorted(range(len(mapping)), key=lambda job: (timings[job][3], job))
    first_two = completed[:2]
    duplicated = sum(n > 1 for n in multiplicities)
    split = data["split"]
    assert type(split) is bool
    w1, w2 = integer(data["w1_rows"]), integer(data["w2_rows"])
    assert w1 + w2 == len(occurrences)
    ready = integer(data["ready_initial"])
    assert 2 <= ready <= len(mapping)
    if has_ready_indices:
        ready_indices = data["ready_job_indices"]
        assert isinstance(ready_indices, list) and len(ready_indices) == ready
        assert all(integer(job) < len(mapping) for job in ready_indices)
        assert len(set(ready_indices)) == ready, "duplicate ready job index"
    if split:
        assert 0 < w1 < len(occurrences) and ready < len(mapping)
        if has_ready_indices:
            assert w1 == sum(multiplicities[job] for job in ready_indices)
        else:
            ordered = sorted(multiplicities)
            assert sum(ordered[:ready]) <= w1 <= sum(ordered[-ready:])
    else:
        assert w1 == 0 and ready == len(mapping)
    return {
        "status": "validated_schedule_only",
        "route_rows": rows,
        "priority": data["priority"],
        "parent_partition_validation": "resident_count"
        if has_resident_count
        else "unavailable_legacy_parent",
        "readiness_validation": "exact_indices" if has_ready_indices else "bounded_multiplicities",
        "timing_bounds_validation": "batch_wall"
        if batch_wall_ns is not None
        else "unavailable_no_batch_span",
        "split": split,
        "ready_initial": ready,
        "w1_rows": w1,
        "w2_rows": w2,
        "jobs": len(mapping),
        "duplicated_jobs": duplicated,
        "duplicated_job_fraction": duplicated / len(mapping),
        "first_two_completed_jobs": first_two,
        "first_two_completed_slots": [mapping[job][1] for job in first_two],
        "first_two_completed_occurrences": sum(multiplicities[job] for job in first_two),
        "read_tail_after_second_completion_ns": timings[completed[-1]][3]
        - timings[completed[1]][3],
        "limits": "read completions (ties by job index) and logical occurrences; not readiness publication, GPU idle or speedup",
    }


def validate_events(events):
    assert isinstance(events, list) and events, "empty trace"
    children = [0] * len(events)
    paths, layers, ends = [], [], {}
    groups = defaultdict(lambda: {"wall": [], "self": []})
    routes, batches, jobs = [], [], []
    signatures, signature_keys, requests, readbacks = {}, set(), [], []
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
        if name == "expert_kernel_signature" and "id" in data:
            identity = integer(data["id"])
            key = data["key"]
            assert identity == len(signatures) and type(data["stable"]) is bool
            assert isinstance(key, list) and len(key) == 3 and all(type(k) is str for k in key)
            assert key[0] and key[2]
            full = (data["stable"], *key)
            assert full not in signature_keys, "duplicate kernel identity"
            signature_keys.add(full)
            signatures[identity] = {
                "id": identity,
                "name": key[0],
                "stable": data["stable"],
                "key_sha256": hashlib.sha256(json.dumps(full).encode()).hexdigest(),
                "header_bytes": len(key[1].encode()),
                "source_bytes": len(key[2].encode()),
            }
        if name == "expert_kernel_request":
            identity = integer(data["signature"])
            assert identity in signatures, "unknown kernel signature"
            requests.append({"layer": layer, "signature": identity})
        if name == "host_readback":
            readbacks.append({"layer": layer, **readback_sample(event)})
        if name == "moe_forward_rows":
            assert isinstance(data, dict)
            rows, cap = integer(data["rows"]), integer(data["capacity"])
            ids, slots = data["ids"], data["slots_hit"]
            assert 1 <= rows <= 8 and 10 <= cap <= 2544
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
        if name in ("disk_read_batch", "cold_ready_batch"):
            cold = name == "cold_ready_batch"
            assert data["io_ok" if cold else "ok"] is True
            timings = data["timings_job_worker_start_end_bytes_failed"]
            mapping = data["jobs_mapping" if cold else "jobs"]
            if cold:
                assert data["status"] == "finished" and integer(data["jobs"]) == len(mapping)
            workers = integer(data["workers"])
            schedule = data.get("schedule", "static")
            assert schedule in ("static", "dynamic"), "unknown disk job schedule"
            assert 1 <= workers <= 4 and 1 <= len(mapping) == len(timings) <= 512
            assert len({slot for _, slot in mapping}) == len(mapping)
            worker_end = {}
            for index, row in enumerate(timings):
                assert len(row) == 6
                job, worker, start, stop, count, failed = map(integer, row)
                assert job == index and 0 <= worker < min(workers, len(mapping))
                if schedule == "static":
                    assert worker == index % min(workers, len(mapping))
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
                    "operation": name,
                    "jobs": len(mapping),
                    "schedule": schedule,
                    "last_completion_ns": max(t[3] for t in timings),
                    "worker_finish_spread_ns": max(worker_end.values()) - min(worker_end.values()),
                    "wall_ns": end - begin,
                }
            )
            if cold:
                ancestor = parent
                while ancestor is not None and events[ancestor]["name"] != "moe_forward_rows":
                    ancestor = events[ancestor]["parent"]
                batches[-1]["cold_schedule"] = cold_schedule_diagnostic(
                    data, None if ancestor is None else events[ancestor]["data"], end - begin
                )
    for i, event in enumerate(events):
        duration = event["end_ns"] - event["start_ns"]
        assert children[i] <= duration
        key = (layers[i], event["name"])
        groups[key]["wall"].append(duration)
        groups[key]["self"].append(duration - children[i])
    assert routes, "trace has no actual routed experts"
    assert not signatures or requests, "kernel dictionary without requests"
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
        "kernel_signatures": list(signatures.values()),
        "kernel_requests": requests,
        "readbacks": readbacks,
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
    budget = {"quality": 17, "fenced": 33, "natural": 128, "context": 128, "verifier": 64}[kind]
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
    if kind in ("natural", "context"):
        assert (
            summary["token_hash"] == golden["token_hash"]
            and summary["text"] == golden["cache_context"]
        )
    if expected_tokens is not None:
        assert tokens == expected_tokens[:budget]
    assert len(summary["steps"]) == budget - 1
    for i, row in enumerate(summary["steps"], 1):
        assert integer(row["step"]) == i and integer(row["token"]) == tokens[i]
        assert (
            type(row["seconds"]) in (int, float)
            and math.isfinite(row["seconds"])
            and row["seconds"] >= 0
        )
    if "dflash" in summary:
        assert type(summary["dflash"]) is bool
        if summary["dflash"]:
            assert summary["mtp"] is False and summary["full_draft_states_byte_exact"] is True
            counters = summary["dflash_stats"]
            profile = summary["dflash_profile"]
            assert len(counters) == 4 and all(type(n) is int and n >= 0 for n in counters[:3])
            assert type(counters[3]) in (int, float) and math.isfinite(counters[3])
            assert 0 <= counters[3] <= summary["decode_seconds"] + 1e-6
            fields = (
                "proposed",
                "accepted",
                "blocks",
                "verified_rows",
                "drafted_rows",
                "expert_demand_bytes",
            )
            totals = dict.fromkeys(fields, 0)
            for row in summary["steps"]:
                d = row["dflash"]
                assert all(type(d[k]) is int and d[k] >= 0 for k in fields)
                assert 0 <= d["accepted"] <= d["proposed"] <= 7 and d["blocks"] in (0, 1)
                assert d["blocks"] == int(d["proposed"] > 0)
                assert (
                    d["verified_rows"] == d["proposed"] + 1
                    if d["blocks"]
                    else d["verified_rows"] == 0
                )
                assert d["verified_rows"] <= d["drafted_rows"] <= 8
                assert all(
                    type(row[k]) is int and row[k] == 0
                    for k in ("mtp_proposed", "mtp_accepted", "mtp_blocks")
                )
                for key in fields:
                    totals[key] += d[key]
            assert [totals[k] for k in fields[:3]] == counters[:3]
            for key in fields[3:]:
                assert totals[key] == profile[key]
            for key, total in (
                ("verified_widths", "verified_rows"),
                ("drafted_widths", "drafted_rows"),
            ):
                widths = profile[key]
                assert len(widths) == 8 and widths[0] == 0
                assert all(type(n) is int and n >= 0 for n in widths)
                assert sum(widths) == totals["blocks"]
                assert sum((i + 1) * n for i, n in enumerate(widths)) == totals[total]
            times = [
                profile[k]
                for k in ("draft_seconds", "verification_seconds", "commit_cache_seconds")
            ]
            assert all(type(t) in (int, float) and math.isfinite(t) and t >= 0 for t in times)
            assert sum(times) <= counters[3] + 1e-6
        else:
            assert summary["full_draft_states_byte_exact"] is None
            assert summary["dflash_stats"] is None and summary["dflash_profile"] is None
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
        "context": ["warmup", "control_before", "trace_1", "trace_2", "trace_3", "control_after"],
        "fenced": ["warmup", "oracle", "fenced_1", "fenced_2"],
        "quality": ["warmup", "oracle", "trace", "fenced"],
    }[kind]
    assert [s["phase"] for s in complete["summaries"]] == expected_phases
    version = complete.get("profile_schema_version", 1)
    assert type(version) is int and version in (1, 2)
    if version == 2:
        assert type(complete["dflash"]) is bool
        for summary in complete["summaries"]:
            assert summary["dflash"] == (
                complete["dflash"] and summary["phase"] != "sequential_oracle"
            )
            assert all(
                "target_rows" in row and (not summary["dflash"] or "dflash" in row)
                for row in summary["steps"]
            )
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
                if version == 2:
                    assert (
                        stats["readbacks"]
                        and stats["kernel_signatures"]
                        and stats["kernel_requests"]
                    )
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


def memory_headroom(record, current, *, launching=False):
    budget = record.get("load_headroom_bytes")
    model = "MLXL3_FLASH_MODEL" in record.get("env", {}) and record["command"][0] != "/bin/sleep"
    if (
        "load_headroom_bytes" not in record
        and not model
        and "minimum_live_free_bytes" not in record
        and "maximum_compressor_bytes" not in record
    ):
        return
    guard = {"status": "failed", "stage": "before_launch" if launching else "live"}
    record["memory_guard"] = guard
    policy = {
        key: record.get(key)
        for key in ("load_headroom_bytes", "minimum_live_free_bytes", "maximum_compressor_bytes")
    }
    assert all(type(n) is int and n > 0 for n in policy.values()), (
        "model admission requires positive integer load_headroom_bytes, minimum_live_free_bytes "
        "and maximum_compressor_bytes; declare the measured working-set plus reserve in the plan"
    )
    guard["policy"] = policy.copy()
    free = current["Pages free"]
    compressor = current["Pages occupied by compressor"]
    assert all(type(n) is int and n >= 0 for n in (free, compressor)), "invalid memory gauges"
    minimum = budget if launching else policy["minimum_live_free_bytes"]
    guard.update(
        free_bytes=free,
        minimum_free_bytes=minimum,
        compressor_bytes=compressor,
        maximum_compressor_bytes=policy["maximum_compressor_bytes"],
    )
    assert free >= minimum, (
        "memory headroom below admission budget" if launching else "live memory headroom exhausted"
    )
    assert compressor <= policy["maximum_compressor_bytes"], "compressor pressure limit exceeded"
    guard["status"] = "passed"


def cumulative_monitor(monitor, baseline, limit=None):
    assert set(baseline) == {"Swapins", "Swapouts"}
    assert all(type(n) is int and n >= 0 for n in baseline.values())
    assert limit is None or (type(limit) is int and limit >= 0), "invalid cumulative swapin limit"
    baseline = baseline.copy()

    def guarded(child, record, parse_vm):
        before = record["before"]["vm_bytes"]
        saved = {key: before[key] for key in baseline}
        assert all(baseline[key] <= saved[key] for key in baseline), "future VM baseline"
        assert saved["Swapouts"] == baseline["Swapouts"], "campaign Swapouts before monitor"
        record["live_guard_basis"] = {"scope": "cumulative_campaign", **baseline}
        before.update(baseline)

        def zero_swapout(text):
            current = parse_vm(text)
            record["last_live_vm_bytes"] = current.copy()
            if "load_headroom_bytes" in record:
                record.setdefault("live_memory_samples", []).append(
                    {"epoch": time.time(), **current}
                )
            value = current["Swapouts"]
            assert type(value) is int and value >= 0, "invalid live Swapouts counter"
            delta = value - baseline["Swapouts"]
            record["maximum_live_swapout_bytes"] = max(
                record.get("maximum_live_swapout_bytes", 0), delta
            )
            assert delta == 0, "live Swapouts changed"
            if limit is not None:
                incoming = current["Swapins"]
                assert type(incoming) is int and incoming >= 0, "invalid live Swapins counter"
                incoming -= baseline["Swapins"]
                record["maximum_live_swapin_bytes"] = max(
                    record.get("maximum_live_swapin_bytes", 0), incoming
                )
                assert 0 <= incoming <= limit, "cumulative live swapin limit exceeded"
            memory_headroom(record, current)
            return current

        try:
            result = monitor(child, record, zero_swapout)
            if "load_headroom_bytes" in record:
                zero_swapout(subprocess.check_output(["vm_stat"], text=True, timeout=5))
            return result
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
        "swapout_limit_bytes": 0,
        "deltas": {},
    }
    record["preflight_guard"] = guard
    for key, start in baseline.items():
        current = record["before"]["vm_bytes"][key]
        assert type(current) is int and current >= 0, "invalid VM counter before launch"
        delta = current - start
        guard["deltas"][key] = delta
        maximum = 0 if key == "Swapouts" else limit
        assert 0 <= delta <= maximum, f"campaign {key} limit exceeded before launch"
    memory_headroom(record, record["before"]["vm_bytes"], launching=True)
    guard["status"] = "passed"


def run_sampler(plan, baseline=None):
    sampler_path = (
        ROOT / "docs/measurements/flash-next-ssd-impl/opt-49-variance-diagnostic/sample-system.txt"
    )
    if baseline is None:
        items = json.loads(plan.read_text())
        assert not any(
            any(
                key in item
                for key in (
                    "load_headroom_bytes",
                    "minimum_live_free_bytes",
                    "maximum_compressor_bytes",
                )
            )
            or "MLXL3_FLASH_MODEL" in item.get("env", {})
            for item in items
        ), "model memory admission requires an explicit cumulative VM baseline"
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
            namespace["monitor"] = cumulative_monitor(
                namespace["monitor"], baseline, namespace["LIMIT"]
            )
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
