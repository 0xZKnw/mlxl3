"""Summarize target-only xctrace tables; CPU samples and wall intervals stay distinct."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from analyze_native_metal_trace import interval_union_ns

TYPES = {
    "time": "sample-time",
    "start": "start-time",
    "duration": "duration",
    "thread": "thread",
    "process": "process",
    "thread-state": "thread-state",
    "weight": "weight",
    "stack": "tagged-backtrace",
    "backtrace": "tagged-backtrace",
    "call": "syscall",
    "cputime": "duration-on-core",
    "waittime": "duration-waiting",
    "arg1": "syscall-arg",
    "arg4": "syscall-arg",
    "return": "syscall-return",
    "errno": "syscall-return",
    "syscall": "syscall",
    "type": "string",
    "bytes": "size-in-bytes",
    "state": "thread-state",
    "thermal-throttled": "boolean",
    "timestamp": "start-time",
    "resource-size": "size-in-bytes",
    "resource-type": "metal-object-label",
    "event-type": "gpu-memory-event-name",
    "current-allocated-size": "size-in-bytes",
}


def number(cell, *, signed=False):
    if cell is None or cell["tag"] == "sentinel":
        return None
    text = cell["text"]
    if not re.fullmatch(r"-?[0-9]+" if signed else r"[0-9]+", text):
        raise ValueError("invalid numeric trace cell: " + text)
    return int(text)


def table_rows(path: Path, schema_name: str, wanted: set[str]):
    """Stream a table, caching only referenced values and required cell types.

    xctrace defines values before referencing them. Forward/missing references,
    malformed/truncated XML and incompatible schemas fail instead of guessing.
    """
    refs = set()
    tail = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            part = tail + chunk
            if b"<!DOCTYPE" in part:
                raise ValueError("DOCTYPE not supported")
            refs.update(re.findall(rb'\bref="([0-9]+)"', part))
            tail = part[-64:]
    refs = {v.decode() for v in refs}
    cache, columns, needed = {}, None, set()
    schema_count = 0
    stack = []

    def cell(element):
        if "ref" in element.attrib:
            value = cache.get(element.attrib["ref"])
            if value is None or value["tag"] != element.tag:
                raise ValueError("missing/forward or mismatched XML reference")
            return value
        value = {
            "tag": element.tag,
            "text": (element.text or "").strip(),
            "fmt": element.get("fmt", ""),
            "pid": None,
            "tid": None,
            "frames": (),
        }
        if element.tag == "process":
            child = element.find("pid")
            if child is not None:
                value["pid"] = number(cell(child))
        elif element.tag == "thread":
            child = element.find("tid")
            if child is not None:
                value["tid"] = number(cell(child))
            child = element.find("process")
            if child is not None:
                value["pid"] = cell(child)["pid"]
        elif element.tag == "frame":
            value["frames"] = (element.get("name", "<unknown frame>"),)
        elif element.tag == "tagged-backtrace":
            value["frames"] = tuple(name for child in element for name in cell(child)["frames"])
        return value

    def remember(element):
        # Children precede their parent, while sibling definitions retain XML order.
        for node in element:
            remember(node)
        key = element.get("id")
        if key in refs and element.tag in needed:
            if key in cache:
                raise ValueError("duplicate referenced XML id")
            cache[key] = cell(element)

    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            stack.append(element)
            if len(stack) == 1 and element.tag != "trace-query-result":
                raise ValueError("wrong XML root")
            continue
        if element.tag == "schema":
            schema_count += 1
            if schema_count != 1 or element.get("name") != schema_name:
                raise ValueError("wrong or multiple trace schemas")
            columns = [col.findtext("mnemonic") for col in element.findall("col")]
            types = [col.findtext("engineering-type") for col in element.findall("col")]
            if None in columns or None in types or len(set(columns)) != len(columns):
                raise ValueError("invalid trace columns")
            if not wanted <= set(columns):
                raise ValueError("missing required columns")
            mapping = dict(zip(columns, types))
            if any(mapping[name] != TYPES.get(name) for name in wanted):
                raise ValueError("incompatible trace column units/types")
            needed = {typ for col, typ in zip(columns, types) if col in wanted}
            needed |= {"process", "pid", "thread", "tid"}
            if "tagged-backtrace" in needed:
                needed.add("frame")
        elif element.tag == "row":
            if columns is None or len(element) != len(columns):
                raise ValueError("row width/schema mismatch")
            remember(element)
            row = {}
            for name, typ, node in zip(columns, types, element):
                if name in wanted:
                    if node.tag not in (typ, "sentinel"):
                        raise ValueError("trace cell does not match schema")
                    row[name] = cell(node)
            yield row
            stack[-2].remove(element)
            element.clear()
        stack.pop()
    if schema_count != 1:
        raise ValueError("missing trace schema")


def distribution(values):
    if not values:
        return {"count": 0}
    values = sorted(values)
    return {
        "count": len(values),
        "total_ns": sum(values),
        "min_ns": values[0],
        "p50_ns": values[(len(values) - 1) // 2],
        "p95_ns": values[(len(values) * 95 - 1) // 100],
        "max_ns": values[-1],
    }


def summarize_cpu(directory: Path, pid: int, output: Path):
    if type(pid) is not int or pid <= 0:
        raise ValueError("positive PID required")
    report = {
        "pid": pid,
        "status": "incomplete",
        "limits": [
            "Weights are Instruments output, not CPU busy durations; context-switch weights may extend before the retained sample window.",
            "Inclusive stacks overlap; their weights must not be added.",
            "Thread-state Unknown before the retained window is excluded.",
            "Read return bytes are logical file traffic, not physical SSD/NAND bytes.",
            "Output chunk times are not guaranteed to be individual model token times.",
        ],
    }
    states, leaf, inclusive, samples_by_thread = Counter(), Counter(), Counter(), Counter()
    sample_times = []
    n = 0
    missing_running_weight = 0
    running_stacks = 0
    for row in table_rows(
        directory / "time-profile.xml",
        "time-profile",
        {"time", "thread", "process", "thread-state", "weight", "stack"},
    ):
        if row["process"]["pid"] != pid:
            continue
        t, w = number(row["time"], signed=True), number(row["weight"])
        if t is None or w is None or w <= 0:
            raise ValueError("invalid CPU sample")
        n += 1
        state = row["thread-state"]["text"]
        states[state] += w
        sample_times.append(t)
        if state == "Running":
            frames = row["stack"]["frames"]
            if not frames:
                if row["stack"]["tag"] != "sentinel":
                    raise ValueError("empty running CPU stack")
                missing_running_weight += w
                frames = ("<unavailable backtrace>",)
            else:
                running_stacks += 1
            leaf[frames[0]] += w
            inclusive.update({frame: w for frame in set(frames)})
            samples_by_thread[row["thread"]["tid"]] += w
    if n == 0:
        raise ValueError("empty target CPU sample coverage")
    if running_stacks == 0:
        raise ValueError("empty running CPU stack coverage")
    report["samples"] = {
        "rows": n,
        "start_ns": min(sample_times),
        "end_ns": max(sample_times),
        "weight_ns_by_state": dict(states),
        "running_weight_ns_by_thread": dict(samples_by_thread),
        "unattributed_running_weight_ns": missing_running_weight,
        "leaf_running_weight_ns": leaf.most_common(70),
        "inclusive_running_weight_ns": inclusive.most_common(100),
    }
    calls, durations, cpu, wait, errors, returns = (
        Counter(),
        defaultdict(list),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    call_threads = defaultdict(Counter)
    call_thread_durations = defaultdict(Counter)
    cpu_cells, wait_cells = Counter(), Counter()
    writes = []
    output.mkdir(parents=True, exist_ok=True)
    with gzip.open(output / "reads.csv.gz", "wt", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["start_ns", "duration_ns", "call", "tid", "fd", "arg4", "returned_bytes", "errno"]
        )
        for row in table_rows(
            directory / "syscall.xml",
            "syscall",
            {
                "start",
                "duration",
                "process",
                "thread",
                "call",
                "cputime",
                "waittime",
                "arg1",
                "arg4",
                "return",
                "errno",
            },
        ):
            if row["process"]["pid"] != pid:
                continue
            name = row["call"]["text"]
            start, duration = number(row["start"], signed=True), number(row["duration"])
            if start is None or duration is None:
                raise ValueError("missing syscall interval")
            calls[name] += 1
            durations[name].append(duration)
            c, w = number(row["cputime"]), number(row["waittime"])
            cpu[name] += c or 0
            wait[name] += w or 0
            cpu_cells[name] += int(c is not None)
            wait_cells[name] += int(w is not None)
            err = number(row["errno"], signed=True)
            errors[name] += int(err not in (None, 0))
            result = number(row["return"], signed=True)
            fd = number(row["arg1"], signed=True)
            call_threads[name][row["thread"]["tid"]] += 1
            call_thread_durations[name][row["thread"]["tid"]] += duration
            if name.startswith("BSC_") and any(part in name for part in ("pread", "readv")):
                if result is not None and result >= 0:
                    returns[name] += result
                writer.writerow(
                    [
                        start,
                        duration,
                        name,
                        row["thread"]["tid"],
                        fd,
                        number(row["arg4"], signed=True),
                        result,
                        err,
                    ]
                )
            if name.startswith("BSC_write") and fd == 1 and result is not None and result > 0:
                writes.append({"start_ns": start, "end_ns": start + duration, "bytes": result})
    if not calls:
        raise ValueError("empty target syscall coverage")
    report["syscalls"] = [
        {
            "name": name,
            **distribution(durations[name]),
            "cpu_ns_known": cpu[name] if cpu_cells[name] else None,
            "wait_ns_known": wait[name] if wait_cells[name] else None,
            "cpu_duration_cells": cpu_cells[name],
            "wait_duration_cells": wait_cells[name],
            "nonzero_errno": errors[name],
            "read_return_bytes": returns[name],
            "threads": dict(call_threads[name]),
            "duration_ns_by_thread": dict(call_thread_durations[name]),
        }
        for name, _ in calls.most_common()
    ]
    report["stdout_writes"] = writes
    fs = Counter()
    fs_bytes = Counter()
    fs_errors = Counter()
    fs_byte_cells = Counter()
    fs_times = []
    for row in table_rows(
        directory / "FsSyscall.xml",
        "FsSyscall",
        {"start", "duration", "process", "syscall", "type", "return", "bytes", "errno"},
    ):
        if row["process"]["pid"] != pid:
            continue
        name = row["syscall"]["text"]
        fs[name] += 1
        size = number(row["bytes"])
        fs_bytes[name] += size or 0
        fs_byte_cells[name] += int(size is not None)
        fs_errors[name] += int(number(row["errno"], signed=True) not in (None, 0))
        start, duration = number(row["start"], signed=True), number(row["duration"])
        if start is not None and duration is not None:
            fs_times.append((start, start + duration))
    report["filesystem"] = {
        "rows": sum(fs.values()),
        "calls": dict(fs),
        "reported_bytes": {name: fs_bytes[name] if fs_byte_cells[name] else None for name in fs},
        "reported_byte_cells": dict(fs_byte_cells),
        "nonzero_errno": dict(fs_errors),
        "interval_union_ns": interval_union_ns(fs_times),
    }
    thread_states, oncore, waiting = Counter(), Counter(), Counter()
    thermal = Counter()
    known = []
    state_times = defaultdict(Counter)
    thread_intervals = defaultdict(list)
    oncore_cells, waiting_cells = Counter(), Counter()
    for row in table_rows(
        directory / "thread-state.xml",
        "thread-state",
        {
            "start",
            "duration",
            "process",
            "thread",
            "state",
            "cputime",
            "waittime",
            "thermal-throttled",
        },
    ):
        if row["process"]["pid"] != pid:
            continue
        state = row["state"]["text"]
        if state == "Unknown":
            continue
        t, d = number(row["start"], signed=True), number(row["duration"])
        if t is None or d is None:
            raise ValueError("missing known thread interval")
        thread_states[state] += d
        tid = row["thread"]["tid"]
        state_times[tid][state] += d
        thread_intervals[tid].append((t, t + d))
        c, w = number(row["cputime"]), number(row["waittime"])
        oncore[tid] += c or 0
        waiting[tid] += w or 0
        oncore_cells[tid] += int(c is not None)
        waiting_cells[tid] += int(w is not None)
        thermal[row["thermal-throttled"]["text"] or "unavailable"] += 1
        known.append((t, t + d))
    report["scheduler"] = {
        "known_state_thread_ns": dict(thread_states),
        "oncore_ns_known_by_thread": {
            tid: oncore[tid] if oncore_cells[tid] else None for tid in state_times
        },
        "wait_ns_known_by_thread": {
            tid: waiting[tid] if waiting_cells[tid] else None for tid in state_times
        },
        "oncore_duration_cells": dict(oncore_cells),
        "wait_duration_cells": dict(waiting_cells),
        "state_duration_ns_by_thread": {tid: dict(v) for tid, v in state_times.items()},
        "thread_interval_union_ns": {
            tid: interval_union_ns(v) for tid, v in thread_intervals.items()
        },
        "thread_interval_overlap_ns": {
            tid: sum(b - a for a, b in v) - interval_union_ns(v)
            for tid, v in thread_intervals.items()
        },
        "thermal_throttled_cell_counts": dict(thermal),
        "known_interval_union_ns": interval_union_ns(known),
    }
    report["status"] = "validated_target_tables_partial_window"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {"status": "incomplete", "error": None}
    try:
        result = summarize_cpu(args.directory, args.pid, args.out)
        (args.out / "cpu-summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        receipt.update(status="validated_target_tables_partial_window")
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        (args.out / "analysis-status.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
