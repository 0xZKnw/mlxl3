"""Analyze target-only encoder activity from four xctrace XML exports, never guess kernels."""

import argparse
import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

GPU = "metal-gpu-intervals"
ENCODERS = "metal-application-encoders-list"
SHADERS = "metal-shader-profiler-shader-list"
TIMELINE = "metal-shader-profiler-intervals"
KERNEL_PREFIX = "mlxl3.kernel/"
REQUIRED = {
    GPU: {
        "start": "start-time",
        "duration": "duration",
        "process": "process",
        "event-depth": "metal-nesting-level",
        "event-label": "formatted-label",
        "state": "gpu-state",
        "channel-name": "gpu-channel-name",
        "cmdbuffer-id": "metal-command-buffer-id",
        "encoder-id": "metal-command-buffer-id",
    },
    ENCODERS: {
        "start": "start-time",
        "duration": "duration",
        "process": "process",
        "event-type": "metal-event-name",
        "encoder-label": "metal-object-label",
        "cmdbuffer-id": "metal-command-buffer-id",
        "encoder-id": "metal-command-buffer-id",
    },
    SHADERS: {"process": "process", "name": "metal-object-label"},
    TIMELINE: {"start": "start-time", "duration": "duration", "process": "process"},
}


def interval_union_ns(intervals: list[tuple[int, int]]) -> int:
    """Union length of half-open integer intervals, including negative relative starts.

    pre: all(a <= b for a, b in intervals)
    post: _ >= 0
    post: _ <= sum(b - a for a, b in intervals)
    """
    total = 0
    end = None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("reversed interval")
        uncovered = start if end is None else max(start, end)
        total += max(0, stop - uncovered)
        end = stop if end is None else max(end, stop)
    return total


def _integer(value: str, field: str, *, signed=False) -> int:
    if not re.fullmatch(r"-?[0-9]+" if signed else r"[0-9]+", value):
        raise ValueError(f"invalid integer field: {field}")
    return int(value)


def _table(path: Path, name: str) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"missing input: {path.name}")
    root = ET.parse(path).getroot()
    schemas = root.findall(".//schema")
    if root.tag != "trace-query-result" or len(schemas) != 1:
        raise ValueError(f"expected one trace schema: {name}")
    schema = schemas[0]
    columns = [c.findtext("mnemonic") for c in schema.findall("col")]
    types = [c.findtext("engineering-type") for c in schema.findall("col")]
    if schema.get("name") != name or None in columns or len(set(columns)) != len(columns):
        raise ValueError(f"wrong or ambiguous schema: {name}")
    mapping = dict(zip(columns, types))
    if any(mapping.get(key) != typ for key, typ in REQUIRED[name].items()):
        raise ValueError(f"missing or incompatible columns: {name}")
    definitions = {}
    for element in root.iter():
        if "id" in element.attrib:
            key = element.attrib["id"]
            if key in definitions:
                raise ValueError(f"duplicate XML id in {name}")
            definitions[key] = element

    def resolve(element):
        seen = set()
        original_tag = element.tag
        while "ref" in element.attrib:
            key = element.attrib["ref"]
            if key not in definitions or key in seen:
                raise ValueError(f"missing or cyclic XML ref in {name}")
            if len(element) or (element.text or "").strip():
                raise ValueError(f"reference with conflicting content in {name}")
            seen.add(key)
            element = definitions[key]
        if element.tag != original_tag:
            raise ValueError(f"reference type mismatch in {name}")
        return element

    def value(element):
        element = resolve(element)
        return (element.text or "").strip() or element.get("fmt", "")

    # References can be declared in a formatted label or thread, not only table cells.
    for element in root.iter():
        if "ref" in element.attrib:
            resolve(element)
    rows = []
    for row in root.findall(".//row"):
        if len(row) != len(columns):
            raise ValueError(f"row width mismatch: {name}")
        cells = dict(zip(columns, row))
        for key, typ in REQUIRED[name].items():
            if cells[key].tag not in (typ, "sentinel"):
                raise ValueError(f"cell type mismatch: {name}/{key}")
        data = {key: value(cell) for key, cell in cells.items()}
        process = resolve(cells["process"])
        pid = process.find("pid")
        data["pid"] = _integer(value(pid), "pid") if pid is not None else None
        rows.append(data)
    return rows


def _interval(row: dict) -> tuple[int, int]:
    start = _integer(row["start"], "start", signed=True)
    duration = _integer(row["duration"], "duration")
    return start, start + duration


def _key(row: dict) -> tuple[int, int]:
    return (
        _integer(row["cmdbuffer-id"], "cmdbuffer-id"),
        _integer(row["encoder-id"], "encoder-id"),
    )


def _kernel(label: str | None) -> str | None:
    if label is None or not label.startswith(KERNEL_PREFIX):
        return None
    name = label[len(KERNEL_PREFIX) :]
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name) else None


def _kernel_sequence(label: str | None) -> tuple[str, ...] | None:
    """Keep exported physical-encoder aggregation intact, including repetition.

    post: _ is None or len(_) > 0
    post: _ is None or label == " & ".join(KERNEL_PREFIX + name for name in _)
    """
    if label is None:
        return None
    names = tuple(_kernel(part) for part in label.split(" & "))
    return names if all(name is not None for name in names) else None


def analyze(directory: Path, pid: int) -> dict:
    if isinstance(pid, bool) or pid <= 0:
        raise ValueError("pid must be positive")
    tables = {name: _table(directory / f"{name}.xml", name) for name in REQUIRED}
    selected = {name: [r for r in rows if r["pid"] == pid] for name, rows in tables.items()}
    gpu = selected[GPU]
    encoders = selected[ENCODERS]
    if not gpu or not encoders:
        raise ValueError("empty requested PID GPU/encoder coverage")
    encodings = {}
    duplicate_encodings = 0
    for row in encoders:
        key = _key(row)
        record = (row["encoder-label"], _interval(row), row["event-type"])
        if key in encodings:
            if encodings[key] != record:
                raise ValueError("conflicting encoder identity")
            duplicate_encodings += 1
        encodings[key] = record
    intervals = defaultdict(list)
    depths = Counter()
    states = Counter()
    channels = Counter()
    matched_rows = 0
    for row in gpu:
        interval = _interval(row)
        key = _key(row)
        depth = _integer(row["event-depth"], "event-depth")
        states[row["state"]] += 1
        if row["state"] != "Active":
            continue
        depths[depth] += 1
        channels[row["channel-name"]] += 1
        matched_rows += key in encodings
        intervals[key].append(interval)
    all_intervals = [x for values in intervals.values() for x in values]
    if not all_intervals or not any(b > a for a, b in all_intervals):
        raise ValueError("no positive active GPU intervals for requested PID")
    raw = sum(b - a for a, b in all_intervals)
    per_encoder = sum(interval_union_ns(values) for values in intervals.values())
    union = interval_union_ns(all_intervals)
    labels = defaultdict(list)
    sequences = defaultdict(list)
    unknown = []
    named_encoders = Counter()
    sequence_encoders = Counter()
    for key, values in intervals.items():
        sequence = _kernel_sequence(encodings[key][0]) if key in encodings else None
        if sequence is None:
            unknown.extend(values)
        elif len(sequence) == 1:
            name = sequence[0]
            labels[name].extend(values)
            named_encoders[name] += 1
        else:
            sequences[sequence].extend(values)
            sequence_encoders[sequence] += 1
    groups = []
    for name, values in labels.items():
        groups.append(
            {
                "declared_single_dispatch_encoder_kernel": name,
                "encoders": named_encoders[name],
                "gpu_intervals": len(values),
                "raw_duration_ns": sum(b - a for a, b in values),
                "interval_union_ns": interval_union_ns(values),
                "per_encoder_union_sum_ns": sum(
                    interval_union_ns(v)
                    for k, v in intervals.items()
                    if k in encodings and _kernel(encodings[k][0]) == name
                ),
            }
        )
    aggregates = []
    for sequence, values in sequences.items():
        aggregates.append(
            {
                "declared_kernel_sequence": list(sequence),
                "encoders": sequence_encoders[sequence],
                "gpu_intervals": len(values),
                "raw_duration_ns": sum(b - a for a, b in values),
                "interval_union_ns": interval_union_ns(values),
                "per_encoder_union_sum_ns": sum(
                    interval_union_ns(v)
                    for k, v in intervals.items()
                    if k in encodings and _kernel_sequence(encodings[k][0]) == sequence
                ),
            }
        )
    sources = {
        name + ".xml": hashlib.sha256((directory / (name + ".xml")).read_bytes()).hexdigest()
        for name in REQUIRED
    }
    return {
        "status": "analyzed_encoder_activity_not_speed_validation",
        "time_unit": "ns; raw xctrace start-time/duration values",
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": sources,
        "target_PID_filtered": True,
        "GPU": {
            "total_rows": len(tables[GPU]),
            "target_rows": len(gpu),
            "active_rows": len(all_intervals),
            "states": dict(states),
            "depths": dict(depths),
            "channels": dict(channels),
            "raw_duration_ns": raw,
            "per_encoder_union_sum_ns": per_encoder,
            "global_interval_union_ns": union,
            "within_encoder_overlap_ns": raw - per_encoder,
            "cross_encoder_overlap_ns": per_encoder - union,
            "observed_span_ns": max(b for _, b in all_intervals) - min(a for a, _ in all_intervals),
        },
        "join": {
            "CPU_encoder_rows": len(encoders),
            "CPU_duplicate_rows": duplicate_encodings,
            "CPU_unique_encoders": len(encodings),
            "GPU_unique_encoders": len(intervals),
            "matched_GPU_rows": matched_rows,
            "unmapped_GPU_encoders": len(intervals.keys() - encodings.keys()),
            "CPU_encoders_without_active_GPU": len(encodings.keys() - intervals.keys()),
        },
        "CPU_encoding": {
            "event_types": dict(Counter(r["event-type"] for r in encoders)),
            "raw_duration_ns": sum(
                b - a for _, (a, b), kind in encodings.values() if kind == "Encoding"
            ),
            "scope": "CPU Encoding duration, not GPU kernel time",
        },
        "shader_metadata": {
            "target_rows": len(selected[SHADERS]),
            "scope": "shader/PSO catalog only; no invocation or timing inferred",
        },
        "shader_timeline": {"target_rows": len(selected[TIMELINE]), "timing_analyzed": False},
        "named_encoder_groups": sorted(groups, key=lambda g: g["interval_union_ns"], reverse=True),
        "aggregated_encoder_groups": sorted(
            aggregates, key=lambda g: g["interval_union_ns"], reverse=True
        ),
        "unknown_or_unmapped": {
            "GPU_intervals": len(unknown),
            "interval_union_ns": interval_union_ns(unknown),
        },
        "limits": [
            "Named groups require the explicit diagnostic encoder-label contract, not shader-catalog guesses.",
            "Aggregated encoder sequences preserve whole-group duration; no constituent kernel timing is assigned.",
            "Encoder activity can contain nested/overlapping events; group sums are not exclusive GPU percentages.",
            "No correctness, clean measurement conditions, full-generation coverage or release speed gain is established.",
            "Zero ShaderTimeline rows do not mean zero kernel work or per-kernel timing.",
        ],
    }


def publish(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".trace-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        os.link(temporary, path)  # Exclusive, atomic publication; existing evidence is preserved.
    finally:
        os.unlink(temporary)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = analyze(args.directory, args.pid)
    except (OSError, ValueError, ET.ParseError) as error:
        report = {"status": "failed", "error": str(error), "kernel_attribution": "not_validated"}
    publish(args.output, report)
    return 0 if report["status"].startswith("analyzed_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
