"""Synthetic xctrace boundary tests; no real private trace/process data is stored."""

import importlib.util
import itertools
import json
import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_native_metal_trace.py"
SPEC = importlib.util.spec_from_file_location("native_trace", SCRIPT)
trace = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trace
SPEC.loader.exec_module(trace)


def write_table(directory, name, records):
    root = ET.Element("trace-query-result")
    schema = ET.SubElement(root, "schema", name=name)
    for key, typ in trace.REQUIRED[name].items():
        col = ET.SubElement(schema, "col")
        ET.SubElement(col, "mnemonic").text = key
        ET.SubElement(col, "engineering-type").text = typ
    for record in records:
        row = ET.SubElement(root, "row")
        for key, typ in trace.REQUIRED[name].items():
            element = ET.SubElement(row, typ)
            if key == "process":
                # The formatted name deliberately disagrees with actual process.pid.
                element.set("fmt", "other-process (999999)")
                ET.SubElement(element, "pid").text = str(record.get("pid", 111))
            else:
                element.text = str(record.get(key, ""))
    path = directory / f"{name}.xml"
    ET.ElementTree(root).write(path)
    return path


def fixture(directory):
    gpu = []
    for start, duration, cb, encoder, depth in [
        (0, 10, 10, 11, 0),
        (2, 4, 10, 11, 1),
        (0, 10, 10, 11, 0),
        (5, 10, 12, 13, 0),
        (20, 5, 14, 15, 0),
    ]:
        gpu.append(
            {
                "start": start,
                "duration": duration,
                "cmdbuffer-id": cb,
                "encoder-id": encoder,
                "event-depth": depth,
                "state": "Active",
                "channel-name": "Compute",
                "event-label": "secret generic event",
            }
        )
    gpu.append(dict(gpu[0], pid=222, duration=1000))
    encoders = [
        {
            "start": 0,
            "duration": 3,
            "cmdbuffer-id": 10,
            "encoder-id": 11,
            "event-type": "Encoding",
            "encoder-label": "mlxl3.kernel/hc_mix_f32",
        },
        {
            "start": 2,
            "duration": 4,
            "cmdbuffer-id": 12,
            "encoder-id": 13,
            "event-type": "Encoding",
            "encoder-label": "Compute Command 0",
        },
    ]
    encoders.append(dict(encoders[0], pid=222, **{"encoder-label": "secret-other-kernel"}))
    write_table(directory, trace.GPU, gpu)
    write_table(directory, trace.ENCODERS, encoders)
    write_table(
        directory,
        trace.SHADERS,
        [
            {"name": "registered_but_not_running"},
            {"pid": 222, "name": "secret-other-shader"},
        ],
    )
    write_table(directory, trace.TIMELINE, [])
    return directory


def test_explicit_join_unions_pid_and_no_catalog_inference(tmp_path):
    result = trace.analyze(fixture(tmp_path), 111)
    assert result["GPU"] == {
        "total_rows": 6,
        "target_rows": 5,
        "active_rows": 5,
        "states": {"Active": 5},
        "depths": {0: 4, 1: 1},
        "channels": {"Compute": 5},
        "raw_duration_ns": 39,
        "per_encoder_union_sum_ns": 25,
        "global_interval_union_ns": 20,
        "within_encoder_overlap_ns": 14,
        "cross_encoder_overlap_ns": 5,
        "observed_span_ns": 25,
    }
    assert result["join"]["matched_GPU_rows"] == 4
    assert result["join"]["unmapped_GPU_encoders"] == 1
    assert result["CPU_encoding"]["raw_duration_ns"] == 7
    assert result["shader_metadata"]["target_rows"] == 1
    assert result["shader_timeline"] == {"target_rows": 0, "timing_analyzed": False}
    assert result["named_encoder_groups"] == [
        {
            "declared_single_dispatch_encoder_kernel": "hc_mix_f32",
            "encoders": 1,
            "gpu_intervals": 3,
            "raw_duration_ns": 24,
            "interval_union_ns": 10,
            "per_encoder_union_sum_ns": 10,
        }
    ]
    assert result["unknown_or_unmapped"] == {"GPU_intervals": 2, "interval_union_ns": 15}
    serialized = json.dumps(result)
    for private in (
        "secret",
        "other-process",
        "999999",
        "registered_but_not_running",
        str(tmp_path),
    ):
        assert private not in serialized


def test_generic_event_or_catalog_never_fabricates_kernel(tmp_path):
    fixture(tmp_path)
    path = tmp_path / f"{trace.ENCODERS}.xml"
    tree = ET.parse(path)
    # Actual cell tag is the engineering type, not column mnemonic.
    for label in tree.findall(".//metal-object-label"):
        label.text = "hc_mix_f32"
    tree.write(path)
    result = trace.analyze(tmp_path, 111)
    assert result["named_encoder_groups"] == []
    assert result["unknown_or_unmapped"]["GPU_intervals"] == 5


@pytest.mark.parametrize(
    "label",
    [
        None,
        "",
        "hc",
        "mlxl3.kernel/",
        "mlxl3.kernel/a/b",
        "mlxl3.kernel/a b",
        "mlxl3.kernel/é",
        "mlxl3.kernel/1x",
    ],
)
def test_unknown_labels_remain_unknown(label):
    assert trace._kernel(label) is None


def test_union_independent_integer_coverage_exhaustive_and_fuzz():
    intervals = [(a, b) for a in range(-3, 4) for b in range(a, 4)]
    rng = random.Random(7503)
    cases = list(itertools.product(intervals, repeat=2))
    cases += [
        [
            tuple(sorted((rng.randrange(-10, 11), rng.randrange(-10, 11))))
            for _ in range(rng.randrange(8))
        ]
        for _ in range(500)
    ]
    for values in cases:
        covered = set().union(*(set(range(a, b)) for a, b in values))
        assert trace.interval_union_ns(list(values)) == len(covered)
    assert trace.interval_union_ns([]) == 0
    with pytest.raises(ValueError, match="reversed"):
        trace.interval_union_ns([(2, 1)])


def crosshair_two_interval_union(a: int, b: int, c: int, d: int) -> int:
    """Actual helper versus independent two-interval inclusion/exclusion.

    pre: a <= b and c <= d
    post: _ == (b - a) + (d - c) - max(0, min(b, d) - max(a, c))
    """
    return trace.interval_union_ns([(a, b), (c, d)])


def test_nested_local_refs_not_formatted_pid(tmp_path):
    fixture(tmp_path)
    path = tmp_path / f"{trace.GPU}.xml"
    tree = ET.parse(path)
    rows = tree.findall(".//row")
    process = rows[0].find("process")
    process.set("id", "p1")
    # Real traces can define a process inside a formatted-label before its cell ref.
    rows[0].remove(process)
    rows[0].find("formatted-label").append(process)
    rows[0].insert(2, ET.Element("process", ref="p1"))
    rows[1].remove(rows[1].find("process"))
    rows[1].insert(2, ET.Element("process", ref="p1"))
    tree.write(path)
    assert trace.analyze(tmp_path, 111)["GPU"]["target_rows"] == 5


@pytest.mark.parametrize(
    "fault",
    [
        "missing-ref",
        "cycle",
        "type-ref",
        "duplicate-id",
        "ref-content",
        "schema",
        "column",
        "cell",
        "width",
        "negative-duration",
        "bad-integer",
        "conflicting-encoder",
    ],
)
def test_reject_malformed_boundaries(tmp_path, fault):
    fixture(tmp_path)
    name = trace.ENCODERS if fault == "conflicting-encoder" else trace.GPU
    path = tmp_path / f"{name}.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    rows = root.findall(".//row")
    process = rows[0].find("process")
    if fault == "missing-ref":
        process.clear()
        process.set("ref", "absent")
    elif fault == "cycle":
        process.clear()
        process.set("id", "cycle")
        process.set("ref", "cycle")
    elif fault == "type-ref":
        rows[0][0].set("id", "start")
        process.clear()
        process.set("ref", "start")
    elif fault == "duplicate-id":
        rows[0][0].set("id", "same")
        rows[0][1].set("id", "same")
    elif fault == "ref-content":
        process.set("id", "p")
        rows[1].find("process").set("ref", "p")
    elif fault == "schema":
        root.find("schema").set("name", "unexpected")
    elif fault == "column":
        root.find(".//engineering-type").text = "unexpected"
    elif fault == "cell":
        rows[0][0].tag = "unexpected"
    elif fault == "width":
        rows[0].remove(rows[0][0])
    elif fault == "negative-duration":
        rows[0].find("duration").text = "-1"
    elif fault == "bad-integer":
        rows[0].find("metal-command-buffer-id").text = "0x10"
    else:
        root.append(ET.fromstring(ET.tostring(rows[0])))
        root[-1].find("metal-object-label").text = "mlxl3.kernel/different"
    tree.write(path)
    with pytest.raises(ValueError):
        trace.analyze(tmp_path, 111)


@pytest.mark.parametrize("pid", [0, -1, True, 999])
def test_empty_or_invalid_target_is_not_validation(tmp_path, pid):
    fixture(tmp_path)
    with pytest.raises(ValueError):
        trace.analyze(tmp_path, pid)


def test_no_positive_active_work_fails(tmp_path):
    fixture(tmp_path)
    path = tmp_path / f"{trace.GPU}.xml"
    tree = ET.parse(path)
    for duration in tree.findall(".//duration"):
        duration.text = "0"
    tree.write(path)
    with pytest.raises(ValueError, match="positive active"):
        trace.analyze(tmp_path, 111)


def test_cli_success_failure_exclusive_atomic_publication(tmp_path):
    fixture(tmp_path)
    output = tmp_path / "report.json"
    args = [sys.executable, str(SCRIPT), str(tmp_path), "--pid", "111", "--output", str(output)]
    result = subprocess.run(args, capture_output=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    first = output.read_bytes()
    assert json.loads(first)["GPU"]["target_rows"] == 5
    result = subprocess.run(args, capture_output=True, timeout=10, check=False)
    assert result.returncode != 0
    assert output.read_bytes() == first
    (tmp_path / f"{trace.GPU}.xml").write_text("<broken>")
    args[-1] = str(tmp_path / "failed.json")
    result = subprocess.run(args, capture_output=True, timeout=10, check=False)
    assert result.returncode == 1
    report = json.loads((tmp_path / "failed.json").read_text())
    assert report["status"] == "failed"
    assert report["kernel_attribution"] == "not_validated"
    assert not list(tmp_path.glob(".trace-*"))


def test_missing_input_fails(tmp_path):
    with pytest.raises(ValueError, match="missing input"):
        trace.analyze(tmp_path, 111)
