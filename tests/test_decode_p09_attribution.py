"""Exercise the real timeline analyzer with deterministic synthetic XML exports."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "docs/measurements/decode-profile-p09-20261101/analyze.py"


def table(path, rows):
    root = ET.Element("trace-query-result")
    schema = ET.SubElement(root, "schema")
    for key in rows[0]:
        ET.SubElement(ET.SubElement(schema, "col"), "mnemonic").text = key
    for values in rows:
        row = ET.SubElement(root, "row")
        for key, value in values.items():
            item = ET.SubElement(row, key)
            if key in ("process", "thread"):
                ET.SubElement(item, "pid" if key == "process" else "tid").text = str(value)
                if key == "thread":
                    item.set("fmt", "qwen4::profile::flash_detailed")
                    ET.SubElement(ET.SubElement(item, "process"), "pid").text = "7"
            elif key == "stack":
                for name in value:
                    ET.SubElement(item, "frame", name=name)
            else:
                item.text = str(value)
    ET.ElementTree(root).write(path)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "gpu_idle",
        "sampling",
        "shared_window",
        "shared_uncovered",
        "shared_short",
        "bytes",
        "missing_read",
        "mapping",
        "parity",
        "child",
        "cpu_gap",
        "too_short",
    ],
)
def test_real_attribution_partition_cold_and_failure_paths(tmp_path, fault):
    profile = tmp_path / "profile"
    profile.mkdir()
    capture = {
        "status": "native_validated_captures_pending_inspection",
        "parity": fault != "parity",
        "target_reaped": True,
        "target_pid": 7,
    }
    (tmp_path / "capture.json").write_text(json.dumps(capture))
    summary = {
        "full_logits_target_head_states_byte_exact": True,
        "tokens": list(range(128)),
        "logit_count": 248320,
        "decode_epoch": 1,
    }
    (profile / "trace_1-summary.json").write_text(json.dumps(summary))
    events = [{"name": "decode", "parent": None, "start_ns": 0, "end_ns": 500000, "data": None}]
    reads = []
    for step in range(16, 28 if fault == "too_short" else 48):
        a = step * 10000
        parent = len(events)
        events.append(
            {
                "name": "output_step",
                "parent": 0,
                "start_ns": a,
                "end_ns": a + 10000,
                "data": {"step": step},
            }
        )
        events.append(
            {
                "name": "disk_read_batch",
                "parent": parent,
                "start_ns": a + 1000,
                "end_ns": a + 3000,
                "data": {"ok": True, "jobs": [[1, 1]]},
            }
        )
        cold_parent = len(events)
        events.append(
            {
                "name": "cold_ready_batch",
                "parent": parent,
                "start_ns": a + 4000,
                "end_ns": a + 8000,
                "data": {
                    "io_ok": True,
                    "status": "finished",
                    "jobs": 3,
                    "jobs_mapping": [[2, 2], [3, 3], [4, 4]],
                },
            }
        )
        events.append(
            {
                "name": "cold_ready_wait",
                "parent": cold_parent,
                "start_ns": a + 4000,
                "end_ns": a + 7000,
                "data": None,
            }
        )
        for start, end in [(1100, 2900), (4100, 6900), (4500, 7900), (4600, 6000)]:
            reads.append(
                {
                    "process": 7,
                    "call": "BSC_sys_preadv",
                    "start": a + start,
                    "duration": end - start,
                    "errno": 0,
                    "return": 1248012,
                }
            )
    if fault == "bytes":
        reads[5]["return"] -= 1
    elif fault == "missing_read":
        reads.pop(5)
    elif fault == "mapping":
        events[7]["data"]["jobs_mapping"].pop()
    elif fault == "child":
        events[8]["end_ns"] = events[7]["end_ns"] + 1
    (profile / "trace_1-decode.json").write_text(json.dumps(events))
    table(tmp_path / "syscall.xml", reads)
    table(tmp_path / "time-info.xml", [{"trace-start-time": 1}])
    table(
        tmp_path / "metal-gpu-intervals.xml",
        [
            {
                "state": "Active",
                "start": 0,
                "duration": 100000
                if fault == "sampling"
                else 250000
                if fault == "gpu_idle"
                else 500000,
                "process": 7,
            }
        ],
    )
    if fault == "sampling":
        table(
            tmp_path / "time-profile.xml",
            [
                {"process": 7, "thread": 8, "time": 175000, "stack": ["malloc", "mlx::eval_impl"]},
                {"process": 7, "thread": 8, "time": 180000, "stack": []},
                {"process": 7, "thread": 8, "time": 300000, "stack": ["blocked_frame"]},
            ],
        )
        table(
            tmp_path / "time-sample-standard.xml",
            [{"thread": 8, "time": 175000, "sample-type": "Timer"}],
        )
    # Split states test the non-additive CPU/GPU partition independently of scopes.
    table(
        tmp_path / "thread-state.xml",
        [
            {"process": 7, "thread": 8, "start": 0, "duration": 250000, "state": "Running"},
            {
                "process": 7,
                "thread": 8,
                "start": 260000 if fault == "cpu_gap" else 250000,
                "duration": 250000,
                "state": "Blocked",
            },
        ],
    )
    command = [sys.executable, str(SOURCE), str(tmp_path)]
    if fault in ("shared_window", "shared_uncovered", "shared_short"):
        command += {
            "shared_window": ["20", "39"],
            "shared_uncovered": ["16", "45"],
            "shared_short": ["20", "30"],
        }[fault]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    output = tmp_path / "attribution.json"
    if fault not in (None, "gpu_idle", "sampling", "shared_window"):
        assert result.returncode != 0, result.stdout
        assert not output.exists(), "a failed analysis published a successful result"
    else:
        assert result.returncode == 0, result.stderr
        data = json.loads(output.read_text())
        count = 20 if fault == "shared_window" else 31  # Step16 starts before coverage.
        assert data["complete_steps"] == count
        assert data["io"]["jobs"] == 4 * count
        assert data["io"]["bytes"] == 4 * count * 1248012
        assert data["io"]["batch_types"] == {"disk_read_batch": count, "cold_ready_batch": count}
        assert data["clock_fit_halfwidth_ns"] == 100
        budget = data["decode_selected"]
        assert budget["wall_ns"] == count * 10000
        assert budget["model_gpu_active_ns"] == (
            0 if fault == "sampling" else 80000 if fault == "gpu_idle" else count * 10000
        )
        assert budget["cpu_Running_ns"] == (50000 if fault == "shared_window" else 80000)
        assert budget["cpu_Blocked_ns"] == (150000 if fault == "shared_window" else 230000)
        assert budget["cpu_Running_gpu_idle_ns"] == (80000 if fault == "sampling" else 0)
        assert budget["cpu_Blocked_gpu_idle_ns"] == (
            230000 if fault in ("gpu_idle", "sampling") else 0
        )
        assert sum(v["wall_ns"] for v in data["components_exclusive"].values()) == count * 10000
        assert data["io"]["wall_ns"] == count * 5600
        if fault == "sampling":
            sampling = data["cpu_idle_sampling"]
            assert sampling["counts"] == {
                "main_samples_in_window": 3,
                "eligible_cpu_running_gpu_idle_samples": 2,
                "eligible_missing_backtrace": 1,
            }
            assert sampling["mapping"] == {"direct_running": 2, "not_running_or_interrupted": 1}
            assert sampling["nearest_mlx_counts"] == [["mlx::eval_impl", 1], ["<no MLX frame>", 1]]
            assert sampling["raw_kinds"] == {"['Timer']": 1, "no_exact_raw_match": 1}
