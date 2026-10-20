"""The dataset cannot turn incomplete, divergent or overlapping records into timings."""

import gzip
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))


def event(name, parent, begin, end, data=None):
    return {"name": name, "parent": parent, "start_ns": begin, "end_ns": end, "data": data}


def trace():
    return [
        event("decode", None, 0, 1000),
        event("layer_0", 0, 1, 999),
        event(
            "moe_forward_rows",
            1,
            2,
            998,
            {
                "rows": 1,
                "capacity": 53,
                "ids": list(range(10)),
                "slots_hit": [[i, False] for i in range(10)],
                "hot_rows": 0,
                "cold_rows": 10,
            },
        ),
        event(
            "disk_read_batch",
            2,
            3,
            900,
            {
                "ok": True,
                "jobs": [[i, i] for i in range(10)],
                "workers": 4,
                "timings_job_worker_start_end_bytes_failed": [
                    [i, i % 4, (i // 4) * 20, (i // 4) * 20 + 10, 1248012, 0] for i in range(10)
                ],
            },
        ),
    ]


def test_worker_overlap_and_nested_times_are_not_added_as_exclusive_gpu():
    out = API["validate_events"](trace())
    assert len(out["disk_jobs"]) == 10 and sum(j["bytes"] for j in out["disk_jobs"]) == 12480120
    root = next(c for c in out["components"] if c["component"] == "decode")
    assert root["wall_ns"]["total"] == 1000 and root["host_self_ns"]["total"] == 2
    assert out["disk_batches"][0]["last_completion_ns"] == 50
    assert out["disk_batches"][0]["worker_finish_spread_ns"] == 20
    assert "not exclusive GPU" in out["limits"]


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "negative",
        "backward",
        "parent",
        "nesting",
        "siblings",
        "booltime",
        "routes",
        "rowdup",
        "slot",
        "hits",
        "short",
        "bytes",
        "failed",
        "worker",
        "job",
        "busyworker",
    ],
)
def test_trace_rejects_corruption(fault):
    rows = trace()
    if fault == "empty":
        rows = []
    elif fault == "negative":
        rows[0]["start_ns"] = -1
    elif fault == "backward":
        rows[0]["end_ns"] = -1
    elif fault == "parent":
        rows[1]["parent"] = 1
    elif fault == "nesting":
        rows[2]["end_ns"] = 1001
    elif fault == "siblings":
        rows.append(event("overlap", 0, 500, 800))
    elif fault == "booltime":
        rows[0]["start_ns"] = False
    elif fault == "routes":
        rows[2]["data"]["ids"] = []
    elif fault == "rowdup":
        rows[2]["data"]["ids"][0] = 1
    elif fault == "slot":
        rows[2]["data"]["slots_hit"][0][0] = 53
    elif fault == "hits":
        rows[2]["data"]["hot_rows"] = 1
    elif fault == "short":
        rows[3]["data"]["timings_job_worker_start_end_bytes_failed"].pop()
    else:
        field = {"bytes": 4, "failed": 5, "worker": 1, "job": 0, "busyworker": 2}[fault]
        rows[3]["data"]["timings_job_worker_start_end_bytes_failed"][
            4 if fault == "busyworker" else 0
        ][field] = 1 if fault in ("failed", "worker", "job") else 0
    with pytest.raises((AssertionError, ValueError, TypeError)):
        API["validate_events"](rows)


def test_distribution_keeps_zero_and_tail_samples():
    assert API["distribution"]([0, 1, 2, 3, 100])["p99"] == 100
    assert API["distribution"]([0, 1])["min"] == 0
    for invalid in ([], [float("nan")], [-1], [float("inf")]):
        with pytest.raises(AssertionError):
            API["distribution"](invalid)


def test_compression_round_trip_is_deterministic_and_never_overwrites(tmp_path):
    raw = b'[{"name":"decode"}]'
    files = []
    for name in ("a", "b"):
        path = tmp_path / name
        path.write_bytes(raw)
        API["compress"](path)
        assert not path.exists()
        files.append(Path(str(path) + ".gz"))
    assert (
        files[0].read_bytes() == files[1].read_bytes()
        and gzip.decompress(files[0].read_bytes()) == raw
    )
    original = files[0].read_bytes()
    (tmp_path / "a").write_bytes(b"other")
    with pytest.raises(AssertionError):
        API["compress"](tmp_path / "a")
    assert files[0].read_bytes() == original and (tmp_path / "a").read_bytes() == b"other"


@pytest.mark.parametrize(
    "error", [RuntimeError("crashed"), TimeoutError("silent"), KeyboardInterrupt()]
)
def test_runner_failure_never_claims_parity_and_is_saved(tmp_path, monkeypatch, error):
    plan = tmp_path / "run.json"
    plan.write_text('[ {"profile_directory":"unused"} ]')
    api = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))
    namespace = api["main"].__globals__
    namespace["ROOT"] = tmp_path
    (tmp_path / "opti.md").write_text("")

    def run_path(path):
        def failed(plan):
            raise error

        return {"main": failed}

    monkeypatch.setattr(namespace["runpy"], "run_path", run_path)
    with pytest.raises(type(error)):
        api["main"](plan)
    result = json.loads((tmp_path / "dataset-result.json").read_text())
    assert result["status"] == "failed_or_interrupted" and result["parity"] is False
    assert "failed_or_interrupted" in (tmp_path / "opti.md").read_text()
    with pytest.raises(AssertionError):
        api["main"](plan)


def test_vm_events_are_not_mislabeled_as_compressed_bytes():
    raw = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" + "\n".join(
        f"{k}: 7."
        for k in [
            "Compressions",
            "Decompressions",
            "Swapins",
            "Swapouts",
            "Pages occupied by compressor",
        ]
    )
    assert API["vm_counts"](raw)["Compressions"] == 7
    for bad in ("", raw + "\nCompressions: 8.", raw.replace("Swapouts:", "missing:")):
        with pytest.raises(AssertionError):
            API["vm_counts"](bad)


def summary_fixture():
    vm = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n" + "\n".join(
        f"{k}: 7."
        for k in [
            "Compressions",
            "Decompressions",
            "Swapins",
            "Swapouts",
            "Pages occupied by compressor",
        ]
    )
    snap = {
        "memory": {
            "process_footprint_bytes": 12_000_000_000,
            "process_lifetime_peak_bytes": 12_000_000_000,
        },
        "vm_stat": vm,
    }
    return {
        "full_logits_target_head_states_byte_exact": True,
        "logit_count": 248320,
        "tokens": [0] * 17,
        "token_hash": "e7d16c6b24f5c8f5",
        "text": "known prefix",
        "steps": [{"step": i, "token": 0, "seconds": 0.1} for i in range(1, 17)],
        "prefill_seconds": 1.0,
        "decode_seconds": 2.0,
        "diagnostic_tps": 8.0,
        "before_counts": [[0, 0, 0]] * 48,
        "prefill_counts": [[10, 0, 0]] * 48,
        "after_counts": [[170, 0, 0]] * 48,
        "before": snap,
        "prefill_boundary": snap,
        "after": snap,
    }


def test_summary_full_shape_memory_boundary_and_prefix_are_checked():
    data = summary_fixture()
    result = API["validate_summary"](
        data, "quality", {"cache_context": "known prefix continued"}, [0] * 128
    )
    assert result["decode_vm_counter_pages_delta"]["Compressions"] == 0
    assert "not model-exclusive" in result["limits"]


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "id",
        "hash",
        "text",
        "parity",
        "shape",
        "steps",
        "token",
        "nan",
        "tps",
        "counter",
        "layer",
        "peak",
        "footprint",
        "vm",
        "expected",
    ],
)
def test_summary_rejects_divergence_incomplete_nonfinite_and_memory_overrun(fault):
    data = summary_fixture()
    oracle = None
    if fault == "empty":
        data["tokens"] = []
    elif fault == "id":
        data["tokens"][0] = 248320
    elif fault == "hash":
        data["token_hash"] = "0" * 16
    elif fault == "text":
        data["text"] = "wrong"
    elif fault == "parity":
        data["full_logits_target_head_states_byte_exact"] = False
    elif fault == "shape":
        data["logit_count"] = 0
    elif fault == "steps":
        data["steps"].pop()
    elif fault == "token":
        data["steps"][0]["token"] = 1
    elif fault == "nan":
        data["decode_seconds"] = float("nan")
    elif fault == "tps":
        data["diagnostic_tps"] = 0
    elif fault == "counter":
        data["after_counts"][0] = [1, 0, 0]
    elif fault == "layer":
        data["prefill_counts"].pop()
    elif fault == "peak":
        data["after"]["memory"]["process_lifetime_peak_bytes"] += 1
    elif fault == "footprint":
        data["after"]["memory"]["process_footprint_bytes"] = 0
    elif fault == "vm":
        data["after"]["vm_stat"] = "invalid"
    elif fault == "expected":
        oracle = [1] * 128
    with pytest.raises(AssertionError):
        API["validate_summary"](
            data, "quality", {"cache_context": "known prefix continued"}, oracle
        )
