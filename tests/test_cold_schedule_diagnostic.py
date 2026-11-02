"""Cold scheduling counters cannot imply GPU idle or validate partial traces."""

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))


def cold_schedule_trace(priority=True, rows=2):
    records = [
        {"name": name, "parent": parent, "start_ns": begin, "end_ns": end, "data": data}
        for name, parent, begin, end, data in [
            ("decode", None, 0, 1000, None),
            ("layer_0", 0, 1, 999, None),
            ("moe_forward_rows", 1, 2, 998, {"capacity": 53}),
            ("cold_ready_batch", 2, 3, 900, None),
        ]
    ]
    ids = list(range(rows * 10))
    ids[11], ids[16] = 3, 7
    route = records[2]["data"]
    route.update(
        rows=rows,
        ids=ids,
        slots_hit=[[expert, expert not in (1, 3, 7, 12)] for expert in ids],
        hot_rows=rows * 10 - 6,
        cold_rows=6,
    )
    records[3]["name"] = "cold_ready_batch"
    records[3]["data"] = {
        "status": "finished",
        "io_ok": True,
        "route_rows": rows,
        "priority": priority,
        "jobs": 4,
        "jobs_mapping": [[3, 3], [7, 7], [1, 1], [12, 12]]
        if priority
        else [[1, 1], [3, 3], [7, 7], [12, 12]],
        "multiplicities": [2, 2, 1, 1] if priority else [1, 2, 2, 1],
        "cold_occurrences": [[1, 1], [3, 3], [7, 7], [11, 3], [12, 12], [16, 7]],
        "cold_rows": 6,
        "workers": 4,
        "schedule": "dynamic",
        "ready_initial": 2,
        "w1_rows": 3,
        "w2_rows": 3,
        "split": True,
        "timings_job_worker_start_end_bytes_failed": [
            [0, 0, 0, 40, 1248012, 0],
            [1, 1, 0, 20, 1248012, 0],
            [2, 2, 0, 50, 1248012, 0],
            [3, 3, 0, 10, 1248012, 0],
        ],
    }
    return records


def test_cold_schedule_diagnostic_uses_completed_jobs_not_launch_prefix():
    data = API["validate_events"](cold_schedule_trace())["disk_batches"][0]["cold_schedule"]
    assert data["status"] == "validated_schedule_only"
    assert data["duplicated_jobs"] == 2 and data["duplicated_job_fraction"] == 0.5
    assert data["first_two_completed_jobs"] == [3, 1]
    assert data["first_two_completed_slots"] == [12, 7]
    assert data["first_two_completed_occurrences"] == 3
    assert data["read_tail_after_second_completion_ns"] == 30
    assert "not readiness publication, GPU idle or speedup" in data["limits"]


@pytest.mark.parametrize("rows", [2, 3, 4])
def test_cold_schedule_accepts_original_launch_order_and_wide_reference(rows):
    data = API["validate_events"](cold_schedule_trace(False, rows))["disk_batches"][0][
        "cold_schedule"
    ]
    assert data["route_rows"] == rows and data["priority"] is False
    assert data["first_two_completed_occurrences"] == 3


def test_cold_schedule_all_ready_and_late_resident_occurrences_are_distinct_from_jobs():
    records = cold_schedule_trace()
    data = records[3]["data"]
    data["cold_occurrences"].insert(3, [8, 8])
    data.update(cold_rows=7, split=False, ready_initial=4, w1_rows=0, w2_rows=7)
    result = API["validate_events"](records)["disk_batches"][0]["cold_schedule"]
    assert result["jobs"] == 4 and result["w2_rows"] == 7 and result["split"] is False
    assert result["first_two_completed_occurrences"] == 3


def test_legacy_cold_schedule_diagnostic_is_explicitly_unavailable():
    records = cold_schedule_trace()
    fields = ("route_rows", "priority", "cold_occurrences", "multiplicities")
    for field in fields:
        del records[3]["data"][field]
    result = API["validate_events"](records)["disk_batches"][0]["cold_schedule"]
    assert result == {"status": "unavailable_legacy", "missing_fields": list(fields)}


def test_cold_schedule_without_a_routed_parent_cannot_be_qualified():
    records = cold_schedule_trace()
    with pytest.raises(AssertionError, match="no routed parent"):
        API["cold_schedule_diagnostic"](records[3]["data"])


@pytest.mark.parametrize(
    "fault",
    [
        "missing_new_field",
        "incomplete",
        "failed_io",
        "duplicate_occurrence",
        "missing_occurrence",
        "wrong_occurrence_slot",
        "out_of_bounds_occurrence",
        "wrong_expert",
        "duplicate_mapping_slot",
        "wrong_multiplicity",
        "boolean_multiplicity",
        "wrong_stable_tie_order",
        "incomplete_timings",
        "missing_job",
        "priority_on_wide_rows",
        "wrong_row_count",
        "wrong_waves",
        "wrong_all_ready",
    ],
)
def test_cold_schedule_rejects_incomplete_or_inconsistent_new_records(fault):
    records = cold_schedule_trace()
    data = records[3]["data"]
    if fault == "missing_new_field":
        del data["multiplicities"]
    elif fault == "incomplete":
        data["status"] = "incomplete"
    elif fault == "failed_io":
        data["io_ok"] = False
    elif fault == "duplicate_occurrence":
        data["cold_occurrences"][1] = data["cold_occurrences"][0][:]
    elif fault == "missing_occurrence":
        data["cold_occurrences"].pop()
        data.update(cold_rows=5, w2_rows=2)
    elif fault == "wrong_occurrence_slot":
        data["cold_occurrences"][0][1] = 8
    elif fault == "out_of_bounds_occurrence":
        data["cold_occurrences"][-1][0] = 20
    elif fault == "wrong_expert":
        data["jobs_mapping"][0][0] = 13
    elif fault == "duplicate_mapping_slot":
        data["jobs_mapping"][0][1] = 7
    elif fault == "wrong_multiplicity":
        data["multiplicities"][0] = 1
    elif fault == "boolean_multiplicity":
        data["multiplicities"][0] = True
    elif fault == "wrong_stable_tie_order":
        data["jobs_mapping"][0], data["jobs_mapping"][1] = (
            data["jobs_mapping"][1],
            data["jobs_mapping"][0],
        )
    elif fault == "incomplete_timings":
        data["timings_job_worker_start_end_bytes_failed"].pop()
    elif fault == "missing_job":
        data["jobs_mapping"].pop()
        data["multiplicities"].pop()
        data["timings_job_worker_start_end_bytes_failed"].pop()
        data["jobs"] = 3
    elif fault == "priority_on_wide_rows":
        records = cold_schedule_trace(True, 3)
    elif fault == "wrong_row_count":
        data["route_rows"] = 3
    elif fault == "wrong_waves":
        data["w2_rows"] = 2
    elif fault == "wrong_all_ready":
        data.update(split=False, ready_initial=2, w1_rows=0, w2_rows=6)
    with pytest.raises((AssertionError, KeyError)):
        API["validate_events"](records)
