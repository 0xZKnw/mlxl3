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
        resident_gpu_rows=rows * 10 - 6,
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
    assert data["readiness_validation"] == "bounded_multiplicities"
    assert data["timing_bounds_validation"] == "batch_wall"
    assert data["parent_partition_validation"] == "resident_count"
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
    records[2]["data"]["resident_gpu_rows"] -= 1
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


@pytest.mark.parametrize("rows", [2, 3, 4])
@pytest.mark.parametrize("exact", [False, True])
def test_ready_wave_size_is_checked_against_multiplicities_and_optional_snapshot(rows, exact):
    records = cold_schedule_trace(False, rows)
    route, data = records[2]["data"], records[3]["data"]
    for row in range(2, rows):
        index = row * 10
        route["ids"][index] = 3
        route["slots_hit"][index] = [3, False]
        route["hot_rows"] -= 1
        route["cold_rows"] += 1
        route["resident_gpu_rows"] -= 1
        data["cold_occurrences"].append([index, 3])
    data.update(cold_rows=rows + 4, w1_rows=rows + 1, w2_rows=3)
    data["multiplicities"][1] = rows
    if exact:
        data["ready_job_indices"] = [3, 1]
    result = API["validate_events"](records)["disk_batches"][0]["cold_schedule"]
    assert result["readiness_validation"] == (
        "exact_indices" if exact else "bounded_multiplicities"
    )
    assert result["first_two_completed_occurrences"] == rows + 1
    assert result["w1_rows"] == rows + 1
    # Two jobs cannot expose more than the sum of the two largest multiplicities.
    data.update(w1_rows=rows + 3, w2_rows=1)
    with pytest.raises(AssertionError):
        API["validate_events"](records)


def test_exact_ready_snapshot_rejects_wave_count_that_bounds_alone_allow():
    records = cold_schedule_trace()
    data = records[3]["data"]
    data.update(w1_rows=2, w2_rows=4)
    bounded = API["validate_events"](records)["disk_batches"][0]["cold_schedule"]
    assert bounded["readiness_validation"] == "bounded_multiplicities"
    data["ready_job_indices"] = [3, 1]
    with pytest.raises(AssertionError):
        API["validate_events"](records)


@pytest.mark.parametrize("indices", [None, [], [1], [1, 1], [1, 4], [-1, 1], [True, 1], [1.0, 3]])
def test_ready_snapshot_must_be_complete_unique_integer_job_indices(indices):
    records = cold_schedule_trace()
    records[3]["data"]["ready_job_indices"] = indices
    with pytest.raises(AssertionError):
        API["validate_events"](records)


def test_partial_ready_snapshot_cannot_masquerade_as_a_legacy_record():
    records = cold_schedule_trace()
    data = records[3]["data"]
    for key in ("route_rows", "priority", "cold_occurrences", "multiplicities"):
        del data[key]
    data["ready_job_indices"] = [3, 1]
    with pytest.raises(AssertionError, match="incomplete Cold schedule diagnostic"):
        API["validate_events"](records)


def test_all_ready_snapshot_retains_zero_first_wave_and_full_second_wave():
    records = cold_schedule_trace()
    records[3]["data"].update(
        ready_job_indices=[3, 1, 0, 2], ready_initial=4, split=False, w1_rows=0, w2_rows=6
    )
    result = API["validate_events"](records)["disk_batches"][0]["cold_schedule"]
    assert result["readiness_validation"] == "exact_indices"
    assert result["w1_rows"] == 0 and result["w2_rows"] == 6


def test_new_read_timestamps_must_fit_the_containing_batch_without_rejecting_legacy():
    records = cold_schedule_trace()
    data = records[3]["data"]
    data["timings_job_worker_start_end_bytes_failed"][2][3] = 10**12
    with pytest.raises(AssertionError, match="timing exceeds batch span"):
        API["validate_events"](records)
    for key in ("route_rows", "priority", "cold_occurrences", "multiplicities"):
        del data[key]
    assert (
        API["validate_events"](records)["disk_batches"][0]["cold_schedule"]["status"]
        == "unavailable_legacy"
    )


@pytest.mark.parametrize("residents", [-1, 21, True, 15])
def test_cold_occurrence_count_must_match_the_known_parent_resident_count(residents):
    records = cold_schedule_trace()
    records[2]["data"]["resident_gpu_rows"] = residents
    with pytest.raises(AssertionError):
        API["validate_events"](records)


def test_parent_without_historical_resident_count_has_an_explicit_validation_gap():
    records = cold_schedule_trace()
    del records[2]["data"]["resident_gpu_rows"]
    result = API["validate_events"](records)["disk_batches"][0]["cold_schedule"]
    assert result["parent_partition_validation"] == "unavailable_legacy_parent"


def test_cohort_batch_uses_its_local_routed_parent_instead_of_the_outer_window():
    records = cold_schedule_trace(False)
    records.insert(
        2,
        {
            "name": "moe_unique_cohorts",
            "parent": 1,
            "start_ns": 2,
            "end_ns": 998,
            "data": {"requested_rows": 4},
        },
    )
    records[3]["parent"], records[4]["parent"] = 2, 3
    result = API["validate_events"](records)["disk_batches"][0]["cold_schedule"]
    assert result["route_rows"] == 2 and result["parent_partition_validation"] == "resident_count"


def test_extra_hot_occurrence_cannot_be_counted_as_cold_without_matching_parent_partition():
    records = cold_schedule_trace()
    records[3]["data"]["cold_occurrences"].insert(3, [8, 8])
    records[3]["data"].update(cold_rows=7, w2_rows=4)
    with pytest.raises(AssertionError, match="Cold/Hot parent count mismatch"):
        API["validate_events"](records)


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
