import copy

import pytest

from scripts.analyze_flash_cache_width import EXPERT_BYTES, summarize


def fixture(split=False):
    events = [{"name": "decode", "parent": None, "data": None}]
    events.append({"name": "verify_draft", "parent": 0, "data": {"rows": 2}})
    for layer in range(48):
        parent = len(events)
        events.append({"name": f"layer_{layer}", "parent": 1, "data": None})
        for cohort in range(2 if split else 1):
            hit = split and cohort == 1
            moe = len(events)
            events.append(
                {
                    "name": "moe_forward_rows",
                    "parent": parent,
                    "data": {
                        "rows": 1 if split else 2,
                        "ids": list(range(10)) * (1 if split else 2),
                        "slots_hit": [[id_, hit] for id_ in range(10)] * (1 if split else 2),
                        "hot_rows": 10 if hit else 0,
                        "cold_rows": 0 if hit else 10 if split else 20,
                    },
                }
            )
            if not hit:
                events.append(
                    {
                        "name": "disk_read_batch",
                        "parent": moe,
                        "data": {"ok": True, "jobs": [[id_, id_] for id_ in range(10)]},
                    }
                )
    counts = [[10 if split else 0, 10 if split else 20, 10 * EXPERT_BYTES] for _ in range(48)]
    return events, counts


def test_cohort_boundaries_change_hits_without_changing_requested_bytes():
    combined = summarize(*fixture(), emitted=2)
    split = summarize(*fixture(True), emitted=2)
    assert combined["verified_windows_by_rows"] == split["verified_windows_by_rows"] == {2: 1}
    assert combined["logical_hits"] == 0 and combined["logical_misses"] == 960
    assert split["logical_hits"] == split["logical_misses"] == 480
    assert (
        combined["requested_record_bytes"] == split["requested_record_bytes"] == 480 * EXPERT_BYTES
    )
    assert combined["requested_bytes_per_token"] == 240 * EXPERT_BYTES
    assert combined["expert_multiplicity_per_window_layer"] == {2: 480}
    assert combined["expert_multiplicity_per_moe_cohort"] == {2: 480}
    assert split["expert_multiplicity_per_moe_cohort"] == {1: 960}


def test_ordinary_tail_is_counted_in_bytes_and_occurrences_separately_from_verify():
    events, counts = fixture()
    step = len(events)
    events.append({"name": "output_step", "parent": 0, "data": None})
    for layer in range(48):
        parent = len(events)
        events.append({"name": f"layer_{layer}", "parent": step, "data": None})
        events.append(
            {
                "name": "moe_forward_rows",
                "parent": parent,
                "data": {
                    "rows": 1,
                    "ids": list(range(10)),
                    "slots_hit": [[id_, True] for id_ in range(10)],
                    "hot_rows": 10,
                    "cold_rows": 0,
                },
            }
        )
        counts[layer][0] += 10
    summary = summarize(events, counts, 3)
    assert summary["verified_windows_by_rows"] == {2: 1}
    assert summary["ordinary_target_steps"] == 1 and summary["target_rows"] == 3
    assert summary["logical_hits"] == 480 and summary["logical_misses"] == 960
    assert summary["requested_bytes_per_token"] == 160 * EXPERT_BYTES
    assert summary["by_target_width_hits_misses_bytes"][1] == [480, 0, 0]


@pytest.mark.parametrize(
    "mutation", ["counter", "missing", "width", "future_parent", "false_hit", "duplicate_job"]
)
def test_incomplete_or_corrupt_data_cannot_announce_exact_counters(mutation):
    events, counts = copy.deepcopy(fixture())
    if mutation == "counter":
        counts[0][2] += 1
    elif mutation == "missing":
        events[3]["name"] = "missing_route"
    elif mutation == "width":
        events[1]["data"]["rows"] = 9
    elif mutation == "future_parent":
        events[3]["parent"] = len(events)
    elif mutation == "false_hit":
        events[3]["data"]["slots_hit"][0][1] = "false"
    else:
        events[4]["data"]["jobs"][1][1] = 0
    with pytest.raises(AssertionError):
        summarize(events, counts, 2)


def test_empty_trace_and_zero_outputs_are_rejected():
    with pytest.raises(AssertionError):
        summarize([], [[0, 0, 0]] * 48, 1)
    with pytest.raises(AssertionError):
        summarize(*fixture(), emitted=0)
