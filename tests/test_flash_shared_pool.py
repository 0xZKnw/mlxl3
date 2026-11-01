"""Current pool extraction must preserve every physical route observation."""

import copy
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTRACT = runpy.run_path(str(ROOT / "docs/measurements/issue58-20261029/extract-current.txt"))[
    "extract"
]


def events():
    result = [{"name": "decode", "parent": None, "start_ns": 0, "end_ns": 144, "data": None}]
    for layer in range(48):
        parent = len(result)
        result.append(
            {
                "name": f"layer_{layer}",
                "parent": 0,
                "start_ns": layer * 3,
                "end_ns": layer * 3 + 3,
                "data": None,
            }
        )
        result.append(
            {
                "name": "moe_forward_rows",
                "parent": parent,
                "start_ns": layer * 3 + 1,
                "end_ns": layer * 3 + 2,
                "data": {
                    "rows": 1,
                    "capacity": 53,
                    "completed_hot": False,
                    "ids": list(range(10)),
                    "slots_hit": [[i, False] for i in range(10)],
                    "hot_rows": 0,
                    "cold_rows": 10,
                },
            }
        )
    return result


def test_extraction_preserves_every_slot_and_hit():
    source = events()
    source[2]["data"]["slots_hit"][0] = [52, True]
    source[2]["data"].update(hot_rows=1, cold_rows=9)
    before = copy.deepcopy(source)
    result = EXTRACT(source, 1)
    assert source == before and len(result) == 48
    assert result[0]["slots_hit"][0] == [52, True]
    assert result[-1]["ids"] == list(range(10))


@pytest.mark.parametrize("fault", range(9))
def test_extraction_rejects_incomplete_invalid_or_wrong_execution_path(fault):
    source = events()
    if fault == 0:
        source.clear()
    elif fault == 1:
        source.pop()
    elif fault == 2:
        source[1]["name"] = "layer_1"
    elif fault == 3:
        source[2]["data"]["rows"] = 2
    elif fault == 4:
        source[2]["data"]["capacity"] = 54
    elif fault == 5:
        source[2]["data"]["completed_hot"] = True
    elif fault == 6:
        source[2]["data"]["ids"][0] = 512
    elif fault == 7:
        source[2]["data"]["slots_hit"][0] = [53, False]
    else:
        source[2]["parent"] = 2
    with pytest.raises(AssertionError):
        EXTRACT(source, 1)


@pytest.mark.parametrize("steps", [0, 513, True, 1.5])
def test_extraction_rejects_invalid_step_counts(steps):
    with pytest.raises(AssertionError):
        EXTRACT(events(), steps)
