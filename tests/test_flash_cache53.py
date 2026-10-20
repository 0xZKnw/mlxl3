"""Replays must preserve every row of48layers and never infer missing initial LFU state."""

import gzip
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "scripts/replay_flash_cache53.py"))
FIXTURE = runpy.run_path(str(ROOT / "tests/test_flash_profile_export.py"))["fixture"]


def test_extract_preserves_both_rows_topk_order_and_every_layer():
    events, _ = FIXTURE()
    actual = API["routes"](events, "decode")
    assert len(actual) == 48
    assert actual == [[list(range(20))] for _ in range(48)]


def test_reads_compressed_events_through_the_existing_loader(tmp_path):
    events, _ = FIXTURE()
    (tmp_path / "trace_1-decode.json.gz").write_bytes(gzip.compress(json.dumps(events).encode()))
    assert API["read_routes"](tmp_path, "decode") == [[list(range(20))] for _ in range(48)]


def test_corrupt_compressed_input_has_a_bounded_explicit_failure(tmp_path, monkeypatch):
    source = tmp_path / "source"
    directory = source / "natural-OFF-c3-MAIN/profile"
    directory.mkdir(parents=True)
    (directory / "trace_1-summary.json").write_text("{}")
    (directory / "trace_1-prefill.json.gz").write_bytes(b"invalid" * 10000)
    monkeypatch.setitem(API["main"].__globals__["API"], "validate_summary", lambda *_: None)
    output = tmp_path / "output"
    with pytest.raises(gzip.BadGzipFile):
        API["main"](source, output)
    raw = (output / "extraction.json").read_text()
    assert len(raw) < 2200
    assert json.loads(raw)["status"] == "failed"


@pytest.mark.parametrize(
    "fault",
    ["empty", "wrong_root", "missing_layer", "wrong_capacity", "duplicate", "overflow", "rows"],
)
def test_invalid_route_traces_cannot_become_calibration(fault):
    events, _ = FIXTURE()
    target = next(e for e in events if e["name"] == "moe_forward_rows")
    if fault == "empty":
        events = []
    elif fault == "wrong_root":
        events[0]["name"] = "prefill"
    elif fault == "missing_layer":
        target["name"] = "not_a_route"
    elif fault == "wrong_capacity":
        target["data"]["capacity"] = 54
    elif fault == "duplicate":
        target["data"]["ids"][1] = 0
    elif fault == "overflow":
        target["data"]["ids"][1] = 512
    elif fault == "rows":
        target["data"]["rows"] = 1
    with pytest.raises(AssertionError):
        API["routes"](events, "decode")


def test_failed_extraction_is_explicit_and_existing_results_are_preserved(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        API["main"](tmp_path / "missing", output)
    content = (output / "extraction.json").read_text()
    assert '"status": "failed"' in content and '"confirmed_tps_gain": false' in content
    with pytest.raises(FileExistsError):
        API["main"](tmp_path / "missing", output)
    assert (output / "extraction.json").read_text() == content
