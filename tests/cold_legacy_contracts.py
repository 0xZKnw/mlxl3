"""Symbolic check of the actual historical ColdReady validator, with a fixed valid route."""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))
TRACE = json.loads((ROOT / "tests/fixtures/cold-ready-d5a776e5.json").read_text())


def historical_wave_counters(ready: int, w1: int, w2: int, split: bool) -> bool:
    """
    post: _ == (2 <= ready <= 4 and w1 >= 0 and w2 >= 0 and w1 + w2 == 4 and ((split and ready < 4 and w1 == ready) or (not split and ready == 4 and w1 == 0)))
    """
    events = [dict(event) for event in TRACE]
    events[3]["data"] = {
        **TRACE[3]["data"],
        "ready_initial": ready,
        "w1_rows": w1,
        "w2_rows": w2,
        "split": split,
    }
    try:
        result = API["validate_events"](events)
    except AssertionError:
        return False
    assert result["disk_jobs"] == result["disk_batches"] == []
    assert result["disk_detail_coverage"] == "unavailable_legacy"
    return True


def read_byte_accounting(observed_jobs: int, legacy_jobs: int, expected: int) -> bool:
    """
    pre: 0 <= observed_jobs <= 512 and 0 <= legacy_jobs <= 20
    post: _ == (expected == (observed_jobs + legacy_jobs) * 1248012)
    """
    stats = {
        "disk_jobs": [{"layer": 0, "bytes": observed_jobs * 1_248_012}],
        "legacy_cold_batches": [{"layer": 0, "jobs": legacy_jobs}] if legacy_jobs else [],
    }
    try:
        result = API["validate_read_bytes"](stats, expected)
    except AssertionError:
        return False
    assert result["observed_job_bytes"] == observed_jobs * 1_248_012
    assert result["legacy_completed_jobs"] == legacy_jobs
    assert result["status"] == (
        "legacy_completed_job_counts" if legacy_jobs else "observed_job_bytes"
    )
    return True
