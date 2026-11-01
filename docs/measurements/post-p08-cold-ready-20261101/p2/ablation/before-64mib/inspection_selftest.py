"""Reuse the P2 failure fixtures for the fixed P/U/D ablation schedule."""

import contextlib
import runpy
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent


def main():
    shared = runpy.run_path(str(BASE.parent / "inspection_selftest.py"))
    adapter = runpy.run_path(str(BASE / "inspection.py"))
    shared["fixture"].__globals__.update(adapter=adapter, BASE=BASE)
    execute = shared["execute"]
    execute.__globals__["fixture"] = shared["fixture"]
    replace = shared["old"]["replace"]
    changes = [
        lambda v: v["manifest"]["plans"].reverse(),
        replace("report", ["environment", "MLXL3_SSD_COLD_READY"], "0"),
        replace("report", ["binary_sha256"], "0" * 64),
        replace("supervisor", ["commands", 0, "maximum_live_swapin_bytes"], (1 << 30) + 1),
    ]
    cases = [(i, None) for i in range(1, 7)] + [(2, change) for change in changes]
    with tempfile.TemporaryDirectory() as folder, contextlib.chdir(folder):
        for i, (position, change) in enumerate(cases):
            context = adapter["configured"]("AR")
            error, _ = execute(context, Path(folder) / str(i), "AR", 128, position, change)
            assert (error is not None) == (change is not None), (i, error)
    print("6 positive ablation inspections, 4 negative cases rejected")


if __name__ == "__main__":
    main()
