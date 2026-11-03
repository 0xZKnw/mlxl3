"""P09 uses the existing supervisors and the authorized cumulative 32 MiB budget."""

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ANCHOR = Path(__file__).with_name("anchor.json")
MAXIMUM_NEW_SWAPOUT_BYTES = 32 << 20
CORE = runpy.run_path(
    str(ROOT / "docs/measurements/post-p08-dense-epilogue-20261031/bounded-gpu-runner.py")
)
CAPTURE = runpy.run_path(str(ROOT / "docs/measurements/decode-profile-20261031/capture.py"))


def load_baseline(path):
    assert path.resolve() == ANCHOR.resolve(), "fixed P09 anchor"
    data = json.loads(path.read_text())
    assert data["campaign"] == "DECODE-PROFILE-09"
    baseline = {key: data["vm_bytes"][key] for key in ("Swapouts", "Swapins")}
    assert all(type(value) is int and value >= 0 for value in baseline.values())
    return baseline


def run(plan, identity):
    sources = json.loads(identity.read_text())["sources"]
    assert {str(ANCHOR.resolve()), str(Path(__file__).resolve())} <= set(sources)
    load_baseline(ANCHOR)
    if isinstance(json.loads(plan.read_text()), dict):
        CAPTURE["guard"].__globals__["MAXIMUM_NEW_SWAPOUT_BYTES"] = MAXIMUM_NEW_SWAPOUT_BYTES
        return CAPTURE["run"](plan, ANCHOR, identity)
    CORE["run"].__globals__["load_baseline"] = load_baseline
    CORE["check_vm"].__globals__["MAXIMUM_NEW_SWAPOUT_BYTES"] = MAXIMUM_NEW_SWAPOUT_BYTES
    return CORE["run"](plan, ANCHOR, identity)


if __name__ == "__main__":
    run(*(Path(value) for value in sys.argv[1:]))
