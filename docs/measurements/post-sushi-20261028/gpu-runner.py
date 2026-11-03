"""Frozen POSTPROFILE05 runner: existing process supervisor with campaign guards."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
MONITOR = (
    ROOT
    / "docs/measurements/flash-next-ssd-impl/opt-43-mtp-route-head/speed-heads/bench-supervisor.txt"
)
SUPERVISOR = ROOT / "docs/measurements/slotstream-20261026/supervisor.py"


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_vm(current, baseline):
    for key in ("Swapouts", "Swapins"):
        value = current[key]
        assert type(value) is int and value >= baseline[key], "invalid VM counter"
        assert value - baseline[key] <= (0 if key == "Swapouts" else 256 << 20), key


def run(plan, anchor_path, identity_path):
    identity = json.loads(identity_path.read_text())
    anchor = json.loads(anchor_path.read_text())
    baseline = {key: anchor["vm_bytes"][key] for key in ("Swapouts", "Swapins")}
    for name, sha in identity["sources"].items():
        assert file_sha(ROOT / name) == sha, name
    binaries = {v["path"]: v["sha256"] for v in identity["binaries"].values()}
    items = json.loads(plan.read_text())
    assert items and len({v["name"] for v in items}) == len(items)
    for item in items:
        binary = item["command"][0]
        assert binary in binaries and file_sha(Path(binary)) == binaries[binary]
        assert item["require_quiet"] is True and 0 < item["timeout_seconds"] <= 900
        if "native_test" in item:
            assert item["command"][1:] == [
                item["native_test"],
                "--exact",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ]
            assert "1 passed; 0 failed" in item["required_text"] and item["expected_exit"] == 0
    original = runpy.run_path(str(MONITOR))

    def before(record):
        vm = record["before"]["vm_bytes"]
        check_vm(vm, baseline)
        available = vm["Pages free"] + vm["File-backed pages"]
        record["available_estimate_bytes"] = available
        assert available >= record["minimum_available_bytes"], "load headroom estimate"

    def monitor(child, record, parser):
        def checked(text):
            vm = parser(text)
            record["trigger_vm_bytes"] = vm
            check_vm(vm, baseline)
            for key in ("Swapouts", "Swapins"):
                field = "maximum_live_" + key.lower()[:-1] + "_bytes"
                record[field] = max(record.get(field, 0), vm[key] - baseline[key])
            return vm

        result = original["monitor"](child, record, checked)
        check_vm(original["snapshot"](parser)["vm_bytes"], baseline)
        return result

    def snapshot(parser):
        result = original["snapshot"](parser)
        check_vm(result["vm_bytes"], baseline)
        return result

    previous = {**original, "snapshot": snapshot, "monitor": monitor, "before_launch": before}
    supervisor = runpy.run_path(str(SUPERVISOR))
    supervisor["main"].__globals__["runpy"] = SimpleNamespace(
        run_path=lambda path: previous if Path(path).resolve() == MONITOR else runpy.run_path(path)
    )
    supervisor["main"](plan)


if __name__ == "__main__":
    run(*(Path(value) for value in sys.argv[1:]))
