"""Replay the focused checks and strictly reparse existing physical logs, no main."""
import contextlib
import io
import json
import runpy
from pathlib import Path

HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    first = runpy.run_path(str(HERE / "quality-controller-selfcheck.py"))
q = first["q"]
s = runpy.run_path(str(q["ROOT"] / "build/ar20/speed.py"))
first["check"]("strict-marker-duplicate-false-true", lambda: q["marker"]('AR20_MODEL_COMPARE {"parity":false,"parity":true}', "AR20_MODEL_COMPARE"), True)
assert len(s["ARMS"]) == 8 and len({row[0] for row in s["ARMS"]}) == 8
assert len(s["SELECTED"]) == len(set(s["SELECTED"])) == 10
assert q["HEAD"] not in s["SELECTED"] and "MLXL3_NATIVE_CACHE_SLRU" not in s["SELECTED"]
assert all(set(flags) <= set(s["SELECTED"]) for _, flags, _ in s["ARMS"])
assert [(name, old) for name, _, old in s["ARMS"] if old] == [("control-start", True), ("control-end", True)]
checks = first["checks"] + ["speed-eight-unique-arms", "speed-ten-selected-exclude-B01-B12", "speed-arm-flags-subset", "speed-frozen-control-positions"]
report = q["read_json"](q["HERE"] / "quality/report.json")
for phase, marker in zip(report["phases"], ["AR20_MODEL_CAPTURE", "AR20_MODEL_COMPARE", "NATIVE_SSD_DIRECT_CLI"], strict=True):
    evidence = q["marker"]((q["HERE"] / "quality" / (phase["name"] + ".log")).read_text(), marker)
    assert evidence == phase["evidence"]
    if phase["name"] != "cli":
        q["paths"](evidence, [key for key, value in phase["flags"].items() if value == "1"])
    checks.append("offline-" + phase["name"])
result = {"status": "validated_selfchecks", "checks": len(checks), "names": checks, "root_controller_sha256": q["sha"](q["ROOT"] / "build/ar20/quality.py"), "speed_sha256": q["sha"](q["ROOT"] / "build/ar20/speed.py"), "offline_quality_log_phases": 3, "scope": "synthetic Python and strict offline log reparse only; no GPU/Cargo"}
print(json.dumps(result, indent=2))
