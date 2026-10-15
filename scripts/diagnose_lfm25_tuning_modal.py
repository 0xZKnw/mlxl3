"""Sequential fresh-process controls with explicit EXL3 autotune caches."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import modal
import quantize_lfm25_modal as base
from diagnose_lfm25_routes import validate_plan
from diagnose_lfm25_routes_modal import validate_result
from diagnose_lfm25_tuning import LEGACY_SHA256, STAGES

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/diagnostic37-fixed-tune")
LEGACY_CAPTURE = Path("/data/diagnostic23")
LEGACY = Path("/opt/mlxl3/diagnostic23-original.py")
VOLUME = "mlxl3-lfm25-v3-20261015"
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
image = base.image.env({"PYTHONPATH": "/opt/mlxl3"})
if modal.is_local():
    for name in (
        "diagnose_lfm25_tuning",
        "diagnose_lfm25_tuning_modal",
        "diagnose_lfm25_routes",
        "diagnose_lfm25_routes_modal",
        "quantize_lfm25_modal",
    ):
        image = image.add_local_file(
            ROOT / "scripts" / f"{name}.py", f"/opt/mlxl3/{name}.py", copy=True
        )
    image = image.add_local_file(
        ROOT
        / "docs/measurements/quant-lfm25-diagnostics35-36-20261015/diagnostic23-original.py.txt",
        str(LEGACY),
        copy=True,
    ).add_local_python_source("quantize_lfm25_modal")
app = modal.App("mlxl3-lfm25-tuning37", image=image)


def cache_name(stage: str) -> str:
    """Choose the registered shared or independent cache.

    raises: ValueError
    post: __return__ in ('cache-a.bin', 'cache-b.bin')
    post: (__return__ == 'cache-b.bin') == (stage == 'cold-b')
    """
    if stage not in STAGES:
        raise ValueError("Unknown tuning diagnostic stage")
    return "cache-b.bin" if stage == "cold-b" else "cache-a.bin"


def cache_snapshot(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False, "bytes": 0, "sha256": None}
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "exists": True, "bytes": path.stat().st_size, "sha256": digest}


def validate_capture(report: dict, stage: str) -> dict:
    if not isinstance(report, dict) or report.get("status") != "complete":
        raise ValueError("Tuning diagnostic capture did not complete")
    if stage == "routes-a":
        result = validate_result(report)
        if result.get("ids_only") != "approved" or len(result["arms"]) != 6:
            raise ValueError("All six registered routing arms must complete")
        return result
    windows = report.get("windows")
    if not isinstance(windows, dict) or set(windows) != {"window1", "window5"}:
        raise ValueError("Incomplete legacy diagnostic windows")
    for record in windows.values():
        if (
            not isinstance(record, dict)
            or record.get("shape") != [1024, 128000]
            or record.get("dtype") != "float16"
            or record.get("bytes", 0) < 262144000
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in record["sha256"])
        ):
            raise ValueError("Incomplete legacy logit evidence")
    return report


def execute(plan: dict) -> dict:
    validate_plan(plan)
    if DATA.exists() or LEGACY_CAPTURE.exists():
        raise FileExistsError("Preserve earlier tuning or legacy diagnostic evidence")
    with LEGACY.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != LEGACY_SHA256:
            raise ValueError("Pinned legacy diagnostic source changed")
    DATA.mkdir()
    plan_file = DATA / "input.json"
    plan_file.write_text(json.dumps(plan, allow_nan=False) + "\n")
    started = time.monotonic()
    report = {
        "status": "running",
        "publication": False,
        "causal_interpretation": False,
        "stages": {},
        "plan_file": str(plan_file),
    }

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = DATA / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(DATA / "report.json")

    save()
    try:
        for stage in STAGES:
            report["stage"] = stage
            folder, cache = DATA / stage, DATA / cache_name(stage)
            if folder.exists() or LEGACY_CAPTURE.exists():
                raise FileExistsError("Preserve existing stage/legacy output")
            before = cache_snapshot(cache)
            warm = stage in ("warm-a", "routes-a")
            if before["exists"] != warm or warm and before["bytes"] <= 0:
                raise ValueError("Tuning cache does not match the registered stage")
            if warm:
                previous = "cold-a" if stage == "warm-a" else "warm-a"
                if before != report["stages"][previous]["run"]["cache_after"]:
                    raise ValueError("Shared tuning cache changed between registered stages")
            command = [
                sys.executable,
                "-u",
                "/opt/mlxl3/diagnose_lfm25_tuning.py",
                stage,
                str(plan_file),
                str(folder),
                str(cache),
                str(LEGACY),
            ]
            report["active_command"] = command
            save()
            base.run_logged(command, DATA / f"{stage}.log", 540 if stage == "routes-a" else 180)
            envelope = json.loads((folder / "run.json").read_text())
            after = cache_snapshot(cache)
            capture_path = folder / "capture/report.json"
            if (
                not isinstance(envelope, dict)
                or envelope.get("status") != "complete"
                or envelope.get("stage") != stage
                or envelope.get("cache_path") != str(cache)
                or envelope.get("capture_report") != str(capture_path)
                or envelope.get("legacy_sha256") != LEGACY_SHA256
                or envelope.get("cache_before") != before
                or envelope.get("cache_after") != after
                or not after["exists"]
                or after["bytes"] <= 0
                or envelope.get("publication") is not False
                or envelope.get("causal_interpretation") is not False
            ):
                raise ValueError("Incomplete or mismatched tuning stage envelope")
            capture = validate_capture(json.loads(capture_path.read_text()), stage)
            if LEGACY_CAPTURE.exists():
                raise FileExistsError("Legacy capture was not moved after child completion")
            report["stages"][stage] = {"run": envelope, "capture": capture}
            save()
        report.update(status="complete", stage="awaiting_cpu_comparison")
        return report
    except BaseException as error:
        report.update(status="failed", launcher_error=repr(error), causal_interpretation=False)
        raise
    finally:
        save()
        volume.commit()


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=1260,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": volume},
)
def diagnose(plan: dict) -> dict:
    return execute(plan)


@app.local_entrypoint()
def main():
    plan = json.loads((ROOT / "build/diagnostic23-input.json").read_text())
    validate_plan(plan)
    result = diagnose.remote(plan)
    (ROOT / "build/lfm25-tuning37-cloud.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result), flush=True)
