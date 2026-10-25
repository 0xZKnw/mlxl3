"""Run the bounded, read-only route diagnostic on the existing private Volume."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import modal
import quantize_lfm25_modal as base

DATA = Path("/data/diagnostic37-routes-recovery")
TUNE_CACHE = Path("/data/diagnostic37-fixed-tune-trace/cache-a-recovery.bin")
CACHE_BYTES = 176
CACHE_SHA256 = "1cd9c092bd749dc10fdc017a936477ed9ef0d6072d4af401711d5a7ce5bb0f65"
VOLUME = "mlxl3-lfm25-v3-20261015"
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
image = base.image.pip_install("modal==1.5.5").env(
    {"PYTHONPATH": "/opt/mlxl3", "EXLLAMAV3_TUNE_CACHE": str(TUNE_CACHE)}
)
if modal.is_local():
    image = image.add_local_file(
        Path(__file__).with_name("diagnose_lfm25_routes.py"),
        "/opt/mlxl3/diagnose_lfm25_routes.py",
        copy=True,
    ).add_local_python_source("quantize_lfm25_modal")
app = modal.App("mlxl3-lfm25-routes-recovery37", image=image)


def validate_result(report: dict) -> dict:
    if not isinstance(report, dict) or report.get("status") != "complete":
        raise ValueError("Route diagnostic child did not complete")
    expected = {"source", "normal", "own_routes", "source_routes"}
    if report.get("ids_only") == "approved":
        expected.update(("own_ids", "source_ids"))
    elif report.get("ids_only") == "not_executed_control_failed":
        expected.add("own_ids")
    elif report.get("ids_only") != "not_executed_binding_absent":
        raise ValueError("Incomplete optional route control")
    if set(report.get("arms", {})) != expected:
        raise ValueError("Incomplete registered route arms")
    for arm, windows in report["arms"].items():
        if set(windows) != {"window1", "window5"}:
            raise ValueError("Incomplete registered route windows")
        for record in windows.values():
            route = record.get("routes", {})
            if (
                record.get("shape") != [1024, 128000]
                or record.get("dtype") != "float16"
                or record.get("bytes", 0) < 1024 * 128000 * 2
                or not isinstance(record.get("sha256"), str)
                or len(record["sha256"]) != 64
                or route.get("layers") != 38
                or route.get("tokens_per_layer") != 1024
                or route.get("top_k") != 4
                or route.get("bytes", 0) <= 0
                or arm == "own_routes"
                and record.get("replay_bitexact") is not True
            ):
                raise ValueError("Incomplete saved route/logit evidence")
            if arm == "own_ids" and report["ids_only"] == "approved":
                control = record.get("normal_comparison", {})
                kl = control.get("kl", float("nan"))
                if (
                    not math.isfinite(kl)
                    or not 0 <= kl <= 1e-6
                    or control.get("tokens") != 1024
                    or control.get("greedy_matches", 0) < 1023
                ):
                    raise ValueError("Optional IDs-only control was not approved")
    return report


def cache_snapshot() -> dict:
    exists = TUNE_CACHE.is_file()
    data = TUNE_CACHE.read_bytes() if exists else b""
    return {
        "path": str(TUNE_CACHE),
        "exists": exists,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest() if exists else None,
        "prefix_sha256": hashlib.sha256(data[:CACHE_BYTES]).hexdigest() if exists else None,
    }


def execute(plan: dict) -> dict:
    if DATA.exists():
        raise FileExistsError("Preserve earlier diagnostic37 recovery attempts")
    plan_file = DATA.with_suffix(".input.json")
    if plan_file.exists():
        raise FileExistsError("Preserve earlier diagnostic37 recovery input")
    plan_file.write_text(json.dumps(plan, allow_nan=False) + "\n")
    report, before = {}, None
    try:
        before = cache_snapshot()
        if not before["exists"] or before["bytes"] != CACHE_BYTES:
            raise ValueError("Recovery requires the preregistered nonempty cache size")
        if before["sha256"] != CACHE_SHA256:
            raise ValueError("Recovery requires the preregistered cache SHA256")
        base.run_logged(
            [
                sys.executable,
                "-u",
                "/opt/mlxl3/diagnose_lfm25_routes.py",
                str(plan_file),
                str(DATA),
            ],
            DATA.with_suffix(".log"),
            540,
        )
        report = json.loads((DATA / "report.json").read_text())
        validate_result(report)
        after = cache_snapshot()
        if not after["exists"] or after["prefix_sha256"] != CACHE_SHA256:
            raise ValueError("Recovery changed or removed the preregistered cache prefix")
        return report
    except BaseException as error:
        DATA.mkdir(exist_ok=True)
        path = DATA / "report.json"
        report = {}
        if path.exists():
            try:
                child = json.loads(path.read_text())
                if not isinstance(child, dict):
                    raise TypeError("Child report is not an object")
                report = child
            except (ValueError, TypeError, UnicodeError):
                path.rename(DATA / "report-first-invalid.json")
        report.update(
            status="failed",
            launcher_error=repr(error),
            causal_interpretation=False,
            publication=False,
        )
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    finally:
        if DATA.exists():
            after = cache_snapshot()
            report["recovery_cache"] = {
                "expected_bytes": CACHE_BYTES,
                "expected_sha256": CACHE_SHA256,
                "before": before,
                "after": after,
                "prefix_preserved": after["exists"] and after["prefix_sha256"] == CACHE_SHA256,
                "bytes_unchanged": before is not None and before["bytes"] == after["bytes"],
                "sha256_unchanged": before is not None and before["sha256"] == after["sha256"],
            }
            report.update(causal_interpretation=False, publication=False)
            (DATA / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        volume.commit()


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=660,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": volume},
)
def diagnose(plan: dict) -> dict:
    return execute(plan)


@app.local_entrypoint()
def main():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / "build/diagnostic23-input.json").read_text())
    result = diagnose.remote(plan)
    output = root / "build/lfm25-tuning37-routes-recovery-cloud.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)
