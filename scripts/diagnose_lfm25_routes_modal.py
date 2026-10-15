"""Run the bounded, read-only route diagnostic on the existing private Volume."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import modal
import quantize_lfm25_modal as base

DATA = Path("/data/diagnostic36-routes")
VOLUME = "mlxl3-lfm25-v3-20261015"
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
image = base.image
if modal.is_local():
    image = image.add_local_python_source("quantize_lfm25_modal").add_local_file(
        Path(__file__).with_name("diagnose_lfm25_routes.py"),
        "/opt/mlxl3/diagnose_lfm25_routes.py",
        copy=True,
    )
app = modal.App("mlxl3-lfm25-routes36", image=image)


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


def execute(plan: dict) -> dict:
    if DATA.exists():
        raise FileExistsError("Preserve earlier diagnostic36 attempts")
    plan_file = DATA.with_suffix(".input.json")
    if plan_file.exists():
        raise FileExistsError("Preserve earlier diagnostic36 input")
    plan_file.write_text(json.dumps(plan, allow_nan=False) + "\n")
    try:
        base.run_logged(
            [
                sys.executable,
                "-u",
                "/opt/mlxl3/diagnose_lfm25_routes.py",
                str(plan_file),
                str(DATA),
            ],
            DATA.with_suffix(".log"),
            900,
        )
        report = json.loads((DATA / "report.json").read_text())
        return validate_result(report)
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
        volume.commit()


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=1020,
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
    output = root / "build/lfm25-routes36-cloud.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result), flush=True)
