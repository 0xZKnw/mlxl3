"""Bounded official BF16 reference capture; no quantization or publication."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal
import quantize_lfm25_modal as base
from diagnose_lfm25_reference import REVISION, validate_windows

DATA = Path("/data/diagnostic35-reference")
SOURCE = Path("/data/hf/hub/models--LiquidAI--LFM2.5-24B-A2B/snapshots") / REVISION
VOLUME = "mlxl3-lfm25-v3-20261015"
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
if modal.is_local():
    import quantize_lfm25_v3_modal as previous

    image = previous.image
    for name in ("diagnose_lfm25_reference", "diagnose_lfm25_reference_modal"):
        image = image.add_local_file(
            Path(__file__).with_name(name + ".py"), "/opt/mlxl3/" + name + ".py", copy=True
        )
else:
    image = base.image
app = modal.App("mlxl3-lfm25-reference35", image=image)


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=900,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": volume},
)
def diagnose(plan: dict) -> str:
    validate_windows(plan)
    if DATA.exists():
        raise FileExistsError("Preserve the existing diagnostic before repeating it")
    plan_path = DATA.with_suffix(".input.json")
    plan_path.write_text(json.dumps(plan, allow_nan=False) + "\n")
    try:
        command = [
            sys.executable,
            "-u",
            "/opt/mlxl3/diagnose_lfm25_reference.py",
            str(SOURCE),
            str(plan_path),
            str(DATA),
        ]
        base.run_logged(command, DATA.with_suffix(".log"), 780)
        report = json.loads((DATA / "report.json").read_text())
        expected = {
            f"{phase}-{name}" for phase in ("prefill", "decode") for name in ("window1", "window5")
        }
        records = report.get("windows", {})
        if (
            report.get("status") != "complete"
            or not isinstance(records, dict)
            or set(records) != expected
            or any(
                not isinstance(row, dict)
                or row.get("shape") != [1024, 128000]
                or row.get("storage_dtype") != "float32"
                or not row.get("logits_dtypes")
                for row in records.values()
            )
        ):
            raise ValueError("Incomplete official reference capture")
        return json.dumps(report)
    except BaseException as error:
        DATA.mkdir(exist_ok=True)
        report_path = DATA / "report.json"
        try:
            report = json.loads(report_path.read_text()) if report_path.exists() else {}
        except (ValueError, OSError):
            report = {}
        report.update(status="failed", publication=False, launcher_error=repr(error))
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    finally:
        volume.commit()


@app.local_entrypoint()
def main(plan_path: str = "build/diagnostic23-input.json"):
    plan = json.loads(Path(plan_path).read_text())
    validate_windows(plan)
    result = diagnose.remote(plan)
    destination = Path("build/lfm25-reference35-cloud.json")
    destination.write_text(result + "\n")
    print(f"Report: {destination}")
