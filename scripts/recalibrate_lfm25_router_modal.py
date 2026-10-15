"""Run the registered router38 recalibration, with bounded spend and private artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import modal
import quantize_lfm25_modal as base
from diagnose_lfm25_routes import V2_SHARDS
from lfm25_router_plan import ALPHAS, child_timeout, validate_plan

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/data/router38")
SOURCE = Path("/data/hf/hub/models--LiquidAI--LFM2.5-24B-A2B/snapshots") / base.REVISION
V2 = Path("/data/v2/output")
CACHE_SOURCE = Path("/data/diagnostic37-fixed-tune-trace/cache-a.bin")
CACHE = Path("/tmp/mlxl3-router38-cache.bin")
CACHE_SHA256 = "1cd9c092bd749dc10fdc017a936477ed9ef0d6072d4af401711d5a7ce5bb0f65"
FIT = Path("/tmp/mlxl3-router38-fit")
VOLUME = "mlxl3-lfm25-v3-20261015"
volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
SCRIPTS = (
    "lfm25_router_plan",
    "capture_lfm25_router",
    "recalibrate_lfm25_router",
    "evaluate_lfm25_router",
    "diagnose_lfm25_reference",
    "diagnose_lfm25_routes",
    "diagnose_lfm25_tuning",
    "generate_lfm25_trace",
    "quantize_lfm25_modal",
    "native_json_process",
)


def build_preflight():
    import importlib
    import importlib.metadata

    for name in SCRIPTS:
        importlib.import_module(name)
    for name, expected in {
        "modal": "1.5.5",
        "torch": "2.10.0",
        "transformers": "5.16.1",
        "accelerate": "1.12.0",
        "exllamav3": "1.5.4",
    }.items():
        if importlib.metadata.version(name).split("+")[0] != expected:
            raise ValueError("Unexpected image distribution: " + name)
    importlib.import_module("eval_lfm25_quality")
    print("router38 CPU image preflight passed; no model or GPU loaded", flush=True)


image = base.image.pip_install("modal==1.5.5", "transformers==5.16.1", "accelerate==1.12.0").env(
    {"PYTHONPATH": "/opt/mlxl3", "EXLLAMAV3_TUNE_CACHE": str(CACHE)}
)
if modal.is_local():
    for name in SCRIPTS:
        image = image.add_local_file(
            ROOT / "scripts" / (name + ".py"), "/opt/mlxl3/" + name + ".py", copy=True
        )
    image = image.add_local_file(
        ROOT / "benchmarks/eval_lfm25_quality.py", "/opt/mlxl3/eval_lfm25_quality.py", copy=True
    )
    image = image.run_function(build_preflight)
app = modal.App("mlxl3-lfm25-router38", image=image)


def atomic_json(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def complete_report(folder: Path) -> dict:
    report = json.loads((folder / "report.json").read_text())
    if not isinstance(report, dict) or report.get("status") != "complete":
        raise ValueError("Child did not complete: " + str(folder))
    return report


def final_guard(baseline: dict, candidate: dict) -> bool:
    if baseline.get("status") != "complete" or candidate.get("status") != "complete":
        raise ValueError("Incomplete final quality reports")
    a, b = baseline["cases"], candidate["cases"]
    if set(a) != {"chat-holdout", "cjk-holdout"} or set(a) != set(b):
        raise ValueError("Missing final quality cases")
    import math

    for name in a:
        x, y = a[name], b[name]
        values = [x[k] for k in ("kl", "ppl", "teacher_ppl")] + [y[k] for k in ("kl", "ppl")]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Invalid final quality values")
        if (
            x["tokens"] != 1024
            or y["tokens"] != 1024
            or x["labels_u32le_sha256"] != y["labels_u32le_sha256"]
            or x["teacher"]["sha256"] != y["teacher"]["sha256"]
        ):
            raise ValueError("Final comparison changed its reference or labels")
    return all(
        b[n]["kl"] <= 0.99 * a[n]["kl"]
        and b[n]["ppl"] <= 1.02 * max(a[n]["ppl"], a[n]["teacher_ppl"])
        for n in a
    )


def execute(plan: dict) -> dict:
    validate_plan(plan)
    if DATA.exists() or FIT.exists() or CACHE.exists():
        raise FileExistsError("Preserve previous router38 attempts")
    cache_bytes = CACHE_SOURCE.read_bytes()
    if len(cache_bytes) != 176 or hashlib.sha256(cache_bytes).hexdigest() != CACHE_SHA256:
        raise ValueError("The fixed diagnostic37 tune cache changed")
    DATA.mkdir()
    CACHE.write_bytes(cache_bytes)
    os.environ["EXLLAMAV3_TUNE_CACHE"] = str(CACHE)
    plan_path = DATA / "input.json"
    atomic_json(plan_path, plan)
    started = time.monotonic()
    report = {
        "status": "running",
        "publication": False,
        "selected": None,
        "final_quality_passed": False,
        "stages": {},
        "cache_before_sha256": CACHE_SHA256,
    }

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(DATA / "report.json", report)

    def child(name, command, limit, folder):
        report["stage"] = name
        save()
        timeout = child_timeout(int(5300 - (time.monotonic() - started)), limit)
        print(json.dumps({"stage": name, "timeout_seconds": timeout}), flush=True)
        base.run_logged([sys.executable, "-u", *command], DATA / (name + ".log"), timeout)
        if CACHE.read_bytes() != cache_bytes:
            raise ValueError("Fixed tune cache changed during child execution")
        result = complete_report(folder)
        if name != "fit":
            expected = (
                {f"layer-{i}.safetensors" for i in range(2, 40)}
                | {"routing-parameters.safetensors"}
                if name in ("teacher", "quant-train")
                else set()
            )
            if name == "teacher" or name.startswith("validation-"):
                expected |= {f"validation-{i:02d}.npy" for i in range(12)}
            if name == "teacher" or name.startswith("final-"):
                expected |= {"chat-holdout.npy", "cjk-holdout.npy"}
            files = result.get("files", {})
            if set(files) != expected or any(
                row.get("bytes", 0) <= 0 or len(row.get("sha256", "")) != 64
                for row in files.values()
            ):
                raise ValueError("Child evidence does not cover the registered files")
            if (
                name in ("teacher", "quant-train")
                and result.get("passive_hook_control", {}).get("bitexact") is not True
            ):
                raise ValueError("Passive capture changed actual model logits")
        elif set(result.get("candidates", {})) != set(ALPHAS):
            raise ValueError("Missing registered candidate exports")
        report["stages"][name] = (
            {
                "status": "complete",
                "report": str(folder / "report.json"),
                "layers_fit": len(result.get("fits", {})),
                "candidates": {
                    k: {
                        field: v[field]
                        for field in ("checkpointdir", "gatefile", "gatefile_sha256", "shards")
                        if field in v
                    }
                    for k, v in result["candidates"].items()
                },
            }
            if name == "fit"
            else result
        )
        save()
        return result

    def quant(name, model, phase, limit, expected_shards=None):
        folder = DATA / name
        captured = child(
            name,
            [
                "/opt/mlxl3/capture_lfm25_router.py",
                "quantized",
                str(plan_path),
                str(folder),
                "--model",
                str(model),
                "--phase",
                phase,
            ],
            limit,
            folder,
        )
        expected = V2_SHARDS if expected_shards is None else expected_shards
        actual = {
            key: captured.get("model_files", {}).get(key, {}).get("sha256") for key in expected
        }
        if actual != expected:
            raise ValueError("Reloaded checkpoint differs from the verified exported weights")
        return folder

    save()
    try:
        teacher = DATA / "teacher"
        child(
            "teacher",
            [
                "/opt/mlxl3/capture_lfm25_router.py",
                "teacher",
                str(plan_path),
                str(teacher),
                "--source",
                str(SOURCE),
            ],
            2300,
            teacher,
        )
        train = quant("quant-train", V2, "train", 900)
        fit = child(
            "fit",
            ["/opt/mlxl3/recalibrate_lfm25_router.py", str(V2), str(teacher), str(train), str(FIT)],
            500,
            FIT,
        )
        from evaluate_lfm25_router import run_score, select_candidate

        validation_start = time.monotonic()
        baseline_folder = quant("validation-baseline", V2, "validation", 450)
        baseline = run_score(plan, teacher, baseline_folder, phase="validation")
        atomic_json(DATA / "validation-baseline-score.json", baseline)
        candidates = {}
        for name, alpha in ALPHAS.items():
            remaining = int(1400 - (time.monotonic() - validation_start))
            folder = quant(
                "validation-" + name,
                Path(fit["candidates"][name]["checkpointdir"]),
                "validation",
                child_timeout(remaining, 450),
                {k: v["sha256"] for k, v in fit["candidates"][name]["shards"].items()},
            )
            candidates[alpha] = run_score(plan, teacher, folder, phase="validation")
            atomic_json(DATA / ("validation-" + name + "-score.json"), candidates[alpha])
        selection = select_candidate(baseline, candidates, plan["validation"]["domains"])
        report["selection"] = selection
        save()
        selected_alpha = selection.get("selected")
        if selected_alpha is None:
            report.update(status="complete", stage="rejected_validation")
            return report
        name = next(k for k, v in ALPHAS.items() if v == selected_alpha)
        report["selected"] = name
        final_baseline = quant("final-baseline", V2, "final", 200)
        final_candidate = quant(
            "final-" + name,
            Path(fit["candidates"][name]["checkpointdir"]),
            "final",
            200,
            {k: v["sha256"] for k, v in fit["candidates"][name]["shards"].items()},
        )
        scores = {
            kind: run_score(plan, teacher, path, phase="final")
            for kind, path in [("baseline", final_baseline), ("candidate", final_candidate)]
        }
        atomic_json(DATA / "final-score.json", scores)
        report["final_quality_passed"] = final_guard(scores["baseline"], scores["candidate"])
        report["final"] = scores
        report.update(status="complete", stage="awaiting_native_verification")
        return report
    except BaseException as error:
        report.update(status="failed", error=repr(error), publication=False)
        raise
    finally:
        try:
            # Small gate files suffice to reproduce each checkpoint from immutable V2.
            if FIT.exists():
                artifacts = DATA / "fit-artifacts"
                artifacts.mkdir()
                for path in FIT.iterdir():
                    if path.is_file() and path.suffix in (".json", ".safetensors"):
                        shutil.copyfile(path, artifacts / path.name)
            after = CACHE.read_bytes()
            report["cache_after_sha256"] = hashlib.sha256(after).hexdigest()
            report["cache_prefix_preserved"] = after[:176] == cache_bytes
            report["cache_bytes_unchanged"] = after == cache_bytes
            shutil.copyfile(CACHE, DATA / "tune-cache.bin")
            if after != cache_bytes:
                report.update(
                    status="failed",
                    error="Fixed tune cache bytes changed",
                    selected=None,
                    final_quality_passed=False,
                )
                raise ValueError("Fixed tune cache bytes changed")
        except BaseException as error:
            report.update(
                status="failed",
                cleanup_error=repr(error),
                publication=False,
                selected=None,
                final_quality_passed=False,
            )
            raise
        finally:
            save()
            volume.commit()


@app.function(
    gpu="H100!",
    cpu=8,
    memory=98304,
    timeout=5500,
    retries=0,
    max_containers=1,
    scaledown_window=2,
    volumes={"/data": volume},
)
def recalibrate(plan: dict) -> dict:
    return execute(plan)


@app.local_entrypoint()
def main():
    plan_path = ROOT / "docs/measurements/quant-lfm25-router38-20261015/input-plan.json"
    plan = validate_plan(json.loads(plan_path.read_text()))
    result = recalibrate.remote(plan)
    output = ROOT / "build/lfm25-router38-cloud.json"
    atomic_json(output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "stage", "selected", "final_quality_passed", "elapsed_seconds")
            },
            allow_nan=False,
        ),
        flush=True,
    )
