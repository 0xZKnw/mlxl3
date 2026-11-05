"""Sequential native UI/engine comparison. Run from the repository root."""
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


FOLDER = Path(__file__).resolve().parent
ROOT = Path.cwd()


def validate(report, mode, reference):
    if report.get("status") != "complete" or report.get("complete") is not True:
        raise ValueError("Incomplete native report")
    samples = report["samples"]
    if len(samples) != (11 if mode == "ui" else 2):
        raise ValueError("Wrong sample count")
    current = {}
    for sample in samples:
        key = (sample["kind"], sample["bytes"], sample["cadence"]) if mode == "ui" else sample["prompt"]
        if key in current or len(sample["text_sha256"]) != 64 or sample.get("window_visible") is not True:
            raise ValueError("Duplicate sample or missing text hash")
        for metric in ["cpu_seconds", "main_cpu_seconds", "wall_seconds", "footprint_bytes"]:
            value = sample[metric]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid metric: " + metric)
        if mode == "engine":
            stats = sample["stats"]
            if not 64 <= stats["generated_tokens"] <= 128 or not math.isfinite(stats["decode_tps"]) or stats["decode_tps"] <= 0:
                raise ValueError("Invalid generation statistics")
            accepted, proposed = stats["mtp_accepted_tokens"], stats["mtp_proposed_tokens"]
            if not all(type(value) is int for value in [accepted, proposed, stats["generated_tokens"]]) or not 0 <= accepted <= proposed <= 128 or proposed == 0:
                raise ValueError("Missing or invalid MTP1 counters")
            current[key] = (sample["text_sha256"], stats["generated_tokens"], accepted, proposed)
        else:
            if sample["appends"] != 24:
                raise ValueError("Incomplete replay")
            current[key] = sample["text_sha256"]
    if reference is not None and current != reference:
        raise ValueError("Text/count/acceptance divergence")
    return current


def stop_group(process):
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    finally:
        # The parent can exit before a child that ignores SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


def main(mode):
    if mode not in {"ui", "engine"}:
        raise ValueError("Expected ui or engine")
    campaign = {"status": "running", "complete": False, "mode": mode, "passes": []}
    destination = FOLDER / (mode + "-campaign.json")

    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(campaign, indent=2) + "\n")
        temporary.replace(destination)

    save()
    reference = None
    try:
        for index, variant in enumerate("abbabaab"):
            label = f"{mode}-{index}-{variant}"
            output = FOLDER / (label + ".json")
            conditions = {name: subprocess.run(command, capture_output=True, text=True, timeout=10).stdout
                          for name, command in {"battery": ["pmset", "-g", "batt"], "thermal": ["pmset", "-g", "therm"],
                                                "swap": ["sysctl", "vm.swapusage"]}.items()}
            processes = subprocess.check_output(["ps", "-axo", "pid=,comm=,args="], text=True)
            conflicts = [line for line in processes.splitlines()
                         if "swift-frontend" in line or ("/mlxl3" in line and " bridge " in line)]
            if conflicts:
                raise RuntimeError("Concurrent compiler/engine: " + "\n".join(conflicts))
            (FOLDER / (label + "-conditions.json")).write_text(json.dumps(conditions, indent=2) + "\n")
            with tempfile.TemporaryDirectory(prefix="mlxl3-app-perf-") as directory:
                command = [str(ROOT / "build/app-fluidite" / ("performance-" + variant)), "--" + mode, str(output)]
                if mode == "engine":
                    command += [str(ROOT / "target/release/mlxl3-rs"),
                                str(ROOT / "models/Qwen3.6-35B-A3B-EXL3-2.49bpw"),
                                str(ROOT / "models/Qwen3.6-35B-A3B-MTP-4bit")]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           text=True, start_new_session=True, env=os.environ | {"MLXL3_HOME": directory})
                try:
                    stdout, stderr = process.communicate(timeout=180)
                except BaseException:
                    stop_group(process)
                    raise
                (FOLDER / (label + ".stdout")).write_text(stdout)
                (FOLDER / (label + ".stderr")).write_text(stderr)
                if process.returncode:
                    raise RuntimeError(f"{label}: native process exited {process.returncode}")
                report = json.loads(output.read_text())
                reference = validate(report, mode, reference)
                campaign["passes"].append({"label": label, "variant": variant, "samples": len(report["samples"]), "exit_code": 0})
                save()
                print(label, "complete", flush=True)
        campaign.update(status="complete", complete=True)
    except BaseException as error:
        campaign.update(status="failed", error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main(sys.argv[1])
