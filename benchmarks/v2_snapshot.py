"""Measure one local model with the production bridge; save every completed run."""

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "benchmarks"))
from compare_native import event, generate
from native_json_process import JsonProcess

binary, model, prompt_path, output = map(Path, sys.argv[1:5])
prompt = prompt_path.read_text()
report = {
    "status": "running",
    "engine_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
    "settings": {
        "context": 4096,
        "tokens": 128,
        "warmups": 1,
        "repeats": 3,
        "temperature": 0,
        "top_k": 0,
        "repetition_penalty": 1,
        "mtp": False,
        "dflash": False,
        "reuse_prompt_cache": False,
    },
    "conditions": {},
    "runs": [],
}


def save():
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)


for name, command in [
    ("power", ["pmset", "-g", "batt"]),
    ("thermal", ["pmset", "-g", "therm"]),
    ("swap", ["sysctl", "vm.swapusage"]),
]:
    try:
        probe = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        output_text = "\n".join(probe.stdout.splitlines()[:1]) if name == "power" else probe.stdout
        report["conditions"][name] = {"returncode": probe.returncode, "output": output_text}
    except (OSError, subprocess.TimeoutExpired) as error:
        report["conditions"][name] = {"error": str(error)}
save()
try:
    with (
        output.with_suffix(".stderr").open("w") as stderr,
        JsonProcess(
            [str(binary), "bridge", str(model), "--context-length", "4096"],
            stderr=stderr,
            env=os.environ.copy(),
        ) as process,
    ):
        deadline = time.monotonic() + 900
        while True:
            ready = event(process, deadline)
            if ready["type"] == "ready":
                report["ready"] = {
                    key: value
                    for key, value in ready.items()
                    if key
                    in (
                        "runtime_version",
                        "runtime_commit",
                        "runtime_profile",
                        "mlx_version",
                        "images_supported",
                        "model_context_limit",
                        "modules",
                        "load_seconds",
                        "resident_gb",
                    )
                }
                break
        for index in range(4):
            run = generate(process, prompt, 128, f"v2-{index}", 900)
            memory = run["stats"].get("memory") or {}
            memory.pop("process_id", None)
            memory.pop("process_start_abstime", None)
            report["runs"].append(run)
            save()
            if run["stats"]["generated_tokens"] != 128:
                raise RuntimeError("Output ended before the registered 128-token budget")
            if index and (run["token_hash"], run["text_sha256"]) != (
                report["runs"][0]["token_hash"],
                report["runs"][0]["text_sha256"],
            ):
                raise RuntimeError("Greedy output diverged from warmup")
        process.send({"type": "shutdown"}, time.monotonic() + 10)
        process.process.wait(timeout=10)
        if process.process.returncode != 0:
            raise RuntimeError(f"Bridge exited with {process.process.returncode}")
    metrics = ("decode_tps", "prefill_tps", "ttft_seconds", "elapsed_seconds")
    report["summary"] = {
        key: {"median": statistics.median(values), "min": min(values), "max": max(values)}
        for key in metrics
        if (values := [run["stats"][key] for run in report["runs"][1:]])
    }
    report["status"] = "complete"
except BaseException as error:
    report["status"] = "failed"
    report["error"] = {"type": type(error).__name__, "message": str(error)}
    raise
finally:
    save()
