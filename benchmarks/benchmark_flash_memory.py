"""Diagnostic RAM/swap campaign; timings include sampling and are not speed claims."""

import argparse
import hashlib
import json
import math
import os
import re
import runpy
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_json_process import JsonProcess

flash = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/benchmark-flash-bridge.py")
)

VM_KEYS = (
    "Pages free",
    "File-backed pages",
    "Pages occupied by compressor",
    "Swapins",
    "Swapouts",
)


def parse_vm(raw: str) -> dict[str, int]:
    """Parse required counters strictly, in bytes rather than pages.

    raises: ValueError
    post: all(value >= 0 for value in __return__.values())
    """
    page = re.search(r"page size of ([0-9]+) bytes", raw)
    if page is None or int(page[1]) <= 0:
        raise ValueError("missing VM page size")
    counters = {}
    for key in VM_KEYS:
        matches = re.findall(rf"^{re.escape(key)}:\s+([0-9]+)\.\s*$", raw, re.MULTILINE)
        if len(matches) != 1:
            raise ValueError(f"missing or duplicate VM counter: {key}")
        counters[key] = int(matches[0]) * int(page[1])
    return counters


def snapshot():
    raw = subprocess.check_output(["vm_stat"], text=True, timeout=5)
    return {"time": time.time(), "bytes": parse_vm(raw), "raw": raw}


def run(engine, model, output, *, cache_gib=None, tokens=64, timeout=300, sample=snapshot):
    if tokens < 2 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("tokens >= 2 and finite positive timeout required")
    if cache_gib is not None and (not math.isfinite(cache_gib) or not 0.6 <= cache_gib <= 12):
        raise ValueError("invalid expert cache budget")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "parity": False,
        "reaped": False,
        "engine": str(engine.resolve()),
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "cache_gib": cache_gib,
        "tokens": tokens,
        "runs": [],
        "samples": [],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    def take(label):
        value = sample()
        if not isinstance(value, dict) or set(value.get("bytes", {})) != set(VM_KEYS):
            raise ValueError("incomplete memory sample")
        if any(type(v) is not int or v < 0 for v in value["bytes"].values()):
            raise ValueError("invalid memory sample")
        report["samples"].append({"label": label, **value})

    child = None
    save()
    try:
        take("before")
        env = {**os.environ, "MLXL3_SSD_OVERLAP": "1"}
        env.pop("MLXL3_SSD_CACHE_GIB", None)
        env.pop("MLXL3_ALLOCATOR_CACHE_MIB", None)
        if cache_gib is not None:
            env["MLXL3_SSD_CACHE_GIB"] = str(cache_gib)
        with (
            tempfile.TemporaryDirectory(prefix="flash-ram-") as directory,
            (output / "stderr.log").open("w") as log,
        ):
            command = [
                str(engine),
                "--registry",
                f"{directory}/registry.json",
                "bridge",
                str(model),
                "--context-length",
                "4096",
            ]
            with JsonProcess(command, stderr=log, env=env) as child:

                def until(kind, request_id=None):
                    deadline = time.monotonic() + timeout
                    mode = None
                    last_sample = 0.0
                    while True:
                        event = child.receive(deadline)
                        if event.get("type") == "error":
                            raise RuntimeError(event)
                        if request_id is not None and event.get("request_id") != request_id:
                            raise ValueError("unexpected response identity")
                        if time.monotonic() - last_sample >= 1:
                            take(f"{request_id or 'loading'}:{event.get('type')}")
                            last_sample = time.monotonic()
                        if event.get("type") == "generation_mode":
                            mode = event
                        if event.get("type") == kind:
                            return event, mode

                report["ready"], _ = until("ready")
                for label, prompt in flash["PROMPTS"].items():
                    for mtp in (False, True):
                        request_id = f"{label}-{mtp}"
                        child.send(
                            {
                                "type": "generate",
                                "request_id": request_id,
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": tokens,
                                "temperature": 0,
                                "top_k": 1,
                                "repetition_penalty": 1,
                                "mtp": mtp,
                                "mtp_depth": 1,
                                "mtp_head_path": str(model),
                                "dflash2": False,
                                "mcp_enabled": False,
                                "reuse_prompt_cache": False,
                            },
                            time.monotonic() + timeout,
                        )
                        event, mode = until("complete", request_id)
                        flash["validate"](event, mode, mtp, tokens)
                        previous = [r for r in report["runs"] if r["prompt"] == label]
                        if previous and any(
                            event[k] != previous[0]["event"][k]
                            for k in ("token_hash", "cache_context")
                        ):
                            raise ValueError("OFF/MTP divergence")
                        ping = f"idle-{request_id}"
                        child.send({"type": "ping", "request_id": ping}, time.monotonic() + timeout)
                        idle, _ = until("pong", ping)
                        for memory in (event["stats"].get("memory"), idle.get("memory")):
                            if not isinstance(memory, dict) or any(
                                type(memory.get(k)) is not int or memory[k] <= 0
                                for k in (
                                    "mlx_active_bytes",
                                    "mlx_peak_bytes",
                                    "process_footprint_bytes",
                                )
                            ):
                                raise ValueError("missing positive allocation/footprint metrics")
                            if (
                                type(memory.get("mlx_cache_bytes")) is not int
                                or memory["mlx_cache_bytes"] < 0
                            ):
                                raise ValueError("invalid allocator cache")
                        report["runs"].append(
                            {"prompt": label, "mtp": mtp, "event": event, "idle": idle}
                        )
                        save()
                        print(
                            json.dumps(
                                {
                                    "case": request_id,
                                    "hash": event["token_hash"],
                                    "idle": idle["memory"],
                                }
                            ),
                            flush=True,
                        )
        report["reaped"] = child.process.poll() is not None
        if child.process.returncode != 0:
            raise RuntimeError("bridge did not exit successfully")
        take("after")
        report.update(status="validated", parity=True)
    except BaseException as error:
        report.update(status="failed", parity=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if child is not None:
            report["reaped"] = child.process.poll() is not None
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-gib", type=float)
    args = parser.parse_args()
    run(args.engine.resolve(), args.model.resolve(), args.output, cache_gib=args.cache_gib)
