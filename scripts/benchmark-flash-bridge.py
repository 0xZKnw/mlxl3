"""Bounded Flash Next baseline/MTP campaign on two fixed prompts in one bridge."""

import argparse
import hashlib
import json
import math
import os
import queue
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path

PROMPTS = {
    "MAIN": "Explique en français, avec un exemple concret, comment un cache LFU évite les lectures SSD inutiles pendant l’inférence d’un modèle MoE.",
    "HOLD": "Écris une fonction Python merge_intervals qui fusionne les intervalles qui se chevauchent, puis explique sa complexité et donne un exemple.",
}


def validate(event: dict, mode: dict | None, mtp: bool, budget: int) -> None:
    if mode is None or mode.get("mtp_active") is not mtp or mode.get("dflash_active") is not False:
        raise ValueError(f"engine mode mismatch: {mode}")
    stats = event["stats"]
    if stats["generated_tokens"] != budget or stats["decode_tokens"] != budget - 1:
        raise ValueError("incomplete generation (including early EOS)")
    if not isinstance(event.get("token_hash"), str) or not event["token_hash"]:
        raise ValueError("missing token hash")
    if not isinstance(event.get("cache_context"), str) or not event["cache_context"]:
        raise ValueError("empty output")
    for key in ("decode_tps", "decode_seconds", "ttft_seconds", "prefill_tps"):
        value = stats[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"invalid timing {key}: {value}")
    if not math.isclose(stats["decode_tps"], (budget - 1) / stats["decode_seconds"], rel_tol=1e-5):
        raise ValueError("inconsistent decode timing")
    if mtp:
        counters = [
            stats.get(key) for key in ("mtp_proposed_tokens", "mtp_accepted_tokens", "mtp_blocks")
        ]
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("invalid MTP counters")
        proposed, accepted, blocks = counters
        if accepted > proposed or proposed != blocks or proposed + accepted > budget - 1:
            raise ValueError("inconsistent depth-one MTP counters")


def run(
    engine: Path,
    model: Path,
    output: Path,
    mtp=False,
    tokens=64,
    repeats=3,
    timeout=300.0,
    compare=False,
):
    if tokens < 2 or repeats < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("tokens >= 2, repeats >= 1 and finite timeout > 0 required")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "incomplete",
        "mode": "comparison" if compare else "mtp" if mtp else "baseline",
        "engine": str(engine.resolve()),
        "binary_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "model": str(model.resolve()),
        "tokens": tokens,
        "repeats": repeats,
        "prompts": PROMPTS,
        "runs": [],
        "warmup": [],
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    save()
    with (
        tempfile.TemporaryDirectory(prefix="flash-bench-") as directory,
        (output / "stderr.log").open("w") as stderr,
    ):
        env = {**os.environ, "MLXL3_SSD_CACHE_GIB": "8", "MLXL3_SSD_OVERLAP": "1"}
        process = subprocess.Popen(
            [
                str(engine.resolve()),
                "--registry",
                str(Path(directory) / "models.json"),
                "bridge",
                str(model.resolve()),
                "--context-length",
                "4096",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            env=env,
        )
        events = queue.Queue()

        def read():
            try:
                for line in process.stdout:
                    events.put(json.loads(line))
            except (ValueError, OSError) as error:
                events.put({"type": "error", "message": str(error)})
            finally:
                events.put({"type": "error", "message": "bridge exited"})

        reader = threading.Thread(target=read, daemon=True)
        reader.start()

        def until(kind, request_id=None):
            deadline, mode = time.monotonic() + timeout, None
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"bridge deadline: {kind}")
                try:
                    event = events.get(timeout=remaining)
                except queue.Empty as error:
                    raise TimeoutError(f"bridge deadline: {kind}") from error
                if not isinstance(event, dict) or "type" not in event:
                    raise ValueError("invalid bridge event")
                if event["type"] == "error":
                    raise RuntimeError(event)
                if request_id is not None and event.get("request_id") != request_id:
                    continue
                if event["type"] == "generation_mode":
                    mode = event
                if event["type"] == kind:
                    return event, mode

        try:
            report["runtime"], _ = until("ready")
            for repeat in range(repeats + 1):
                modes = (False, True) if compare else (mtp,)
                if compare and repeat % 2:
                    modes = tuple(reversed(modes))
                schedule = [
                    (label, prompt, enabled)
                    for label, prompt in PROMPTS.items()
                    for enabled in modes
                ]
                for label, prompt, enabled in schedule:
                    request_id = f"{repeat}-{label}-{enabled}"
                    request = {
                        "type": "generate",
                        "request_id": request_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": tokens,
                        "temperature": 0,
                        "top_k": 1,
                        "repetition_penalty": 1,
                        "mtp": enabled,
                        "mtp_depth": 1,
                        "mtp_head_path": str(model.resolve()),
                        "dflash2": False,
                        "mcp_enabled": False,
                        "reuse_prompt_cache": False,
                    }
                    process.stdin.write(json.dumps(request) + "\n")
                    process.stdin.flush()
                    event, mode = until("complete", request_id)
                    validate(event, mode, enabled, tokens)
                    entry = {
                        "mode": "mtp" if enabled else "baseline",
                        "prompt": label,
                        "repeat": repeat,
                        "token_hash": event["token_hash"],
                        "text": event["cache_context"],
                        "stats": event["stats"],
                    }
                    previous = [
                        r for r in report["runs"] + report["warmup"] if r["prompt"] == label
                    ]
                    if previous and (
                        previous[0]["token_hash"] != entry["token_hash"]
                        or previous[0]["text"] != entry["text"]
                    ):
                        raise ValueError(f"non deterministic output: {label}")
                    report["warmup" if repeat == 0 else "runs"].append(entry)
                    save()
                    print(
                        json.dumps(
                            {
                                "prompt": label,
                                "mode": entry["mode"],
                                "repeat": repeat,
                                "decode_tps": event["stats"]["decode_tps"],
                                "ttft_seconds": event["stats"]["ttft_seconds"],
                                "token_hash": event["token_hash"],
                            }
                        ),
                        flush=True,
                    )
            process.stdin.write('{"type":"shutdown"}\n')
            process.stdin.flush()
            if process.wait(timeout=10) != 0:
                raise RuntimeError("bridge shutdown failed")
            medians = {
                mode: {
                    label: statistics.median(
                        r["stats"]["decode_tps"]
                        for r in report["runs"]
                        if r["prompt"] == label and r["mode"] == mode
                    )
                    for label in PROMPTS
                }
                for mode in ({r["mode"] for r in report["runs"]})
            }
            report["median_decode_tps"] = (
                medians if compare else medians["mtp" if mtp else "baseline"]
            )
            report["status"] = "validated"
        except BaseException as error:
            report.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            reader.join(timeout=5)
            process.stdin.close()
            process.stdout.close()
            save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mtp", action="store_true")
    modes.add_argument(
        "--compare", action="store_true", help="alternate baseline/MTP on the same binary"
    )
    args = parser.parse_args()
    run(args.engine, args.model, args.output, args.mtp, compare=args.compare)
