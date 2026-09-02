"""Measure MLXL3 bridge overhead independently from the native UI."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


def read_event(process: subprocess.Popen[str]) -> dict:
    line = process.stdout.readline()
    if not line:
        stderr = process.stderr.read()
        raise RuntimeError(f"bridge exited before completing an event: {stderr}")
    return json.loads(line)


def wait_for(process: subprocess.Popen[str], event_type: str) -> dict:
    while True:
        event = read_event(process)
        if event["type"] == "error":
            raise RuntimeError(event["message"])
        if event["type"] == event_type:
            return event


def generate(
    process: subprocess.Popen[str],
    prompt: str,
    max_tokens: int,
    request_number: int,
) -> dict:
    request_id = f"bench-{request_number}"
    process.stdin.write(
        json.dumps(
            {
                "type": "generate",
                "request_id": request_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
            }
        )
        + "\n"
    )
    process.stdin.flush()

    started = time.perf_counter()
    first_delta_at = None
    delta_events = 0
    delta_bytes = 0
    while True:
        event = read_event(process)
        if event.get("request_id") != request_id:
            continue
        if event["type"] == "delta":
            if first_delta_at is None:
                first_delta_at = time.perf_counter()
            delta_events += 1
            delta_bytes += len(event.get("text", "").encode("utf-8"))
        elif event["type"] == "error":
            raise RuntimeError(event["message"])
        elif event["type"] == "complete":
            completed_at = time.perf_counter()
            stats = event["stats"]
            client_decode_seconds = completed_at - (first_delta_at or started)
            client_decode_tps = (
                max(0, stats["generated_tokens"] - 1) / client_decode_seconds
                if client_decode_seconds > 0
                else 0.0
            )
            return {
                **stats,
                "client_decode_tps": client_decode_tps,
                "delta_events": delta_events,
                "delta_bytes": delta_bytes,
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--prompt",
        default="Explique en français, en trois points concis, pourquoi le ciel est bleu.",
    )
    args = parser.parse_args()

    process = subprocess.Popen(
        [sys.executable, "-m", "mlxl3", "bridge", str(args.model)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        wait_for(process, "ready")
        generate(process, "Dis simplement bonjour.", 8, -1)
        runs = [
            generate(process, args.prompt, args.max_tokens, repeat)
            for repeat in range(args.repeats)
        ]
        print(
            json.dumps(
                {
                    "repeats": args.repeats,
                    "median_engine_decode_tps": statistics.median(
                        run["decode_tps"] for run in runs
                    ),
                    "median_client_decode_tps": statistics.median(
                        run["client_decode_tps"] for run in runs
                    ),
                    "median_delta_events": statistics.median(
                        run["delta_events"] for run in runs
                    ),
                    "runs": runs,
                },
                indent=2,
            )
        )
    finally:
        if process.poll() is None:
            process.stdin.write('{"type":"shutdown"}\n')
            process.stdin.flush()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()
