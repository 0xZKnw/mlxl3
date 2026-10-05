"""Alternate two native binaries, warm each prompt and check token/text parity.

No cache reuse, speculative decoding or external tools. Run one campaign at a
time on the physical GPU, with no concurrent builds or other inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_json_process import JsonProcess, positive_seconds


class ParityError(RuntimeError):
    """A completed generation differs from the campaign reference."""


def record_failure(report: dict, current: dict, error: BaseException) -> None:
    """Preserve failure state without certifying an incomplete campaign.

    post[report, current, error]: report['status'] == 'failed' and report['parity'] is not True
    post: current['status'] == 'failed' and report['error'] == current['error']
    """
    report["status"] = current["status"] = "failed"
    report["parity"] = (
        False if isinstance(error, ParityError) or report.get("parity") is False else None
    )
    current["error"] = {"type": type(error).__name__, "message": str(error)}
    report["error"] = current["error"]


def save_report(output, report):
    snapshots = {"results.json": report}
    for current in report["passes"]:
        snapshots[f"{current['index']}-{current['label']}.json"] = current
    for name, value in snapshots.items():
        path = output / name
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)


def event(process, deadline):
    value = process.receive(deadline)
    if not isinstance(value.get("type"), str) or not value.get("type"):
        raise RuntimeError("native bridge response has no event type")
    if value.get("type") == "error":
        raise RuntimeError(value["message"])
    return value


def generate(process, prompt, tokens, request_id, timeout):
    deadline = time.monotonic() + timeout
    process.send(
        {
            "type": "generate",
            "request_id": request_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": 0.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
            "reuse_prompt_cache": False,
            "mcp_enabled": False,
            "mtp": False,
            "dflash2": False,
        },
        deadline,
    )
    text = []
    while True:
        value = event(process, deadline)
        if value.get("request_id") != request_id:
            continue
        if value["type"] == "delta":
            text.append(value.get("text", ""))
        if value["type"] == "cancelled":
            raise RuntimeError(f"native generation cancelled ({request_id})")
        if value["type"] == "complete":
            stats = value["stats"]
            if value.get("token_hash") is None:
                raise RuntimeError("native bridge did not provide a token hash")
            if stats["cached_prompt_tokens"] != 0:
                raise RuntimeError("benchmark unexpectedly reused a prompt cache")
            return {
                "stats": stats,
                "token_hash": value.get("token_hash"),
                "text_sha256": hashlib.sha256("".join(text).encode()).hexdigest(),
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--order", choices=["ABBA", "BAAB"], default="ABBA")
    parser.add_argument("--settle-seconds", type=float, default=0.0)
    parser.add_argument("--load-timeout", type=positive_seconds, default=600.0)
    parser.add_argument("--request-timeout", type=positive_seconds, default=600.0)
    parser.add_argument("--baseline-env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--candidate-env", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    if (
        args.tokens <= 1
        or args.repeats <= 0
        or not math.isfinite(args.settle_seconds)
        or args.settle_seconds < 0
    ):
        parser.error("tokens > 1, repeats > 0 and settle-seconds >= 0 are required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty; preserve previous campaigns")
    environments = {}
    for label, overrides in [("A", args.baseline_env), ("B", args.candidate_env)]:
        if any("=" not in entry or not entry.startswith("MLXL3_") for entry in overrides):
            parser.error("environment overrides must have the form MLXL3_NAME=VALUE")
        environments[label] = dict(entry.split("=", 1) for entry in overrides)
    prompts = {"short": "Explique en français, en trois points concis, pourquoi le ciel est bleu."}
    if args.prompt_file:
        prompts["document"] = args.prompt_file.read_text()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            "order": args.order,
            "tokens": args.tokens,
            "repeats": args.repeats,
            "warmup_per_prompt": 1,
            "settle_seconds": args.settle_seconds,
            "load_timeout": args.load_timeout,
            "request_timeout": args.request_timeout,
            "context": 4096,
            "cache": False,
        },
        "binaries": {
            label: {
                "path": str(binary.resolve()),
                "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "environment": environments[label],
            }
            for label, binary in [("A", args.baseline), ("B", args.candidate)]
        },
        "passes": [],
        "status": "running",
        "parity": None,
    }
    save_report(args.output, report)
    expected = {}
    current = {}
    try:
        for pass_index, label in enumerate(args.order):
            current = {
                "label": label,
                "index": pass_index,
                "status": "running",
                "conditions": {"recorded_at_unix": time.time()},
                "prompts": {},
            }
            report["passes"].append(current)
            save_report(args.output, report)
            time.sleep(args.settle_seconds)
            binary = args.baseline if label == "A" else args.candidate
            for name, command in [
                ("power", ["pmset", "-g", "batt"]),
                ("thermal", ["pmset", "-g", "therm"]),
                ("swap", ["sysctl", "vm.swapusage"]),
            ]:
                try:
                    probe = subprocess.run(
                        command, capture_output=True, text=True, check=False, timeout=5
                    )
                    current["conditions"][name] = {
                        "returncode": probe.returncode,
                        "stdout": probe.stdout,
                        "stderr": probe.stderr,
                    }
                except (OSError, subprocess.TimeoutExpired) as error:
                    current["conditions"][name] = {
                        "returncode": None,
                        "stdout": "",
                        "stderr": str(error),
                    }
            stderr_path = args.output / f"{pass_index}-{label}.stderr"
            with stderr_path.open("w") as stderr:
                started = time.perf_counter()
                deadline = time.monotonic() + args.load_timeout
                with JsonProcess(
                    [
                        str(binary.resolve()),
                        "bridge",
                        str(args.model.resolve()),
                        "--context-length",
                        "4096",
                    ],
                    stderr=stderr,
                    env={**os.environ, **environments[label]},
                ) as process:
                    while True:
                        ready = event(process, deadline)
                        if ready["type"] == "ready":
                            current["ready"] = ready
                            break
                    current["ready_seconds"] = time.perf_counter() - started
                    save_report(args.output, report)
                    for name, prompt in prompts.items():
                        measured = {"runs": []}
                        current["prompts"][name] = measured
                        for index in range(args.repeats + 1):
                            request_id = f"run-{name}-{index - 1}" if index else f"warm-{name}"
                            run = generate(
                                process, prompt, args.tokens, request_id, args.request_timeout
                            )
                            if index:
                                measured["runs"].append(run)
                            else:
                                measured["warmup"] = run
                            save_report(args.output, report)
                            actual = (
                                run["token_hash"],
                                run["text_sha256"],
                                run["stats"]["generated_tokens"],
                            )
                            expected.setdefault(name, actual)
                            if actual != expected[name]:
                                report["parity"] = False
                                raise ParityError(f"token/text divergence in {name} pass {label}")
                        measured["medians"] = {
                            metric: statistics.median(
                                run["stats"][metric] for run in measured["runs"]
                            )
                            for metric in [
                                "decode_tps",
                                "decode_seconds",
                                "prefill_tps",
                                "prefill_seconds",
                                "ttft_seconds",
                                "elapsed_seconds",
                            ]
                        }
                        print(
                            json.dumps(
                                {
                                    "pass": pass_index,
                                    "label": label,
                                    "prompt": name,
                                    **measured["medians"],
                                }
                            ),
                            flush=True,
                        )
            current["status"] = "complete"
            save_report(args.output, report)
        summary = {}
        for name in prompts:
            summary[name] = {}
            for metric in ["decode_tps", "prefill_tps", "ttft_seconds", "elapsed_seconds"]:
                medians = {
                    label: statistics.median(
                        p["prompts"][name]["medians"][metric]
                        for p in report["passes"]
                        if p["label"] == label
                    )
                    for label in "AB"
                }
                summary[name][metric] = {
                    **medians,
                    "change_percent": (medians["B"] / medians["A"] - 1) * 100,
                }
        report.update(summary=summary, status="complete", parity=True)
    except BaseException as error:
        record_failure(report, current, error)
        raise
    finally:
        save_report(args.output, report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
