"""Alternate two native binaries, warm each prompt and check token/text parity.

No cache reuse, speculative decoding or external tools. Run one campaign at a
time on the physical GPU, with no concurrent builds or other inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def event(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"native bridge exited ({process.poll()})")
    value = json.loads(line)
    if value.get("type") == "error":
        raise RuntimeError(value["message"])
    return value


def generate(process, prompt, tokens, request_id):
    process.stdin.write(
        json.dumps(
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
            }
        )
        + "\n"
    )
    process.stdin.flush()
    text = []
    while True:
        value = event(process)
        if value.get("request_id") != request_id:
            continue
        if value["type"] == "delta":
            text.append(value.get("text", ""))
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
    parser.add_argument("--baseline-env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--candidate-env", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()
    if args.tokens <= 1 or args.repeats <= 0 or args.settle_seconds < 0:
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
        "parity": True,
    }
    expected = {}
    for pass_index, label in enumerate(args.order):
        time.sleep(args.settle_seconds)
        binary = args.baseline if label == "A" else args.candidate
        conditions = {"recorded_at_unix": time.time()}
        for name, command in [
            ("power", ["pmset", "-g", "batt"]),
            ("thermal", ["pmset", "-g", "therm"]),
            ("swap", ["sysctl", "vm.swapusage"]),
        ]:
            probe = subprocess.run(command, capture_output=True, text=True, check=False)
            conditions[name] = {
                "returncode": probe.returncode,
                "stdout": probe.stdout,
                "stderr": probe.stderr,
            }
        current = {"label": label, "index": pass_index, "conditions": conditions, "prompts": {}}
        stderr_path = args.output / f"{pass_index}-{label}.stderr"
        with stderr_path.open("w") as stderr:
            started = time.perf_counter()
            process = subprocess.Popen(
                [
                    str(binary.resolve()),
                    "bridge",
                    str(args.model.resolve()),
                    "--context-length",
                    "4096",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                bufsize=1,
                env={**os.environ, **environments[label]},
            )
            try:
                while True:
                    ready = event(process)
                    if ready["type"] == "ready":
                        current["ready"] = ready
                        break
                current["ready_seconds"] = time.perf_counter() - started
                for name, prompt in prompts.items():
                    warmup = generate(process, prompt, args.tokens, f"warm-{name}")
                    runs = [
                        generate(process, prompt, args.tokens, f"run-{name}-{i}")
                        for i in range(args.repeats)
                    ]
                    key = (
                        warmup["token_hash"],
                        warmup["text_sha256"],
                        warmup["stats"]["generated_tokens"],
                    )
                    expected.setdefault(name, key)
                    for run in [warmup, *runs]:
                        actual = (
                            run["token_hash"],
                            run["text_sha256"],
                            run["stats"]["generated_tokens"],
                        )
                        if actual != expected[name]:
                            raise RuntimeError(f"token/text divergence in {name} pass {label}")
                    medians = {
                        metric: statistics.median(run["stats"][metric] for run in runs)
                        for metric in [
                            "decode_tps",
                            "decode_seconds",
                            "prefill_tps",
                            "prefill_seconds",
                            "ttft_seconds",
                            "elapsed_seconds",
                        ]
                    }
                    current["prompts"][name] = {"warmup": warmup, "runs": runs, "medians": medians}
                    print(
                        json.dumps({"pass": pass_index, "label": label, "prompt": name, **medians}),
                        flush=True,
                    )
            finally:
                if process.poll() is None:
                    process.stdin.write('{"type":"shutdown"}\n')
                    process.stdin.flush()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        report["passes"].append(current)
        (args.output / f"{pass_index}-{label}.json").write_text(
            json.dumps(current, indent=2) + "\n"
        )
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
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
    report["summary"] = summary
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
