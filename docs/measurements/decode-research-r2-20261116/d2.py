#!/usr/bin/env python3
"""DIAG-SAMPLER-COST-20261116 D2: bridge decode rate, greedy vs Desktop default sampling.

Diagnostic only (no parity: sampled text differs from greedy by design).
Every run is saved; a failed or partial campaign is written with an explicit
status and never reported as complete.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GREEDY = {"temperature": 0, "top_k": 1, "repetition_penalty": 1}
APP = {
    "temperature": 0.2,
    "top_k": 80,
    "repetition_penalty": 1.05,
    "top_p": 1.0,
    "presence_penalty": 0.0,
}
# Qwen3.6 model-card presets shipped in the Desktop catalog (SamplingPresets.swift).
Q36_THINKING = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "repetition_penalty": 1, "presence_penalty": 1.5}
Q36_CODING = {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "repetition_penalty": 1, "presence_penalty": 0.0}
ARMS = {"G": GREEDY, "S": APP, "T": Q36_THINKING, "C": Q36_CODING}

CODE = '''def load_users(path):
    users = []
    with open(path) as handle:
        for line in handle:
            name, age = line.strip().split(",")
            users.append({"name": name, "age": int(age)})
    return users
'''
CASES = {
    "fr": "Explique en quelques paragraphes pourquoi le ciel est bleu.",
    "code": "Write a Rust function that parses an ISO 8601 date like 2026-11-16.",
    "chat": "Salut ! Je cherche des idées de repas simples pour la semaine, tu peux m'aider ?",
    "howto": "How do I set up a Python virtual environment and install packages? Explain step by step.",
    "story": "Raconte une courte histoire sur un chat qui découvre la neige pour la première fois.",
    "edit": "Rename the variable `users` to `people` everywhere and return the complete "
    f"code only:\n\n```python\n{CODE}```",
}


class Bridge:
    def __init__(self, scripts, engine, model, context, env, log, timeout):
        sys.path.insert(0, str(scripts))
        from native_json_process import JsonProcess  # noqa: PLC0415

        self.directory = tempfile.TemporaryDirectory(prefix="mlxl3-d2-")
        full_env = dict(os.environ)
        for name in ("MLXL3_LFM3_LOOKUP", "MLXL3_LFM3_CERT_HEAD"):
            full_env.pop(name, None)
        full_env.update(env)
        registry = str(Path(self.directory.name) / "models.json")
        self.process = JsonProcess(
            [engine, "--registry", registry, "bridge", model, "--context-length", str(context)],
            stderr=log,
            env=full_env,
        )
        self.timeout = timeout
        self.count = 0

    def close(self):
        try:
            self.process.close()
        finally:
            self.directory.cleanup()

    def generate(self, prompt, budget, sampling, extra):
        self.count += 1
        request_id = str(self.count)
        deadline = time.monotonic() + self.timeout
        request = {
            "type": "generate",
            "request_id": request_id,
            "conversation_id": f"d2-{request_id}",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": budget,
            "reuse_prompt_cache": False,
            "mcp_enabled": False,
        }
        request.update(sampling)
        request.update(extra)
        self.process.send(request, deadline)
        text = ""
        while True:
            event = self.process.receive(deadline)
            if event.get("type") == "error" and event.get("request_id") in (None, request_id):
                raise RuntimeError(f"bridge error: {event}")
            if event.get("request_id") != request_id:
                continue
            if event["type"] == "delta":
                text += event.get("text", "")
            if event["type"] == "cancelled":
                raise RuntimeError("generation cancelled")
            if event["type"] == "complete":
                stats = event.get("stats") or {}
                generated = stats.get("generated_tokens")
                rate = stats.get("decode_tps")
                if type(generated) is not int or not 1 <= generated <= budget:
                    raise ValueError(f"invalid completion: {event}")
                if not isinstance(rate, (int, float)) or not rate > 0:
                    raise ValueError(f"invalid decode rate: {event}")
                return {
                    "text": text,
                    "cache_context": event.get("cache_context"),
                    "stop_reason": event.get("stop_reason"),
                    "token_hash": event.get("token_hash"),
                    "stats": stats,
                }


def run_process(args, env, pattern, extra, log, names):
    bridge = Bridge(args.scripts, args.engine, args.model, args.context, env, log, args.timeout)
    rows = []
    try:
        for name in names:
            bridge.generate(CASES[name], args.warmup, ARMS[pattern[0]], extra)
        for name in names:
            for arm in pattern:
                time.sleep(args.settle)
                result = bridge.generate(CASES[name], args.tokens, ARMS[arm], extra)
                rows.append({"case": name, "arm": arm, **result})
    finally:
        bridge.close()
    return rows


def summarize(runs):
    table = {}
    for run in runs:
        for row in run["rows"]:
            key = (run["label"], row["case"], row["arm"])
            table.setdefault(key, []).append(row["stats"]["decode_tps"])
    return [
        {
            "label": label,
            "case": case,
            "arm": arm,
            "n": len(values),
            "mean_decode_tps": statistics.fmean(values),
            "min": min(values),
            "max": max(values),
        }
        for (label, case, arm), values in sorted(table.items())
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plan", choices=("lfm3", "q36", "q36-presets"), required=True)
    parser.add_argument("--mtp-head", type=Path)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--cases", default="fr,code,chat,howto,story,edit")
    args = parser.parse_args()
    names = args.cases.split(",")
    unknown = [name for name in names if name not in CASES]
    if unknown:
        parser.error(f"unknown cases: {unknown}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.plan == "lfm3":
        # A: default env (certified head on greedy). B: certified head off.
        plan = [
            ("A", {}, "GSSG", {}),
            ("B", {"MLXL3_LFM3_CERT_HEAD": "0"}, "GSSG", {}),
            ("B", {"MLXL3_LFM3_CERT_HEAD": "0"}, "SGGS", {}),
            ("A", {}, "SGGS", {}),
        ]
    elif args.plan == "q36-presets":
        if args.mtp_head is None:
            parser.error("--mtp-head is required for the q36-presets plan")
        head = str(args.mtp_head.resolve())
        mtp1 = {"mtp": True, "mtp_depth": 1, "mtp_head_path": head}
        mtp3 = {"mtp": True, "mtp_depth": 3, "mtp_head_path": head}
        plan = [
            ("MTP1", {}, "GTCCTG", mtp1),
            ("MTP3", {}, "GTCCTG", mtp3),
        ]
    else:
        if args.mtp_head is None:
            parser.error("--mtp-head is required for the q36 plan")
        mtp = {"mtp": True, "mtp_depth": 3, "mtp_head_path": str(args.mtp_head.resolve())}
        plan = [
            ("AR", {}, "GSSG", {}),
            ("MTP3", {}, "GSSG", mtp),
            ("MTP3", {}, "SGGS", mtp),
            ("AR", {}, "SGGS", {}),
        ]
    report = {
        "status": "running",
        "plan": args.plan,
        "engine": args.engine,
        "model": args.model,
        "tokens": args.tokens,
        "warmup": args.warmup,
        "cases": names,
        "arms": ARMS,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "runs": [],
    }
    path = args.out / "results.json"
    try:
        for index, (label, env, pattern, extra) in enumerate(plan):
            with open(args.out / f"bridge-{index}-{label}.log", "wb") as log:
                rows = run_process(args, env, pattern, extra, log, names)
            report["runs"].append({"label": label, "env": env, "pattern": pattern, "rows": rows})
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        report["status"] = "complete"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        report["summary"] = summarize(report["runs"])
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({"status": report["status"], "runs": len(report["runs"])}))
        for row in report["summary"]:
            print(json.dumps(row))


if __name__ == "__main__":
    main()
