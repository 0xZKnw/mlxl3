#!/usr/bin/env python3
"""ABBA timing of one bridge option (environment flags) on the same binary.

Passes OFF/ON/ON/OFF, one fresh bridge process each (one GPU model at a
time), warmup of every selected case at 32 tokens per pass, then `--repeats`
timed generations per case. Cases are those of `check-lfm3-lookup.py`.

Greedy samplers must keep identical token hashes across all passes (parity
guard); sampled ones (bridge clock seed) only need valid completions, and the
report flags them `sampled` so a speed difference is read with that noise.
The report is written on every exit with an explicit status; an error,
timeout or invalid completion never yields `passed`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from native_json_process import JsonProcess, positive_seconds  # noqa: E402

_spec = importlib.util.spec_from_file_location("lfm3_cases", HERE / "check-lfm3-lookup.py")
_cases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cases)

SAMPLERS = {
    "greedy": {"temperature": 0, "top_k": 1, "repetition_penalty": 1},
    "desktop": {"temperature": 0.2, "top_k": 80, "repetition_penalty": 1.05},
    "lfm3": {"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.1},
}
# Every option this script may toggle; inherited values are removed first.
OPTION_NAMES = (
    "MLXL3_LFM3_CERT_HEAD", "MLXL3_LFM3_CERT_SAMPLE", "MLXL3_LFM3_CERT_SCREEN",
    "MLXL3_LFM3_CERT_CHECK", "MLXL3_LFM3_LOOKUP", "MLXL3_LFM3_EMBED_LAZY",
    "MLXL3_TOPK_READBACK", "MLXL3_MTP_SAMPLED_DRAFT",
)
COUNTERS = (
    "lfm3_cert_tokens", "lfm3_cert_fallbacks", "lfm3_cert_check_mismatches",
    "topk_readback_rows", "mtp_proposed_tokens", "mtp_accepted_tokens",
    "mtp_blocks", "mtp_sampled_draft_levels",
)


def parse_env(items):
    env = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or name not in OPTION_NAMES:
            raise SystemExit(f"invalid option {item!r}; allowed: {OPTION_NAMES}")
        env[name] = value
    return env


class Bridge:
    def __init__(self, args, flags, log):
        self.directory = tempfile.TemporaryDirectory(prefix="mlxl3-abba-")
        env = {k: v for k, v in os.environ.items() if k not in OPTION_NAMES}
        env.update(flags)
        registry = str(Path(self.directory.name) / "models.json")
        self.process = JsonProcess(
            [args.engine, "--registry", registry, "bridge", args.model,
             "--context-length", str(args.context)],
            stderr=log, env=env,
        )
        self.args = args
        self.count = 0

    def close(self):
        try:
            self.process.close()
        finally:
            self.directory.cleanup()

    def generate(self, prompt, budget):
        self.count += 1
        request_id = str(self.count)
        deadline = time.monotonic() + self.args.timeout
        request = {
            "type": "generate", "request_id": request_id,
            "conversation_id": f"abba-{request_id}",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": budget, "reuse_prompt_cache": False,
            **SAMPLERS[self.args.sampler],
        }
        if self.args.mtp_head:
            request.update(mtp=True, mtp_head_path=self.args.mtp_head,
                           mtp_depth=self.args.mtp_depth)
        self.process.send(request, deadline)
        while True:
            event = self.process.receive(deadline)
            if event.get("type") == "error" and event.get("request_id") in (None, request_id):
                raise RuntimeError(f"bridge error: {event}")
            if event.get("request_id") != request_id or event.get("type") != "complete":
                continue
            stats = event.get("stats") or {}
            generated = stats.get("generated_tokens")
            tps = stats.get("decode_tps")
            if (type(generated) is not int or not 1 <= generated <= budget
                    or not isinstance(tps, (int, float)) or not tps > 0
                    or not isinstance(event.get("token_hash"), str) or not event["token_hash"]):
                raise ValueError(f"invalid completion: {event}")
            memory = stats.get("memory") or {}
            return {
                "token_hash": event["token_hash"],
                "stop_reason": event.get("stop_reason"),
                "generated_tokens": generated,
                "decode_tps": tps,
                "elapsed_seconds": stats.get("elapsed_seconds"),
                "mlx_active_bytes": memory.get("mlx_active_bytes"),
                "mlx_peak_bytes": memory.get("mlx_peak_bytes"),
                "process_footprint_bytes": memory.get("process_footprint_bytes"),
                **{name: stats.get(name) for name in COUNTERS if stats.get(name) is not None},
            }


def run_pass(args, flags, selected, log):
    bridge = Bridge(args, flags, log)
    try:
        for _, prompt, budget in selected:
            bridge.generate(prompt, min(budget, 32))
        return {name: [bridge.generate(prompt, budget) for _ in range(args.repeats)]
                for name, prompt, budget in selected}
    finally:
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("model")
    parser.add_argument("--long-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="report JSON path")
    parser.add_argument("--off", action="append", default=[], help="NAME=VALUE (baseline)")
    parser.add_argument("--on", action="append", default=[], help="NAME=VALUE (candidate)")
    parser.add_argument("--sampler", choices=tuple(SAMPLERS), default="greedy")
    parser.add_argument("--mtp-head", default="")
    parser.add_argument("--mtp-depth", type=int, default=2)
    parser.add_argument("--cases", default="copy,edit,code,fr,long")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--timeout", type=positive_seconds, default=600.0)
    parser.add_argument("--settle", type=float, default=2.0)
    args = parser.parse_args()
    off, on = parse_env(args.off), parse_env(args.on)
    if off == on or args.repeats < 1:
        raise SystemExit("OFF and ON must differ and repeats must be positive")
    long_text = args.long_file.read_text()
    wanted = args.cases.split(",")
    selected = [case for case in _cases.cases(long_text) if case[0] in wanted]
    if [c[0] for c in selected] != [w for w in wanted if w in {c[0] for c in selected}] \
            or len(selected) != len(wanted):
        raise SystemExit("unknown or duplicated case name")
    sampled = args.sampler != "greedy"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "off": off, "on": on, "sampler": args.sampler,
        "sampled": sampled, "mtp_head": args.mtp_head or None,
        "mtp_depth": args.mtp_depth if args.mtp_head else None,
        "engine_sha256": hashlib.sha256(Path(args.engine).read_bytes()).hexdigest(),
        "model": args.model, "cases": wanted, "repeats": args.repeats,
        "long_file_sha256": hashlib.sha256(long_text.encode()).hexdigest(),
        "power_before": subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                       text=True, check=False).stdout.strip(),
        "load_before": os.getloadavg(),
    }
    try:
        with open(args.out.with_suffix(".stderr.log"), "wb") as log:
            passes = []
            for order, (label, flags) in enumerate(
                    (("off", off), ("on", on), ("on", on), ("off", off))):
                time.sleep(args.settle)
                passes.append({"arm": label, "load": os.getloadavg(),
                               "results": run_pass(args, flags, selected, log)})
                print(f"pass {order} {label} done", flush=True)
        problems, summary = [], {}
        for name in wanted:
            rows = {"off": [], "on": []}
            for item in passes:
                rows[item["arm"]].extend(item["results"][name])
            hashes = {run["token_hash"] for arm in rows.values() for run in arm}
            if not sampled and len(hashes) != 1:
                problems.append(f"{name}: greedy token hashes differ across passes")
            mismatches = [r.get("lfm3_cert_check_mismatches") for r in rows["on"]]
            if any(m not in (None, 0) for m in mismatches):
                problems.append(f"{name}: certified-head check mismatches {mismatches}")
            tps = {arm: [r["decode_tps"] for r in runs] for arm, runs in rows.items()}
            mean = {arm: statistics.mean(v) for arm, v in tps.items()}
            # Drift: first versus last pass of the same arm (passes 0/3 and 1/2).
            per_pass = [statistics.mean(r["decode_tps"] for r in item["results"][name])
                        for item in passes]
            summary[name] = {
                "decode_tps_off": mean["off"], "decode_tps_on": mean["on"],
                "decode_gain_percent": 100 * (mean["on"] / mean["off"] - 1),
                "pass_means": per_pass,
                "drift_off_percent": 100 * (per_pass[3] / per_pass[0] - 1),
                "drift_on_percent": 100 * (per_pass[2] / per_pass[1] - 1),
                "generated_tokens": {arm: [r["generated_tokens"] for r in runs]
                                     for arm, runs in rows.items()},
                "distinct_hashes": len(hashes),
                "mlx_active_bytes": {arm: max((r["mlx_active_bytes"] or 0) for r in runs)
                                     for arm, runs in rows.items()},
                "mlx_peak_bytes": {arm: max((r["mlx_peak_bytes"] or 0) for r in runs)
                                   for arm, runs in rows.items()},
                "process_footprint_bytes": {
                    arm: max((r["process_footprint_bytes"] or 0) for r in runs)
                    for arm, runs in rows.items()},
                "counters_on_first": {k: v for k, v in rows["on"][0].items() if k in COUNTERS},
            }
        gains = [row["decode_gain_percent"] for row in summary.values()]
        report.update(passes=passes, summary=summary, problems=problems,
                      mean_gain_percent=statistics.mean(gains),
                      status="passed" if not problems else "failed")
    except BaseException as error:
        report["status"] = "error"
        report["error"] = repr(error)
        raise
    finally:
        report["power_after"] = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                               text=True, check=False).stdout.strip()
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({k: report.get(k) for k in ("status", "problems", "mean_gain_percent")}))
        for name, row in (report.get("summary") or {}).items():
            print(name, f"{row['decode_tps_off']:.2f} -> {row['decode_tps_on']:.2f}",
                  f"({row['decode_gain_percent']:+.1f} %)",
                  f"drift off {row['drift_off_percent']:+.1f} % on {row['drift_on_percent']:+.1f} %")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
