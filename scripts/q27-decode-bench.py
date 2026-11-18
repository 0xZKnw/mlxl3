#!/usr/bin/env python3
"""Decode timing of named environment arms across AR, MTP and DFlash2 modes.

One fresh bridge process per pass (one GPU model at a time). A pass sets one
arm's environment, then for every requested mode (`ar`, `mtp:<depth>`,
`dflash:<mode>`) warms each case up at 32 tokens and times `--repeats`
generations per case. `--order` lists the arm of each pass (default ABBA).

Greedy runs must keep one token hash per case across every pass and mode of
the report (AR, MTP and DFlash2 are exact, so their IDs must agree); sampled
runs only need complete outputs and are flagged. The requested speculative
mode must be reported active by the bridge. The report is saved on every exit
with an explicit status; an error, timeout, inactive mode, short output or
hash divergence never yields `passed`.
"""

from __future__ import annotations

import argparse
import hashlib
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
from native_json_process import JsonProcess, positive_seconds

SAMPLERS = {
    "greedy": {"temperature": 0, "top_k": 1, "repetition_penalty": 1},
    "desktop": {"temperature": 0.2, "top_k": 80, "repetition_penalty": 1.05},
    "qwen": {"temperature": 0.6, "top_k": 20, "repetition_penalty": 1.0},
}
CASES = {
    "fr": "Explique en quelques paragraphes pourquoi le ciel est bleu.",
    "code": "Write a Rust function that parses an ISO 8601 date like 2026-11-16.",
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "math": "Explain step by step how to solve the quadratic equation 3x^2 - 5x - 2 = 0.",
}
COUNTERS = (
    "mtp_proposed_tokens",
    "mtp_accepted_tokens",
    "mtp_blocks",
    "mtp_depth_blocks",
    "dflash_proposed_tokens",
    "dflash_accepted_tokens",
    "dflash_blocks",
    "dflash_block_seconds",
    "topk_readback_rows",
)


def parse_arm(text):
    name, sep, rest = text.partition(":")
    if not sep or not name:
        raise SystemExit(f"invalid arm {text!r}; expected NAME:KEY=VALUE,...")
    env = {}
    for item in filter(None, rest.split(",")):
        key, eq, value = item.partition("=")
        if not eq or not key.startswith("MLXL3_"):
            raise SystemExit(f"invalid arm option {item!r}")
        env[key] = value
    return name, env


def parse_mode(text):
    kind, _, value = text.partition(":")
    if kind == "ar" and not value:
        return ("ar", 0)
    if kind in ("mtp", "dflash") and value.isdigit() and 1 <= int(value) <= 3:
        return (kind, int(value))
    raise SystemExit(f"invalid mode {text!r}; expected ar, mtp:1..3 or dflash:1..3")


class Bridge:
    def __init__(self, args, env_flags, toggled, log):
        self.directory = tempfile.TemporaryDirectory(prefix="mlxl3-q27-")
        env = {k: v for k, v in os.environ.items() if k not in toggled}
        env.update(env_flags)
        registry = str(Path(self.directory.name) / "models.json")
        self.process = JsonProcess(
            [
                args.engine,
                "--registry",
                registry,
                "bridge",
                args.model,
                "--context-length",
                str(args.context),
            ],
            stderr=log,
            env=env,
        )
        self.args = args
        self.count = 0
        self.dflash = None

    def close(self):
        try:
            self.process.close()
        finally:
            self.directory.cleanup()

    def set_dflash(self, mode):
        """Load or release the DFlash2 draft; `mode` is None to disable."""
        if self.dflash == mode:
            return
        self.count += 1
        request_id = f"dflash-{self.count}"
        deadline = time.monotonic() + self.args.timeout
        self.process.send(
            {
                "type": "set_dflash",
                "request_id": request_id,
                "enabled": mode is not None,
                "dflash_draft_path": self.args.dflash_draft,
                "dflash_mode": mode or 1,
            },
            deadline,
        )
        while True:
            event = self.process.receive(deadline)
            if event.get("request_id") != request_id:
                continue
            if event.get("type") == "error":
                raise RuntimeError(f"bridge error: {event}")
            if event.get("type") == "dflash_status":
                if bool(event.get("dflash_active")) != (mode is not None):
                    raise RuntimeError(f"unexpected DFlash state: {event}")
                self.dflash = mode
                return

    def generate(self, prompt, budget, mode):
        kind, level = mode
        self.set_dflash(level if kind == "dflash" else None)
        self.count += 1
        request_id = str(self.count)
        deadline = time.monotonic() + self.args.timeout
        request = {
            "type": "generate",
            "request_id": request_id,
            "conversation_id": f"q27-{request_id}",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": budget,
            "reuse_prompt_cache": False,
            **SAMPLERS[self.args.sampler],
        }
        if kind == "mtp":
            request.update(mtp=True, mtp_head_path=self.args.mtp_head, mtp_depth=level)
        if kind == "dflash":
            request.update(
                dflash2=True, dflash_draft_path=self.args.dflash_draft, dflash_mode=level
            )
        self.process.send(request, deadline)
        active = None
        while True:
            event = self.process.receive(deadline)
            if event.get("type") == "error" and event.get("request_id") in (None, request_id):
                raise RuntimeError(f"bridge error: {event}")
            if event.get("request_id") != request_id:
                continue
            if event.get("type") == "generation_mode":
                active = (bool(event.get("mtp_active")), bool(event.get("dflash_active")))
            if event.get("type") != "complete":
                continue
            if active != (kind == "mtp", kind == "dflash"):
                raise RuntimeError(f"mode {mode} not active: {active}")
            stats = event.get("stats") or {}
            generated = stats.get("generated_tokens")
            tps = stats.get("decode_tps")
            if (
                type(generated) is not int
                or generated != budget
                or not isinstance(tps, (int, float))
                or not tps > 0
                or not isinstance(event.get("token_hash"), str)
                or not event["token_hash"]
            ):
                raise ValueError(f"invalid or short completion: {event}")
            memory = stats.get("memory") or {}
            return {
                "token_hash": event["token_hash"],
                "generated_tokens": generated,
                "decode_tps": tps,
                "decode_seconds": stats.get("decode_seconds"),
                "prefill_seconds": stats.get("prefill_seconds"),
                "prompt_tokens": stats.get("prompt_tokens"),
                "mlx_active_bytes": memory.get("mlx_active_bytes"),
                "mlx_peak_bytes": memory.get("mlx_peak_bytes"),
                "process_footprint_bytes": memory.get("process_footprint_bytes"),
                **{name: stats.get(name) for name in COUNTERS if stats.get(name) is not None},
            }


def run_pass(args, flags, toggled, modes, log, results):
    """Fill `results` mode by mode so a later failure keeps completed modes."""
    bridge = Bridge(args, flags, toggled, log)
    try:
        for mode in modes:
            label = mode[0] if mode[0] == "ar" else f"{mode[0]}:{mode[1]}"
            for case in args.case_names:
                bridge.generate(CASES[case], 32, mode)
            results[label] = {
                case: [bridge.generate(CASES[case], args.tokens, mode) for _ in range(args.repeats)]
                for case in args.case_names
            }
            print(
                f"  {label}: "
                + ", ".join(
                    f"{case} {statistics.mean(r['decode_tps'] for r in rows):.2f}"
                    for case, rows in results[label].items()
                ),
                flush=True,
            )
    finally:
        bridge.close()


def conditions():
    def read(command):
        return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()

    return {
        "power": read(["pmset", "-g", "batt"]),
        "thermal": read(["pmset", "-g", "therm"]),
        "load": os.getloadavg(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("model")
    parser.add_argument("--out", type=Path, required=True, help="new report JSON path")
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="NAME:MLXL3_X=1,... (repeat; empty option list allowed)",
    )
    parser.add_argument("--order", default="", help="comma list of arm names per pass")
    parser.add_argument(
        "--mode", action="append", required=True, help="ar, mtp:<depth> or dflash:<mode> (repeat)"
    )
    parser.add_argument("--sampler", choices=tuple(SAMPLERS), default="greedy")
    parser.add_argument("--mtp-head", default="")
    parser.add_argument("--dflash-draft", default="")
    parser.add_argument("--cases", default="fr,code")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--timeout", type=positive_seconds, default=600.0)
    parser.add_argument("--settle", type=float, default=2.0)
    args = parser.parse_args()
    arms = dict(parse_arm(item) for item in args.arm)
    if len(arms) != len(args.arm):
        raise SystemExit("duplicated arm name")
    order = (
        args.order.split(",")
        if args.order
        else (list(arms) if len(arms) == 1 else [*arms, *reversed(arms)])
    )
    modes = [parse_mode(item) for item in args.mode]
    args.case_names = args.cases.split(",")
    if (
        any(name not in arms for name in order)
        or len(set(modes)) != len(modes)
        or any(name not in CASES for name in args.case_names)
        or len(set(args.case_names)) != len(args.case_names)
        or args.repeats < 1
        or not 33 <= args.tokens <= 1024
        or args.out.exists()
        or (any(kind == "mtp" for kind, _ in modes) and not Path(args.mtp_head).is_dir())
        or (any(kind == "dflash" for kind, _ in modes) and not Path(args.dflash_draft).is_dir())
    ):
        raise SystemExit(
            "invalid order, modes, cases, budget, existing report or missing head/draft"
        )
    toggled = {key for env in arms.values() for key in env}
    sampled = args.sampler != "greedy"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "arms": arms,
        "order": order,
        "modes": args.mode,
        "sampler": args.sampler,
        "sampled": sampled,
        "tokens": args.tokens,
        "repeats": args.repeats,
        "context": args.context,
        "cases": args.case_names,
        "engine": args.engine,
        "engine_sha256": hashlib.sha256(Path(args.engine).read_bytes()).hexdigest(),
        "model": args.model,
        "mtp_head": args.mtp_head or None,
        "dflash_draft": args.dflash_draft or None,
        "before": conditions(),
        "passes": [],
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    try:
        with open(args.out.with_suffix(".stderr.log"), "wb") as log:
            for index, name in enumerate(order):
                time.sleep(args.settle)
                print(f"pass {index} arm {name}", flush=True)
                entry = {
                    "arm": name,
                    "conditions": conditions(),
                    "status": "running",
                    "results": {},
                }
                report["passes"].append(entry)
                run_pass(args, arms[name], toggled, modes, log, entry["results"])
                entry["status"] = "complete"
                args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        problems, summary = [], {}
        for case in args.case_names:
            hashes = {
                run["token_hash"]
                for item in report["passes"]
                for rows in item["results"].values()
                for run in rows[case]
            }
            if not sampled and len(hashes) != 1:
                problems.append(f"{case}: {len(hashes)} greedy token hashes across arms/modes")
        for label in report["passes"][0]["results"]:
            for case in args.case_names:
                per_pass = [
                    statistics.mean(r["decode_tps"] for r in item["results"][label][case])
                    for item in report["passes"]
                ]
                by_arm = {
                    name: [
                        value
                        for value, item in zip(per_pass, report["passes"])
                        if item["arm"] == name
                    ]
                    for name in arms
                }
                means = {name: statistics.mean(values) for name, values in by_arm.items() if values}
                first = next(iter(means))
                summary[f"{label}/{case}"] = {
                    "decode_tps": means,
                    "pass_means": per_pass,
                    "gain_percent_vs_first": {
                        name: 100 * (value / means[first] - 1) for name, value in means.items()
                    },
                    "drift_percent": {
                        name: 100 * (values[-1] / values[0] - 1)
                        for name, values in by_arm.items()
                        if len(values) > 1
                    },
                    "counters": {
                        name: next(
                            (
                                {
                                    k: v
                                    for k, v in item["results"][label][case][0].items()
                                    if k in COUNTERS
                                }
                                for item in report["passes"]
                                if item["arm"] == name
                            ),
                            {},
                        )
                        for name in arms
                    },
                }
        report.update(
            summary=summary, problems=problems, status="passed" if not problems else "failed"
        )
    except BaseException as error:
        report["status"] = "error"
        report["error"] = repr(error)
        raise
    finally:
        report["after"] = conditions()
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({k: report.get(k) for k in ("status", "problems")}))
        for name, row in (report.get("summary") or {}).items():
            print(
                name,
                " ".join(f"{arm} {value:.2f}" for arm, value in row["decode_tps"].items()),
                " ".join(
                    f"{arm} {gain:+.1f}%"
                    for arm, gain in list(row["gain_percent_vs_first"].items())[1:]
                ),
                "drift",
                " ".join(f"{arm} {value:+.1f}%" for arm, value in row["drift_percent"].items()),
            )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
