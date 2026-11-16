#!/usr/bin/env python3
"""LFM3-PLD / LFM3-CERT-HEAD: bridge parity and ABBA timing for an opt-in flag.

`--feature lookup` (default) drives MLXL3_LFM3_LOOKUP and its lfm3_lookup_*
stats; `--feature cert` drives MLXL3_LFM3_CERT_HEAD and lfm3_cert_* stats.

`parity` runs every case with the lookup off, then on (one bridge process at
a time) and requires identical token hashes, text, counts and stop reasons,
plus verification blocks on copy cases. `abba` times the same binary with
the flag 0/1/1/0 after warmup. Results are written even on failure with an
explicit status; a failed or partial run never reports `parity: true`.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_json_process import JsonProcess, positive_seconds  # noqa: E402

PASSAGE = (
    "Le port de Saint-Malo s'anime avant l'aube. Les pêcheurs vérifient leurs filets, "
    "les boulangers sortent les premières fournées et les mouettes tournent au-dessus "
    "des remparts. Quand la marée remonte, les bateaux quittent le bassin un par un, "
    "en suivant le chenal balisé par des bouées rouges et vertes. Le soir, les familles "
    "se retrouvent sur la plage du Sillon pour regarder le soleil disparaître derrière "
    "l'île du Grand Bé, tandis que le vent apporte une odeur d'iode et de pain chaud."
)
CODE = '''def load_users(path):
    users = []
    with open(path) as handle:
        for line in handle:
            name, age = line.strip().split(",")
            users.append({"name": name, "age": int(age)})
    return users


def adults(users):
    return [user for user in users if user["age"] >= 18]
'''


def cases(long_text: str):
    return [
        ("copy", f"Recopie exactement le texte suivant, sans rien changer :\n\n{PASSAGE}", 256),
        (
            "edit",
            "Rename the variable `users` to `people` everywhere and return the complete "
            f"code only:\n\n```python\n{CODE}```",
            256,
        ),
        ("code", "Write a Rust function that parses an ISO 8601 date like 2026-11-16.", 256),
        ("fr", "Explique en quelques paragraphes pourquoi le ciel est bleu.", 256),
        ("long", f"{long_text}\n\nRésume ce document en cinq points.", 256),
        ("eos", "Recopie exactement cette phrase et rien d'autre : « Bonjour le monde. »", 64),
    ] + [
        (f"copy-budget-{budget}", f"Recopie exactement :\n\n{PASSAGE}", budget)
        for budget in (1, 2, 3, 4, 5, 9, 200, 201, 202, 203)
    ] + [
        ("eos-long", "Recopie exactement cette phrase et rien d'autre : « Bonjour le monde. »", 768),
    ]


class Bridge:
    def __init__(self, engine, model, context, env_name, lookup, timeout, log):
        self.directory = tempfile.TemporaryDirectory(prefix="mlxl3-lfm3-lookup-")
        env = dict(os.environ)
        env.pop("MLXL3_LFM3_LOOKUP", None)
        env.pop("MLXL3_LFM3_CERT_HEAD", None)
        if lookup != "unset":
            env[env_name] = lookup
        registry = str(Path(self.directory.name) / "models.json")
        self.process = JsonProcess(
            [engine, "--registry", registry, "bridge", model, "--context-length", str(context)],
            stderr=log,
            env=env,
        )
        self.timeout = timeout
        self.count = 0

    def close(self):
        try:
            self.process.close()
        finally:
            self.directory.cleanup()

    def generate(self, prompt, budget):
        self.count += 1
        request_id = str(self.count)
        deadline = time.monotonic() + self.timeout
        self.process.send(
            {
                "type": "generate",
                "request_id": request_id,
                "conversation_id": f"lfm3-lookup-{request_id}",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": budget,
                "temperature": 0,
                "top_k": 1,
                "repetition_penalty": 1,
                "reuse_prompt_cache": False,
            },
            deadline,
        )
        text = ""
        while True:
            event = self.process.receive(deadline)
            if event.get("type") == "error" and event.get("request_id") in (None, request_id):
                raise RuntimeError(f"bridge error: {event}")
            if event.get("request_id") != request_id:
                continue
            if event["type"] == "delta" and event.get("phase") == "answer":
                text += event.get("text", "")
            if event["type"] == "complete":
                stats = event.get("stats") or {}
                generated = stats.get("generated_tokens")
                if (
                    type(generated) is not int
                    or not 1 <= generated <= budget
                    or not isinstance(event.get("token_hash"), str)
                    or not event["token_hash"]
                ):
                    raise ValueError(f"invalid completion: {event}")
                return {
                    "token_hash": event["token_hash"],
                    "text": text,
                    "cache_context": event.get("cache_context"),
                    "stop_reason": event.get("stop_reason"),
                    "generated_tokens": generated,
                    "prompt_tokens": stats.get("prompt_tokens"),
                    "decode_tps": stats.get("decode_tps"),
                    "decode_seconds": stats.get("decode_seconds"),
                    "elapsed_seconds": stats.get("elapsed_seconds"),
                    "blocks": stats.get("lfm3_lookup_blocks"),
                    "proposed": stats.get("lfm3_lookup_proposed_tokens"),
                    "accepted": stats.get("lfm3_lookup_accepted_tokens"),
                    "cert_tokens": stats.get("lfm3_cert_tokens"),
                    "cert_fallbacks": stats.get("lfm3_cert_fallbacks"),
                }


def run_pass(args, lookup, selected, log, repeats=1, warmup=0):
    bridge = Bridge(
        args.engine, args.model, args.context, ENV[args.feature], lookup, args.timeout, log
    )
    try:
        for name, prompt, budget in selected[:warmup]:
            bridge.generate(prompt, min(budget, 32))
        results = {}
        for name, prompt, budget in selected:
            results[name] = [bridge.generate(prompt, budget) for _ in range(repeats)]
        return results
    finally:
        bridge.close()


ENV = {"lookup": "MLXL3_LFM3_LOOKUP", "cert": "MLXL3_LFM3_CERT_HEAD"}


def check_cert_parity(off, on):
    problems = []
    fields = ("token_hash", "text", "cache_context", "stop_reason", "generated_tokens")
    for name, runs in off.items():
        for index, (a, b) in enumerate(zip(runs, on[name])):
            for field in fields:
                if a[field] != b[field]:
                    problems.append(f"{name}[{index}].{field} differs")
            if a["cert_tokens"] is not None or a["cert_fallbacks"] is not None:
                problems.append(f"{name}: cert stats present with the flag off")
            if b["blocks"] is not None or a["blocks"] is not None:
                problems.append(f"{name}: lookup stats present")
            steps = (b["cert_tokens"], b["cert_fallbacks"])
            if any(type(v) is not int for v in steps):
                problems.append(f"{name}: cert stats missing with the flag on")
            # One forward per emitted ID except the last (or the ID before an end ID).
            elif sum(steps) not in (b["generated_tokens"] - 1, b["generated_tokens"]):
                problems.append(f"{name}: {steps} cert steps for {b['generated_tokens']} IDs")
    if not any((runs[0]["cert_tokens"] or 0) > 0 for runs in on.values()):
        problems.append("no certified step")
    eos = [name for name, runs in on.items() if runs[0]["stop_reason"] != "max_tokens"]
    if "eos-long" in on and "eos-long" not in eos:
        problems.append("eos-long: generation did not end before its budget")
    return problems


def check_parity(off, on):
    problems = []
    fields = ("token_hash", "text", "cache_context", "stop_reason", "generated_tokens")
    for name, runs in off.items():
        for index, (a, b) in enumerate(zip(runs, on[name])):
            for field in fields:
                if a[field] != b[field]:
                    problems.append(f"{name}[{index}].{field} differs")
            if a["blocks"] is not None:
                problems.append(f"{name}: lookup stats present with the flag off")
            if b["blocks"] is None:
                problems.append(f"{name}: lookup stats missing with the flag on")
    eos = [name for name, runs in on.items() if runs[0]["stop_reason"] != "max_tokens"]
    if "eos-long" in on and "eos-long" not in eos:
        problems.append("eos-long: generation did not end before its budget")
    for name in ("copy", "edit", *(f"copy-budget-{b}" for b in (200, 201, 202, 203))):
        if name in on and not (on[name][0]["blocks"] or 0) > 0:
            problems.append(f"{name}: no verification block")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parity", "abba"))
    parser.add_argument("engine")
    parser.add_argument("model")
    parser.add_argument("--long-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--timeout", type=positive_seconds, default=600.0)
    parser.add_argument("--cases", default="")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument(
        "--flag", default="1", help="flag value for the candidate (`unset` removes it)"
    )
    parser.add_argument("--off", default="0", help="flag value for the baseline")
    parser.add_argument("--feature", choices=tuple(ENV), default="lookup")
    args = parser.parse_args()
    long_text = args.long_file.read_text()
    selected = cases(long_text)
    if args.cases:
        wanted = args.cases.split(",")
        selected = [case for case in selected if case[0] in wanted]
        if len(selected) != len(wanted):
            raise SystemExit("unknown case name")
    args.out.mkdir(parents=True, exist_ok=True)
    engine_hash = hashlib.sha256(Path(args.engine).read_bytes()).hexdigest()
    report = {
        "status": "running",
        "mode": args.mode,
        "flag": args.flag,
        "off": args.off,
        "feature": args.feature,
        "env_name": ENV[args.feature],
        "engine_sha256": engine_hash,
        "model": args.model,
        "long_file_sha256": hashlib.sha256(long_text.encode()).hexdigest(),
        "cases": [name for name, _, _ in selected],
        "power": subprocess.run(
            ["pmset", "-g", "batt"], capture_output=True, text=True, check=False
        ).stdout.strip(),
    }
    result_path = args.out / f"{args.mode}-{args.flag}.json"
    try:
        with open(args.out / f"{args.mode}-stderr.log", "wb") as log:
            if args.mode == "parity":
                off = run_pass(args, args.off, selected, log)
                on = run_pass(args, args.flag, selected, log)
                problems = (check_cert_parity if args.feature == "cert" else check_parity)(
                    off, on
                )
                report.update(off=off, on=on, problems=problems)
                report["parity"] = not problems
                report["status"] = "passed" if not problems else "failed"
            else:
                passes = []
                for order, flag in enumerate((args.off, args.flag, args.flag, args.off)):
                    time.sleep(args.settle)
                    passes.append(
                        {
                            "flag": flag,
                            "results": run_pass(
                                args, flag, selected, log, args.repeats, warmup=len(selected)
                            ),
                        }
                    )
                    print(f"pass {order} flag={flag} done", flush=True)
                summary = {}
                problems = []
                for name, _, _ in selected:
                    rows = {args.off: [], args.flag: []}
                    hashes = set()
                    for item in passes:
                        for run in item["results"][name]:
                            rows[item["flag"]].append(run)
                            hashes.add(run["token_hash"])
                    if len(hashes) != 1:
                        problems.append(f"{name}: token hashes differ across passes")
                    tps = {k: [r["decode_tps"] for r in v] for k, v in rows.items()}
                    total = {k: [r["elapsed_seconds"] for r in v] for k, v in rows.items()}
                    on_rows = rows[args.flag]
                    tps["0"], total["0"] = tps[args.off], total[args.off]
                    tps["1"], total["1"] = tps[args.flag], total[args.flag]
                    base, cand = statistics.mean(tps["0"]), statistics.mean(tps["1"])
                    summary[name] = {
                        "decode_tps_off": base,
                        "decode_tps_on": cand,
                        "decode_gain_percent": 100 * (cand / base - 1),
                        "elapsed_off": statistics.mean(total["0"]),
                        "elapsed_on": statistics.mean(total["1"]),
                        "elapsed_change_percent": 100
                        * (statistics.mean(total["1"]) / statistics.mean(total["0"]) - 1),
                        "drift_off_percent": 100 * (tps["0"][-1] / tps["0"][0] - 1),
                        "drift_on_percent": 100 * (tps["1"][-1] / tps["1"][0] - 1),
                        "generated_tokens": rows[args.off][0]["generated_tokens"],
                        "blocks_on": on_rows[0]["blocks"],
                        "accepted_on": on_rows[0]["accepted"],
                        "proposed_on": on_rows[0]["proposed"],
                        "cert_tokens_on": on_rows[0]["cert_tokens"],
                        "cert_fallbacks_on": on_rows[0]["cert_fallbacks"],
                    }
                report.update(passes=passes, summary=summary, problems=problems)
                report["parity"] = not problems
                report["status"] = "passed" if not problems else "failed"
    except BaseException as error:
        report["status"] = "error"
        report["parity"] = False
        report["error"] = repr(error)
        raise
    finally:
        result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps({k: report.get(k) for k in ("status", "parity", "problems")}))
        if report.get("summary"):
            for name, row in report["summary"].items():
                print(name, json.dumps(row))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
