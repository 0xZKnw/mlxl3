#!/usr/bin/env python3
"""SIM-DRAFTLESS-20261116: offline sieve of network-free drafters on real greedy outputs.

A CPU simulation, not a speed measurement. Token IDs are a retokenization of
the emitted text (approximation of the emitted IDs). The drafter proposes the
continuation of the latest earlier occurrence of the longest committed suffix
(anchor included) of length >= m; acceptance is the longest proposal prefix
equal to the real continuation. Cost model in T1 steps per verification of R
rows comes from measured microbenchmarks (see opti.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

COSTS = {
    "lfm3_p300": {1: 1.0, 2: 1.24, 3: 1.31, 4: 1.31},
    "lfm3_p2000": {1: 1.0, 2: 1.27, 3: 1.39, 4: 1.43},
    "q36": {1: 1.0, 2: 1.40, 3: 1.80, 4: 2.20},
    # HYPOTHETICAL: rows 5..8 are not supported by verify_rows today (R <= 4).
    # +0.05 step per extra row extrapolates the measured R3 -> R4 increment at
    # prefix 2000 (+1.1 ms); it is not a measurement.
    "lfm3_hyp8": {1: 1.0, 2: 1.27, 3: 1.39, 4: 1.43, 5: 1.48, 6: 1.53, 7: 1.58, 8: 1.63},
}
MAX_L = 8


def simulate(prompt, output, m, width, window, adaptive=False, oracle=False):
    """Returns blocks, proposed, accepted and rows histogram for one policy."""
    seq = prompt + output
    start = len(prompt)
    last = {}  # (L-gram) -> end index of its latest occurrence
    inserted = 0  # grams ending at e <= inserted are indexed

    def index_upto(limit):
        nonlocal inserted
        for end in range(inserted + 1, limit + 1):
            for length in range(1, MAX_L + 1):
                if end - length >= 0:
                    last[tuple(seq[end - length : end])] = end
        inserted = max(inserted, limit)

    i = start
    blocks = proposed = accepted = 0
    rows = dict.fromkeys(range(1, 9), 0)
    while i < len(seq):
        # Committed tokens are seq[:i]; the anchor is seq[i-1]. Earlier
        # occurrences must end strictly before i so a continuation exists.
        index_upto(i - 1)
        remaining = len(seq) - i  # tokens still to emit
        proposal = []
        match = 0
        for length in range(min(MAX_L, i), m - 1, -1):
            end = last.get(tuple(seq[i - length : i]))
            if end is None or (window and i - end > window):
                continue
            limit = width
            if adaptive:
                limit = 1 if length < 3 else 2 if length < 5 else 3
            limit = min(limit, remaining - 1, i - end)
            if limit >= 1:
                proposal = seq[end : end + limit]
                match = length
            break
        good = 0
        for offset, token in enumerate(proposal):
            if seq[i + offset] != token:
                break
            good += 1
        if oracle:
            proposal = proposal[:good]
        if proposal:
            blocks += 1
            proposed += len(proposal)
            accepted += good
            rows[1 + len(proposal)] += 1
            i += good + 1
        else:
            rows[1] += 1
            i += 1
        del match
    return {"blocks": blocks, "proposed": proposed, "accepted": accepted, "rows": rows}


def speedup(result, tokens, cost):
    used = {r: count for r, count in result["rows"].items() if count}
    if any(r not in cost for r in used):
        return None  # this cost table has no figure for that many rows
    return tokens / sum(cost[r] * count for r, count in used.items())


def load_cases(args, tokenizer):
    cases = []

    def add(name, prompt_text, output_text, generated, source, measured=None):
        prompt = tokenizer.encode(prompt_text, add_special_tokens=False).ids
        output = tokenizer.encode(output_text, add_special_tokens=False).ids
        cases.append(
            {
                "name": name,
                "source": source,
                "prompt": prompt,
                "output": output,
                "generated_reported": generated,
                "measured": measured,
            }
        )

    if args.parity:
        import importlib.util

        spec = importlib.util.spec_from_file_location("lookup", args.lookup_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        long_text = Path(args.long_file).read_text()
        prompts = {name: prompt for name, prompt, _ in module.cases(long_text)}
        report = json.loads(Path(args.parity).read_text())
        for name in ("copy", "edit", "code", "fr", "long", "eos-long"):
            row = report["on"][name][0]
            add(
                name,
                prompts[name],
                row["cache_context"],
                row["generated_tokens"],
                "lfm3-pld parity4",
                {k: row[k] for k in ("blocks", "proposed", "accepted")},
            )
    for path in args.d2 or []:
        report = json.loads(Path(path).read_text())
        import importlib.util

        spec = importlib.util.spec_from_file_location("d2", args.d2_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        seen = set()
        for run in report["runs"]:
            if run["label"] not in args.labels.split(","):
                continue
            for row in run["rows"]:
                if row["arm"] != "G" or row["case"] in seen:
                    continue
                seen.add(row["case"])
                add(
                    row["case"],
                    module.CASES[row["case"]],
                    row["cache_context"] or row["text"],
                    row["stats"]["generated_tokens"],
                    f"d2 {run['label']}",
                )
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--parity")
    parser.add_argument("--lookup-script")
    parser.add_argument("--long-file")
    parser.add_argument("--d2", action="append")
    parser.add_argument("--d2-script")
    parser.add_argument("--labels", default="A,AR")
    parser.add_argument("--costs", default="lfm3_p300,lfm3_p2000")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cases = load_cases(args, tokenizer)
    if not cases:
        raise SystemExit("no cases")
    policies = []
    for m in (1, 2, 3, 4, 6, 8):
        policies.append((f"m{m}-w3-win1024", dict(m=m, width=3, window=1024)))
    for m in (1, 2, 3, 4):
        policies.append((f"m{m}-w1-win1024", dict(m=m, width=1, window=1024)))
        policies.append((f"m{m}-adaptive-all", dict(m=m, width=3, window=0, adaptive=True)))
        policies.append((f"m{m}-w3-all", dict(m=m, width=3, window=0)))
        policies.append((f"m{m}-w3-all-ORACLE", dict(m=m, width=3, window=0, oracle=True)))
    for m in (2, 3, 4):
        policies.append((f"m{m}-w7-all-HYP", dict(m=m, width=7, window=0)))
    report = {"costs": {k: COSTS[k] for k in args.costs.split(",")}, "cases": []}
    for case in cases:
        tokens = len(case["output"])
        entry = {
            "name": case["name"],
            "source": case["source"],
            "prompt_tokens_retokenized": len(case["prompt"]),
            "output_tokens_retokenized": tokens,
            "generated_reported": case["generated_reported"],
            "measured": case["measured"],
            "policies": {},
        }
        for name, options in policies:
            result = simulate(case["prompt"], case["output"], **options)
            result["speedup"] = {}
            for key in args.costs.split(","):
                value = speedup(result, tokens, COSTS[key])
                result["speedup"][key] = None if value is None else round(value, 4)
            entry["policies"][name] = result
        report["cases"].append(entry)
    Path(args.out).write_text(json.dumps(report, indent=2))
    keys = args.costs.split(",")
    for entry in report["cases"]:
        print(
            f"\n== {entry['name']} ({entry['source']}) out={entry['output_tokens_retokenized']}"
            f" reported={entry['generated_reported']} prompt={entry['prompt_tokens_retokenized']}"
            f" measured={entry['measured']}"
        )
        for name, result in entry["policies"].items():
            rate = result["accepted"] / result["proposed"] if result["proposed"] else 0
            gains = " ".join(
                f"{key}=n/a"
                if result["speedup"][key] is None
                else f"{key}={(result['speedup'][key] - 1) * 100:+.1f}%"
                for key in keys
            )
            print(
                f"  {name:22s} blocks={result['blocks']:4d} acc={result['accepted']:4d}/"
                f"{result['proposed']:4d} ({rate:.2f}) {gains}"
            )


if __name__ == "__main__":
    main()
