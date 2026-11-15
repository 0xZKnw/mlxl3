"""Q36-MTP-POSTNORM-T1: acceptance of the four hidden wirings against the pre-norm baseline.

Inputs: t1/<variant>/diag.json for pre, seed, chain, post (confidence 0, 256 tokens) and the T0a
run1 diag.json. Fails closed. Usage: python -I analyze_t1.py <output.json> <t1 dir> <t0a run1 dir>
"""

import json
import math
import sys
from pathlib import Path

Z = 1.959963984540054
VARIANTS = ("pre", "seed", "chain", "post")


def wilson(hit, n):
    if n == 0:
        return None
    p = hit / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return {"rate": p, "low": c - h, "high": c + h, "hit": hit, "n": n}


def positions(h):
    out = []
    for p in range(3):
        hit = sum(h[w][a] for w in range(4) for a in range(4) if w > p and a > p)
        tested = sum(h[w][a] for w in range(4) for a in range(4) if w > p and a >= p)
        out.append((hit, tested))
    return out


def main():
    output, root, t0a = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    errors, result = [], {"cases": [], "pooled": {}, "decision": {}}
    try:
        runs = {}
        for v in VARIANTS:
            diag = json.loads((root / v / "diag.json").read_text())
            if diag.get("hidden_variant") != v or diag.get("confidence"):
                errors.append(f"{v}: wrong variant or confidence capture on")
            if not diag.get("compact_head"):
                errors.append(f"{v}: compact head not active")
            runs[v] = {(c["prompt"], c["depth"]): c for c in diag["cases"]}
        base = {
            (c["prompt"], c["depth"]): c
            for c in json.loads((t0a / "diag.json").read_text())["cases"]
        }
        keys = sorted(base)
        if len(keys) != 12 or any(set(runs[v]) != set(keys) for v in VARIANTS):
            errors.append("case sets differ or are not 4 prompts x 3 depths")
        for key in keys:
            pre, b = runs["pre"][key], base[key]
            for field in ("tokens", "cycles", "histogram"):
                if pre[field] != b[field]:
                    errors.append(f"pre {key}: {field} differs from T0a")
            for v in VARIANTS:
                if runs[v][key]["tokens"] != b["tokens"]:
                    errors.append(f"{v} {key}: tokens differ from the AR-checked baseline")
        for key in keys:
            row = {"prompt": key[0], "depth": key[1]}
            for v in VARIANTS:
                c = runs[v][key]
                row[v] = {
                    "accepted": c["accepted"],
                    "proposed": c["proposed"],
                    "cycles": len(c["cycles"]),
                    "positions": positions(c["histogram"]),
                    "changed_cycles_vs_pre": sum(
                        1 for x, y in zip(c["cycles"], runs["pre"][key]["cycles"], strict=False) if x != y
                    )
                    + abs(len(c["cycles"]) - len(runs["pre"][key]["cycles"])),
                }
            result["cases"].append(row)
        for v in VARIANTS:
            for depth in (1, 2, 3):
                tot = [[0, 0] for _ in range(3)]
                acc = prop = 0
                for key in keys:
                    if key[1] != depth:
                        continue
                    c = runs[v][key]
                    acc += c["accepted"]
                    prop += c["proposed"]
                    for p, (hit, n) in enumerate(positions(c["histogram"])):
                        tot[p][0] += hit
                        tot[p][1] += n
                result["pooled"][f"{v}-MTP{depth}"] = {
                    "accepted": acc,
                    "proposed": prop,
                    "positions": [wilson(hit, n) for hit, n in tot[:depth]],
                }
        for v in VARIANTS[1:]:
            for depth in (1, 2, 3):
                gains = [
                    runs[v][k]["accepted"] - runs["pre"][k]["accepted"]
                    for k in keys
                    if k[1] == depth
                ]
                result["decision"][f"{v}-MTP{depth}"] = {
                    "accepted_delta_by_prompt": gains,
                    "improved_prompts": sum(g > 0 for g in gains),
                }
            ok = [
                d
                for d in (1, 2, 3)
                if result["decision"][f"{v}-MTP{d}"]["improved_prompts"] >= 3
                and all(
                    min(result["decision"][f"{v}-MTP{o}"]["accepted_delta_by_prompt"]) >= -1
                    for o in (1, 2, 3)
                    if o != d
                )
            ]
            result["decision"][f"{v}-qualifies_at_depths"] = ok
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    result["status"] = "passed" if not errors else "failed"
    result["errors"] = errors
    output.write_text(json.dumps(result, indent=2))
    for row in result["cases"]:
        print(
            f"{row['prompt']:14s} MTP{row['depth']} "
            + " | ".join(
                f"{v} {row[v]['accepted']}/{row[v]['proposed']} (chg {row[v]['changed_cycles_vs_pre']})"
                for v in VARIANTS
            )
        )
    for k, s in result["pooled"].items():
        print(
            k,
            s["accepted"],
            "/",
            s["proposed"],
            [f"{w['rate']:.3f} [{w['low']:.3f};{w['high']:.3f}]" for w in s["positions"]],
        )
    for k, s in result["decision"].items():
        print(k, s)
    print("\n".join(errors) if errors else "all T1 checks passed")
    print("t1_status", result["status"])
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
