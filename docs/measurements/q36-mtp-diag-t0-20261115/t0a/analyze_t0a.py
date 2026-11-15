"""Q36-MTP-ACCEPT-POS-T0A: per-position acceptance and draft confidence from mtp_cycle_diagnostics.

Inputs: run directories with diag.json produced with MLXL3_MTP_DIAG_CONFIDENCE=1. The first run is
analysed; every other run must be identical (tokens, cycles, histograms, proposals and bitwise
confidence). Fails closed. Usage: python -I analyze_t0a.py <output.json> <run1> [<run2> ...]
"""

import json
import math
import sys
from pathlib import Path

Z = 1.959963984540054
BANDS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.0000001)]


def wilson(hit, n):
    if n == 0:
        return None
    p = hit / n
    denominator = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denominator
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denominator
    return {"rate": p, "low": centre - half, "high": centre + half, "hit": hit, "n": n}


def auc(positive, negative):
    """Mann-Whitney AUC, ties count one half; None without both classes."""
    if not positive or not negative:
        return None
    wins = 0.0
    for p in positive:
        for q in negative:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(positive) * len(negative))


def samples(case):
    """(position, accepted, top1, margin) for every tested draft position, in cycle order."""
    levels = case["trace"]["levels"]
    out, cursor = [], 0
    for width, accepted in case["cycles"]:
        if not 0 <= accepted <= width <= case["depth"]:
            raise ValueError(f"invalid cycle {width},{accepted}")
        chunk = levels[cursor : cursor + width]
        if len(chunk) != width:
            raise ValueError("fewer confidence levels than proposals")
        cursor += width
        for position, level in enumerate(chunk):
            if level["proposal"] != level["host_proposal"] or not level["finite"]:
                raise ValueError("host argmax mismatch or non-finite level")
            if position > accepted:
                break  # after the first rejection, later positions are never tested
            top1, top2 = level["top1_log_prob"], level["top2_log_prob"]
            out.append((position, position < accepted, top1, top1 - top2))
    if cursor != len(levels):
        raise ValueError("more confidence levels than proposals")
    return out


def summarize(sample_list):
    by_position = []
    for position in range(3):
        s = [x for x in sample_list if x[0] == position]
        hit = sum(1 for x in s if x[1])
        by_position.append(wilson(hit, len(s)))
    accepted = [x for x in sample_list if x[1]]
    rejected = [x for x in sample_list if not x[1]]
    bands = []
    for low, high in BANDS:
        s = [x for x in sample_list if low <= math.exp(x[2]) < high]
        bands.append(
            {
                "p_top1": [low, min(high, 1.0)],
                "n": len(s),
                "accept_rate": (sum(1 for x in s if x[1]) / len(s)) if s else None,
            }
        )
    return {
        "positions": by_position,
        "auc_top1_log_prob": auc([x[2] for x in accepted], [x[2] for x in rejected]),
        "auc_margin": auc([x[3] for x in accepted], [x[3] for x in rejected]),
        "auc_top1_by_position": [
            auc([x[2] for x in accepted if x[0] == p], [x[2] for x in rejected if x[0] == p])
            for p in range(3)
        ],
        "bands": bands,
        "tested": len(sample_list),
    }


def main():
    output, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    errors, result = [], {"runs": [str(r) for r in runs], "cases": [], "pooled": {}}
    try:
        diags = [json.loads((r / "diag.json").read_text()) for r in runs]
        base = diags[0]
        if not base.get("confidence") or not base.get("compact_head"):
            errors.append("first run lacks confidence capture or the compact head")
        for other, run in zip(diags[1:], runs[1:], strict=True):
            for a, b in zip(base["cases"], other["cases"], strict=True):
                for key in ("prompt", "depth", "tokens", "cycles", "histogram"):
                    if a[key] != b[key]:
                        errors.append(f"{run.name} {a['prompt']} MTP{a['depth']}: {key} differs")
                if a["trace"]["levels"] != b["trace"]["levels"]:
                    errors.append(f"{run.name} {a['prompt']} MTP{a['depth']}: confidence differs")
        pooled = {}
        for case in base["cases"]:
            s = samples(case)
            row = {
                "prompt": case["prompt"],
                "depth": case["depth"],
                "input_tokens": case["input_tokens"],
                "budget": case["budget"],
                "histogram": case["histogram"],
                "proposed": case["proposed"],
                "accepted": case["accepted"],
                **summarize(s),
            }
            result["cases"].append(row)
            pooled.setdefault(case["depth"], []).extend(s)
        result["pooled"] = {str(d): summarize(s) for d, s in sorted(pooled.items())}
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    result["status"] = "passed" if not errors else "failed"
    result["errors"] = errors

    def fmt(w):
        return (
            "—"
            if w is None
            else f"{w['rate']:.3f} [{w['low']:.3f};{w['high']:.3f}] ({w['hit']}/{w['n']})"
        )

    for row in result["cases"]:
        print(
            f"{row['prompt']:14s} MTP{row['depth']} p1 {fmt(row['positions'][0])} "
            f"p2|1 {fmt(row['positions'][1])} p3|2 {fmt(row['positions'][2])} "
            f"AUC {row['auc_top1_log_prob']}"
        )
    for depth, s in result["pooled"].items():
        print(
            f"pooled MTP{depth}: p1 {fmt(s['positions'][0])} p2|1 {fmt(s['positions'][1])} "
            f"p3|2 {fmt(s['positions'][2])} AUC top1 {s['auc_top1_log_prob']} "
            f"margin {s['auc_margin']} by position {s['auc_top1_by_position']}"
        )
        print("   bands", [(b["p_top1"], b["n"], b["accept_rate"]) for b in s["bands"]])
    output.write_text(json.dumps(result, indent=2))
    print("\n".join(errors) if errors else "all T0a checks passed")
    print("t0a_status", result["status"])
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
