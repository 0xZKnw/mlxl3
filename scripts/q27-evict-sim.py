#!/usr/bin/env python3
"""Offline MTP EVICT calibration for one model from `mtp_cycle_diagnostics` traces.

Input: report written with MLXL3_MTP_EVICT=0, MLXL3_MTP_DIAG_CONFIDENCE=1, depth 3
(every cycle drafts and verifies three levels). A cycle truncated to k levels accepts
min(accepted, k) because greedy verification of level i depends only on levels < i.
Compares, in tokens per AR step with this model's measured costs: fixed widths,
the shipped 35B EVICT tables, and tables refitted on one half of the prompts and
scored on the other half (2-fold cross-validation).
"""

import json
import math
import sys

EDGES = [math.log(p) for p in (0.5, 0.8, 0.95, 0.99)]
SURVIVAL_35B = [[368, 645, 905, 968, 998], [256, 631, 924, 958, 995], [172, 685, 864, 923, 986]]
# 27B costs in AR steps: verify by rows (1 = plain step), draft per drafted width.
VERIFY = [1.00, 1.17, 1.52, 1.80]
DRAFT_PER_LEVEL = 0.075
TAIL = 0.007


def band(lp):
    return sum(lp >= e for e in EDGES)


def cycles(path):
    with open(path) as handle:
        report = json.load(handle)
    out = []
    for case in report["cases"]:
        levels = case["trace"]["levels"]
        it = iter(levels)
        for width, accepted in case["cycles"]:
            if width == 0:
                continue
            tops = [next(it)["top1_log_prob"] for _ in range(width)]
            if width == 3:
                out.append((case["prompt"], [band(t) for t in tops], accepted))
    return out


def evict_width(bands, survival):
    best, best_val = 0, 1.0 / (DRAFT_PER_LEVEL * 3 + VERIFY[0] + TAIL)
    surv, expected = 1.0, 0.0
    for k in range(1, 4):
        surv *= survival[k - 1][bands[k - 1]] / 1000
        expected += surv
        val = (1 + expected) / (DRAFT_PER_LEVEL * 3 + VERIFY[k] + TAIL)
        if val > best_val:
            best, best_val = k, val
    return best


def score(data, width_of):
    tokens = cost = 0.0
    for _, bands, accepted in data:
        k = width_of(bands)
        tokens += min(accepted, k) + 1
        cost += DRAFT_PER_LEVEL * 3 + VERIFY[k] + TAIL
    return tokens / cost


def fit(data):
    table = []
    for level in range(3):
        row = []
        for b in range(5):
            # survival of level given previous levels survived, Laplace smoothing
            reached = [d for d in data if d[2] >= level and d[1][level] == b]
            ok = sum(d[2] > level for d in reached)
            row.append(round(1000 * (ok + 1) / (len(reached) + 2)))
        table.append(row)
    return table


def main(path):
    data = cycles(path)
    prompts = sorted({d[0] for d in data})
    print(f"{len(data)} full-width cycles over {len(prompts)} prompts")
    for k in range(4):
        print(f"fixed k={k}: {score(data, lambda b, k=k: k):.3f}")
    print(f"EVICT 35B tables: {score(data, lambda b: evict_width(b, SURVIVAL_35B)):.3f}")
    full = fit(data)
    print("27B table (all prompts):", full)
    halves = [set(prompts[0::2]), set(prompts[1::2])]
    for train, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        table = fit([d for d in data if d[0] in train])
        held = [d for d in data if d[0] in test]
        ours = score(held, lambda b, table=table: evict_width(b, table))
        shipped = score(held, lambda b: evict_width(b, SURVIVAL_35B))
        best_fixed = max(score(held, lambda b, k=k: k) for k in range(4))
        print(
            f"held-out {sorted(test)}: 27B tables {ours:.3f}, 35B tables {shipped:.3f} ({100 * (ours / shipped - 1):+.1f}%), best fixed {best_fixed:.3f}"
        )


if __name__ == "__main__":
    main(sys.argv[1])
