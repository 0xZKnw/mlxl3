#!/usr/bin/env python3
"""Q27D-43: offline best-first DFlash2 trees with a leaf (path) budget on the recorded lattice.

Q27D-41 measured that verification rows are nearly free while every root-to-leaf path costs
one more GDN pass. This replays the recorded selector lattice (`dflash_selector_lattice_dump`,
greedy outputs) with best-first trees whose number of leaves is capped, and scores them with a
verification cost model fitted on the Q27D-41 micro-benchmark. Parameters are chosen on the
even prompts and evaluated on the odd prompts (and the reverse).
"""

import argparse
import heapq
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("tree_sim", HERE / "q27-dflash-selector-tree-sim.py")
_sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sim)

DRAFT_MS = 22.4  # Q27D-38: draft + lattice readback per block (median)
COMMIT_MS = 0.6  # Q27D-38


def fit_cost(path_cost):
    """Least squares ms = a + b * rows + c * gdn_steps + d * paths on the Q27D-41 cases."""
    # Sum of root-to-leaf path lengths (GDN steps) of the Q27D-41 synthetic shapes.
    steps_of = {"8 rows / 1 path": 8, "12 rows / 2 paths": 13, "12 rows / 5 paths": 22,
                "15 rows / 2 paths": 16, "16 rows / 9 paths": 46}
    rows = []
    for case in json.loads(path_cost.read_text())["cases"]:
        steps = steps_of[case["shape"]]
        rows.append(([1.0, case["rows"], steps, case["paths"]], case["median_ms"]))
    n = 4
    ata = [[sum(x[i] * x[j] for x, _ in rows) for j in range(n)] for i in range(n)]
    atb = [sum(x[i] * y for x, y in rows) for i in range(n)]
    # Gauss-Jordan on the 4x4 normal equations.
    m = [ata[i] + [atb[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            raise SystemExit("singular cost fit")
        for r in range(n):
            if r != col:
                factor = m[r][col] / m[col][col]
                m[r] = [a - factor * b for a, b in zip(m[r], m[col])]
    coef = [m[i][n] / m[i][i] for i in range(n)]
    residuals = [y - sum(c * v for c, v in zip(coef, x)) for x, y in rows]
    return coef, residuals


def build_tree(block, nodes, leaves, scale, tau):
    """Best-first tree; a node that would open a new path beyond `leaves` is skipped."""
    positions = block["proposals"]
    chosen = set()
    child_count = {(): 0}
    leaf_count = 0
    frontier = []
    for c, score in enumerate(_sim.children(block, 0, 0, scale, tau)):
        heapq.heappush(frontier, (-score, (c,)))
    while frontier and len(chosen) < nodes:
        negative, path = heapq.heappop(frontier)
        parent = path[:-1]
        # Extending a leaf keeps the path count; any other child opens a new path.
        delta = 0 if parent != () and child_count[parent] == 0 else 1
        if leaves is not None and leaf_count + delta > leaves:
            continue
        new_leaves = leaf_count + delta
        chosen.add(path)
        child_count[parent] = child_count.get(parent, 0) + 1
        child_count[path] = 0
        leaf_count = new_leaves
        if len(path) < positions:
            for c, score in enumerate(_sim.children(block, len(path), path[-1], scale, tau)):
                heapq.heappush(frontier, (negative - score, path + (c,)))
    leaves_set = [p for p in chosen if child_count[p] == 0]
    steps = sum(len(p) + 1 for p in leaves_set)
    return chosen, len(leaves_set), steps


def evaluate(blocks, targets, nodes, leaves, coef, scale, tau):
    tokens = ms = paths_total = 0.0
    for block, target in zip(blocks, targets):
        chosen, paths, steps = build_tree(block, nodes, leaves, scale, tau)
        rows = len(chosen) + 1
        verify = coef[0] + coef[1] * rows + coef[2] * steps + coef[3] * paths
        tokens += _sim.tree_accept(block, chosen, target) + 1
        ms += DRAFT_MS + verify + COMMIT_MS
        paths_total += paths
    count = len(blocks)
    return {"tokens_per_block": tokens / count, "ms_per_block": ms / count,
            "tokens_per_second": 1000.0 * tokens / ms, "paths": paths_total / count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lattice", type=Path)
    parser.add_argument("path_cost", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("preserve previous simulation reports")
    report = json.loads(args.lattice.read_text())
    if report.get("status") != "complete":
        raise SystemExit("lattice dump is incomplete")
    coef, residuals = fit_cost(args.path_cost)
    groups = {0: ([], []), 1: ([], [])}
    for block in report["blocks"]:
        output = report["outputs"][block["prompt"]]
        if output[block["anchor_index"]] != block["anchor"]:
            raise SystemExit("anchor index does not match the recorded output")
        start = block["anchor_index"] + 1
        target = output[start : start + block["proposals"]]
        if block["proposals"] != 7 or len(target) < 7:
            continue
        group = groups[block["prompt"] % 2]
        group[0].append(block)
        group[1].append(target)
    grid = [(n, leaves) for n in (7, 9, 11, 13, 15) for leaves in (1, 2, 3, 4, 5, None)]
    scale, tau = 0.6, 1.0  # production scoring (Q27D-36/37)
    results = []
    for n, leaves in grid:
        row = {"nodes": n, "leaves": leaves}
        for name, (blocks, targets) in (("even", groups[0]), ("odd", groups[1])):
            row[name] = evaluate(blocks, targets, n, leaves, coef, scale, tau)
        results.append(row)
    production = next(r for r in results if r["nodes"] == 11 and r["leaves"] is None)
    # Out of sample: choose on one half, evaluate on the other.
    selection = {}
    for fit_half, test_half in (("even", "odd"), ("odd", "even")):
        best = max(results, key=lambda r: r[fit_half]["tokens_per_second"])
        selection[f"chosen_on_{fit_half}"] = {
            "nodes": best["nodes"], "leaves": best["leaves"],
            "test_gain_vs_production": best[test_half]["tokens_per_second"]
            / production[test_half]["tokens_per_second"] - 1,
        }
    summary = {"status": "complete", "lattice": str(args.lattice), "path_cost": str(args.path_cost),
               "blocks": {"even": len(groups[0][0]), "odd": len(groups[1][0])},
               "cost_fit_ms": {"intercept": coef[0], "per_row": coef[1], "per_gdn_step": coef[2],
                               "per_path": coef[3], "residuals": residuals},
               "draft_ms": DRAFT_MS, "commit_ms": COMMIT_MS, "selection": selection,
               "production": production, "trees": results}
    args.out.write_text(json.dumps(summary, indent=2))
    for r in results:
        print(f"n={r['nodes']:2d} leaves={str(r['leaves']):4s} even {r['even']['tokens_per_block']:.3f} tok/blk "
              f"{r['even']['tokens_per_second']:.2f}/s  odd {r['odd']['tokens_per_block']:.3f} "
              f"{r['odd']['tokens_per_second']:.2f}/s paths {r['even']['paths']:.1f}")
    print(json.dumps(selection, indent=2))
    print("cost fit", summary["cost_fit_ms"])


if __name__ == "__main__":
    sys.exit(main())
