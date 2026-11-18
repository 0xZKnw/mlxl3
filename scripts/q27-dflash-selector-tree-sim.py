#!/usr/bin/env python3
"""Q27D-36: offline best-first DFlash2 trees on the recorded selector lattice.

Input: a `dflash_selector_lattice_dump` report (per block: 16 candidates, their unary scores and
the 16 x 16 edge scores from the previous position's candidates, the production chain, and the
recorded greedy output). A node at depth p holds a candidate index of position p; its children
are the 16 candidates of position p + 1, scored by log-softmax((unary + w * edge[parent]) / tau).
Trees grow best first on the cumulative score. A tree accepts the longest root path whose tokens
equal the recorded greedy continuation. Block boundaries stay those of the production policy.
"""

import argparse
import heapq
import json
import math
import sys
from pathlib import Path

CANDIDATES = 16


def chain_from_lattice(block, scale):
    positions = block["proposals"]
    unary, edges, candidates = block["unary"], block["edges"], block["candidates"]
    previous, tokens = 0, []
    for p in range(positions):
        best = 0
        for c in range(1, CANDIDATES):
            score = unary[p * 16 + c] + scale * edges[(p * 16 + previous) * 16 + c]
            current = unary[p * 16 + best] + scale * edges[(p * 16 + previous) * 16 + best]
            if score > current:
                best = c
        previous = best
        tokens.append(candidates[p * 16 + best])
    return tokens


def children(block, depth, parent, scale, tau):
    unary, edges = block["unary"], block["edges"]
    scores = [
        (unary[depth * 16 + c] + scale * edges[(depth * 16 + parent) * 16 + c]) / tau
        for c in range(CANDIDATES)
    ]
    top = max(scores)
    total = math.log(sum(math.exp(s - top) for s in scores)) + top
    return [s - total for s in scores]


def build_tree(block, nodes, scale, tau):
    """Return a dict path(tuple of candidate indices) -> True for the chosen nodes."""
    positions = block["proposals"]
    chosen = set()
    frontier = []
    for c, score in enumerate(children(block, 0, 0, scale, tau)):
        heapq.heappush(frontier, (-score, (c,)))
    while frontier and len(chosen) < nodes:
        negative, path = heapq.heappop(frontier)
        chosen.add(path)
        depth = len(path)
        if depth < positions:
            for c, score in enumerate(children(block, depth, path[-1], scale, tau)):
                heapq.heappush(frontier, (negative - score, path + (c,)))
    return chosen


def tree_accept(block, chosen, target):
    candidates = block["candidates"]
    path = ()
    accepted = 0
    for depth, token in enumerate(target[: block["proposals"]]):
        match = None
        for c in range(CANDIDATES):
            if candidates[depth * 16 + c] == token and path + (c,) in chosen:
                match = c
                break
        if match is None:
            break
        path = path + (match,)
        accepted += 1
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cost8", type=float, default=178.0)
    parser.add_argument("--cost12", type=float, default=190.0)
    parser.add_argument("--cost16", type=float, default=203.0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("preserve previous simulation reports")
    report = json.loads(args.profile.read_text())
    if report.get("status") != "complete":
        raise SystemExit("profile is incomplete")
    scale0 = report["edge_scale"]
    blocks, targets = [], []
    mismatched_chains = 0
    for block in report["blocks"]:
        output = report["outputs"][block["prompt"]]
        if output[block["anchor_index"]] != block["anchor"]:
            raise SystemExit("anchor index does not match the recorded output")
        start = block["anchor_index"] + 1
        target = output[start : start + block["proposals"]]
        if chain_from_lattice(block, scale0) != block["chain"]:
            mismatched_chains += 1
        if block["proposals"] != 7 or len(target) < 7:
            continue
        blocks.append(block)
        targets.append(target)
    if mismatched_chains:
        raise SystemExit(f"{mismatched_chains} chains differ from the lattice recomputation")
    count = len(blocks)
    base = sum(block["accepted"] + 1 for block in blocks) / count
    results = []
    for nodes, cost in ((7, args.cost8), (11, args.cost12), (15, args.cost16)):
        for scale in (0.25, 0.6, 1.0):
            for tau in (1.0, 1.5):
                tokens = sum(
                    tree_accept(block, build_tree(block, nodes, scale, tau), target) + 1
                    for block, target in zip(blocks, targets)
                ) / count
                results.append(
                    {
                        "nodes": nodes,
                        "rows": nodes + 1,
                        "edge_scale": scale,
                        "tau": tau,
                        "tokens_per_block": tokens,
                        "over_chain": tokens / base,
                        "tokens_per_second": 1000.0 * tokens / cost,
                    }
                )
    results.sort(key=lambda row: -row["tokens_per_second"])
    summary = {
        "status": "complete",
        "profile": str(args.profile),
        "blocks": count,
        "chain_recomputed_exactly": True,
        "chain_tokens_per_block": base,
        "chain_tokens_per_second": 1000.0 * base / args.cost8,
        "cost_ms": {"rows8": args.cost8, "rows12": args.cost12, "rows16": args.cost16},
        "trees": results,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    json.dump(summary, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
