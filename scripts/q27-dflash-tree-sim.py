#!/usr/bin/env python3
"""Q27D-32: offline simulation of small DFlash2 verification trees on a recorded profile.

Input: a `dflash_block_profile_and_confidence` report with `chain`, `top_ids`, `anchor_index`
per block and the greedy `outputs` per prompt. For each block the target continuation is the
recorded greedy output after the anchor, so any tree is scored without the GPU: a tree accepts
the longest path whose tokens equal the continuation, then gains the target's bonus token.
Block boundaries stay those of the chain policy (an approximation stated in the report).
"""

import argparse
import json
import sys
from pathlib import Path


def chain_accept(chain, target):
    accepted = 0
    for proposed, actual in zip(chain, target):
        if proposed != actual:
            break
        accepted += 1
    return accepted


def alternatives(top_ids, chain, depth, count):
    picks = [token for token in top_ids[depth] if token != chain[depth]]
    return picks[:count]


def tree_accept(block, target, plan):
    """plan: list of (depth, alternatives, continuation length)."""
    chain = block["chain"]
    k = min(len(chain), len(target))
    accepted = chain_accept(chain[:k], target[:k])
    best = accepted
    if accepted >= k:
        return best, 0
    for depth, count, continuation in plan:
        if depth != accepted or depth >= k:
            continue
        for token in alternatives(block["top_ids"], chain, depth, count):
            if token != target[depth]:
                continue
            gained = depth + 1
            for offset in range(1, continuation + 1):
                index = depth + offset
                if index >= k or chain[index] != target[index]:
                    break
                gained += 1
            best = max(best, gained)
    return best, best - accepted


def nodes(plan):
    return sum(count * (1 + continuation) for _, count, continuation in plan)


PLANS = {
    "chain": [],
    "sib1_d0-6": [(d, 1, 0) for d in range(7)],
    "sib2_d0-3": [(d, 2, 0) for d in range(4)],
    "sib1_d0-3+sib2_d0-1": [(0, 2, 0), (1, 2, 0), (2, 1, 0), (3, 1, 0)],
    "branch_d0_L6+sib_d1": [(0, 1, 6), (1, 1, 0)],
    "branch_d0_L3+branch_d1_L3": [(0, 1, 3), (1, 1, 2)],
    "sib3_d0+sib2_d1+sib1_d2-4": [(0, 3, 0), (1, 2, 0), (2, 1, 0), (3, 1, 0), (4, 1, 0)],
    "branch_d0_L1x2+d1_L1+d2": [(0, 2, 1), (1, 1, 1), (2, 1, 0)],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("preserve previous simulation reports")
    report = json.loads(args.profile.read_text())
    if report.get("status") != "complete" or "outputs" not in report:
        raise SystemExit("profile is incomplete or lacks recorded outputs")
    outputs = report["outputs"]
    blocks = report["blocks"]
    usable = []
    for block in blocks:
        if "top_ids" not in block or len(block["top_ids"]) != len(block["chain"]):
            raise SystemExit("block lacks top_ids")
        output = outputs[block["prompt"]]
        start = block["anchor_index"] + 1
        target = output[start : start + len(block["chain"])]
        if output[block["anchor_index"]] != block["anchor"]:
            raise SystemExit("anchor index does not match the recorded output")
        if chain_accept(block["chain"], target) != block["accepted"] and len(target) == len(
            block["chain"]
        ):
            raise SystemExit("recorded acceptance disagrees with the recorded output")
        if len(target) == len(block["chain"]):
            usable.append((block, target))
    if not usable:
        raise SystemExit("no complete block")
    results = {}
    base = None
    for name, plan in PLANS.items():
        tokens = 0
        rescued = 0
        for block, target in usable:
            accepted, gain = tree_accept(block, target, plan)
            tokens += accepted + 1
            rescued += gain > 0
        mean = tokens / len(usable)
        if name == "chain":
            base = mean
        results[name] = {
            "extra_nodes": nodes(plan),
            "rows": 1 + 7 + nodes(plan),
            "tokens_per_block": mean,
            "over_chain": mean / base,
            "rescued_blocks": rescued,
        }
    first_failure = {}
    second_hit = {}
    for block, target in usable:
        accepted = block["accepted"]
        if accepted < len(block["chain"]):
            first_failure[accepted] = first_failure.get(accepted, 0) + 1
            ranks = block["top_ids"][accepted]
            rank = ranks.index(target[accepted]) if target[accepted] in ranks else -1
            second_hit[str(rank)] = second_hit.get(str(rank), 0) + 1
    summary = {
        "status": "complete",
        "profile": str(args.profile),
        "blocks": len(usable),
        "note": "block boundaries of the chain policy; target = recorded greedy output",
        "plans": results,
        "first_failure_depth": dict(sorted(first_failure.items())),
        "target_rank_in_draft_top4_at_failure": second_hit,
    }
    args.out.write_text(json.dumps(summary, indent=2))
    json.dump(summary, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
