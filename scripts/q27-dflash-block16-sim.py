#!/usr/bin/env python3
"""Q27D-34: offline scoring of a 16-row DFlash2 mask block against the recorded greedy output.

Input: a `dflash_block16_depth_accuracy` report. For each block the target continuation is the
recorded greedy output after the anchor. Scores the production selector chain (at most 7), the
per-position argmax of the 8-row draft (7) and of the 16-row draft truncated to 7 and complete
(15), then an adaptive policy that pays for 16 rows only when the cumulative draft confidence of
the first seven positions clears a threshold. The threshold is fitted on even prompts and scored
on odd prompts, and the reverse. Block boundaries stay those of the production policy (stated
approximation). Costs are parameters measured elsewhere, in milliseconds per block.
"""

import argparse
import json
import sys
from pathlib import Path


def accepted(chain, target):
    count = 0
    for proposed, actual in zip(chain, target):
        if proposed != actual:
            break
        count += 1
    return count


def rate(blocks, wide, cost8, cost16):
    tokens = 0
    milliseconds = 0.0
    for block, use in zip(blocks, wide):
        tokens += (block["a16"] if use else block["a8"]) + 1
        milliseconds += cost16 if use else cost8
    return 1000.0 * tokens / milliseconds


def best_threshold(blocks, cost8, cost16):
    candidates = sorted({block["confidence7"] for block in blocks})
    best = (rate(blocks, [False] * len(blocks), cost8, cost16), float("inf"))
    for threshold in candidates:
        value = rate(blocks, [b["confidence7"] >= threshold for b in blocks], cost8, cost16)
        if value > best[0]:
            best = (value, threshold)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cost8", type=float, default=178.0, help="ms per 8-row block")
    parser.add_argument("--cost16", type=float, default=203.0, help="ms per 16-row block")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("preserve previous simulation reports")
    report = json.loads(args.profile.read_text())
    if report.get("status") != "complete":
        raise SystemExit("profile is incomplete")
    blocks = []
    for block in report["blocks"]:
        output = report["outputs"][block["prompt"]]
        if output[block["anchor_index"]] != block["anchor"]:
            raise SystemExit("anchor index does not match the recorded output")
        start = block["anchor_index"] + 1
        target = output[start : start + 15]
        if block["proposals"] != 7 or len(target) < 15 or len(block["marginal16"]) != 15:
            continue
        if accepted(block["chain"], target) != block["accepted"]:
            raise SystemExit("recorded acceptance disagrees with the recorded output")
        blocks.append(
            {
                "prompt": block["prompt"],
                "selector7": block["accepted"],
                "marginal8": accepted(block["marginal8"], target),
                "marginal16_7": accepted(block["marginal16"][:7], target),
                "marginal16": accepted(block["marginal16"], target),
                "same_first7": block["marginal16"][:7] == block["marginal8"],
                "confidence7": sum(block["confidence16"][:7]),
            }
        )
    if not blocks:
        raise SystemExit("no complete block")
    count = len(blocks)

    def mean(key):
        return sum(block[key] for block in blocks) / count + 1.0

    survival = []
    for depth in range(1, 16):
        reached = sum(block["marginal16"] >= depth - 1 for block in blocks)
        passed = sum(block["marginal16"] >= depth for block in blocks)
        survival.append({"depth": depth, "reached": reached, "accepted": passed})
    # The wide policy keeps the selector chain for the first seven positions when it survives
    # them all, then continues with the 16-row argmax; otherwise the block ends as today.
    for block in blocks:
        block["a8"] = block["selector7"]
        block["a16"] = max(block["selector7"], block["marginal16"])
    always8 = rate(blocks, [False] * count, args.cost8, args.cost16)
    always16 = rate(blocks, [True] * count, args.cost8, args.cost16)
    oracle = rate(blocks, [b["a16"] > b["a8"] for b in blocks], args.cost8, args.cost16)
    halves = [[b for b in blocks if b["prompt"] % 2 == p] for p in (0, 1)]
    held_out = []
    for fit, test in ((halves[0], halves[1]), (halves[1], halves[0])):
        _, threshold = best_threshold(fit, args.cost8, args.cost16)
        wide = [b["confidence7"] >= threshold for b in test]
        held_out.append(
            {
                "threshold": threshold,
                "wide_share": sum(wide) / len(test),
                "tokens_per_second": rate(test, wide, args.cost8, args.cost16),
                "baseline_tokens_per_second": rate(
                    test, [False] * len(test), args.cost8, args.cost16
                ),
            }
        )
    summary = {
        "status": "complete",
        "profile": str(args.profile),
        "blocks": count,
        "note": "production block boundaries; target = recorded greedy output",
        "tokens_per_block": {
            "selector_chain_7": mean("selector7"),
            "marginal_8rows_7": mean("marginal8"),
            "marginal_16rows_first7": mean("marginal16_7"),
            "marginal_16rows_15": mean("marginal16"),
            "selector7_then_16rows": sum(b["a16"] for b in blocks) / count + 1.0,
        },
        "first7_identical_share": sum(b["same_first7"] for b in blocks) / count,
        "blocks_accepting_all_7": sum(b["selector7"] == 7 for b in blocks),
        "survival_16rows": survival,
        "cost_ms": {"rows8": args.cost8, "rows16": args.cost16},
        "tokens_per_second": {
            "always_8": always8,
            "always_16": always16,
            "oracle_choice": oracle,
            "adaptive_held_out": held_out,
        },
    }
    args.out.write_text(json.dumps(summary, indent=2))
    json.dump(summary, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
