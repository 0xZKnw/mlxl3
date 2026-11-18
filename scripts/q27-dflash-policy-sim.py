#!/usr/bin/env python3
"""Offline DFlash2 width/confidence policy simulation on a recorded block profile.

Input: the JSON written by `tests::dflash_block_profile_and_confidence` (7 proposals
per block, greedy target, per-position draft log-probabilities). A truncated block of
width k accepts min(accepted, k) proposals because the greedy target check of
position i depends only on positions < i. Costs come from measured verify passes
(AR-step units, M = k + 1 rows) plus the measured draft cost per block. Prints
tokens per AR step for fixed widths and confidence-band policies, with a 2-fold
cross-validation over prompts (thresholds fitted on one half, scored on the other).
"""

import json
import math
import statistics
import sys

VERIFY = {1: 1.00, 2: 1.17, 3: 1.52, 4: 1.80, 5: 2.73, 6: 3.10, 7: 3.37, 8: 3.68}
# Draft logits through the target K3 head for k rows (kernel lab Q27D-05, ms).
HEAD_MS = {1: 5.15, 2: 5.94, 3: 8.14, 4: 9.83}


def load(path):
    with open(path) as handle:
        report = json.load(handle)
    if report.get("status") != "complete" or not report["blocks"]:
        raise SystemExit("incomplete profile")
    return report["blocks"]


def score(blocks, width_of, draft_steps):
    """`draft_steps(k)`: draft network + head for k rows + selector, in AR steps."""
    tokens = cost = 0.0
    for b in blocks:
        k = width_of(b)
        accepted = min(b["accepted"], k)
        tokens += accepted + 1
        cost += VERIFY[k + 1] + draft_steps(k)
    return tokens / cost


def band_policy(threshold, cumulative):
    def width(b):
        k, run = 0, 0.0
        for c in b["confidence"][: b["proposals"]]:
            run = run + c["chosen"] if cumulative else c["chosen"]
            if run < threshold:
                break
            k += 1
        return max(1, min(k, 4))

    return width


def main(path):
    blocks = [b for b in load(path) if b["proposals"] == 7]
    ar_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 106.0
    network_ms = statistics.median(b["draft_ms"] for b in blocks)
    select_ms = max(0.0, statistics.median(b["head_select_ms"] for b in blocks) - 9.83 - 8.14)
    draft_steps = lambda k: (network_ms + HEAD_MS[k] + select_ms) / ar_ms
    print(
        f"blocks {len(blocks)}  AR step {ar_ms} ms  draft network {network_ms:.1f} ms; per k: "
        + ", ".join(f"k={k} {draft_steps(k):.3f} step" for k in range(1, 5))
    )
    for key in ("draft_ms", "head_select_ms", "verify_ms", "commit_ms"):
        print(f"  {key}: median {statistics.median(b[key] for b in blocks):.1f} ms")
    survival = [sum(b["accepted"] > i for b in blocks) / len(blocks) for i in range(7)]
    print("  survival by position:", [round(x, 3) for x in survival])
    for k in range(1, 5):
        print(f"fixed k={k}: {score(blocks, lambda b, k=k: k, draft_steps):.3f} tokens/step")
    prompts = sorted({b["prompt"] for b in blocks})
    halves = [set(prompts[0::2]), set(prompts[1::2])]
    grid = [math.log(p) for p in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)]
    for cumulative in (False, True):
        totals = []
        for train, test in ((halves[0], halves[1]), (halves[1], halves[0])):
            fit = [b for b in blocks if b["prompt"] in train]
            held = [b for b in blocks if b["prompt"] in test]
            best = max(grid, key=lambda t: score(fit, band_policy(t, cumulative), draft_steps))
            base = max(range(1, 5), key=lambda k: score(fit, lambda b, k=k: k, draft_steps))
            totals.append(
                (
                    score(held, band_policy(best, cumulative), draft_steps),
                    score(held, lambda b, k=base: k, draft_steps),
                    math.exp(best),
                    base,
                )
            )
        label = "cumulative" if cumulative else "per-position"
        for policy, fixed, t, k in totals:
            print(
                f"{label} band p>={t:.2f}: held-out {policy:.3f} vs best fixed k={k} {fixed:.3f} ({100 * (policy / fixed - 1):+.1f}%)"
            )
    oracle = sum(min(b["accepted"], 4) + 1 for b in blocks) / sum(
        VERIFY[min(max(b["accepted"], 1), 4) + 1] + draft_steps(min(max(b["accepted"], 1), 4))
        for b in blocks
    )
    print(f"oracle (verify exactly accepted, width 1..4): {oracle:.3f} tokens/step")


if __name__ == "__main__":
    main(sys.argv[1])
