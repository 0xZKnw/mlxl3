"""Q36-MTP-POSTNORM-T1B: compact (79 616 IDs) versus full target head, pre and post wiring.

Pre-registered criterion, per depth (pooled over 4 prompts):
  gap(full - compact accepted, post) <= gap(full - compact accepted, pre) + 1 % of compact post proposed.
Usage: python -I analyze_compact.py <oracles dir> <output.json>. Fails closed.
"""

import json
import sys
from pathlib import Path

RUNS = {"compact-pre": "default", "compact-post": "post1", "full-pre": "full-pre", "full-post": "full-post"}
EXPECT = {
    "compact-pre": ("pre", True),
    "compact-post": ("post", True),
    "full-pre": ("pre", False),
    "full-post": ("post", False),
}


def main():
    root, output = Path(sys.argv[1]), Path(sys.argv[2])
    errors, result = [], {"depths": {}}
    try:
        runs = {}
        for label, directory in RUNS.items():
            diag = json.loads((root / directory / "diag.json").read_text())
            variant, compact = EXPECT[label]
            if diag.get("hidden_variant") != variant or diag.get("compact_head") is not compact:
                errors.append(f"{label}: variant/compact {diag.get('hidden_variant')}/{diag.get('compact_head')}")
            runs[label] = {(c["prompt"], c["depth"]): c for c in diag["cases"]}
        keys = sorted(runs["compact-pre"])
        if len(keys) != 12 or any(set(r) != set(keys) for r in runs.values()):
            errors.append("case sets differ or not 12 cases")
        for key in keys:
            reference = runs["compact-pre"][key]["tokens"]
            if not reference:
                errors.append(f"{key}: empty tokens")
            for label, r in runs.items():
                if r[key]["tokens"] != reference:
                    errors.append(f"{label} {key}: tokens differ (exact verification broken)")
        for depth in (1, 2, 3):
            pooled = {}
            for label, r in runs.items():
                acc = sum(r[k]["accepted"] for k in keys if k[1] == depth)
                prop = sum(r[k]["proposed"] for k in keys if k[1] == depth)
                cycles = sum(len(r[k]["cycles"]) for k in keys if k[1] == depth)
                pooled[label] = {"accepted": acc, "proposed": prop, "cycles": cycles,
                                 "rate": acc / prop if prop else None}
            gap_pre = pooled["full-pre"]["accepted"] - pooled["compact-pre"]["accepted"]
            gap_post = pooled["full-post"]["accepted"] - pooled["compact-post"]["accepted"]
            margin = 0.01 * pooled["compact-post"]["proposed"]
            passed = gap_post <= gap_pre + margin
            result["depths"][f"MTP{depth}"] = {**pooled, "gap_pre": gap_pre, "gap_post": gap_post,
                                              "margin": margin, "passed": passed}
            if not passed:
                errors.append(f"MTP{depth}: compact loses more under post ({gap_post} > {gap_pre} + {margin:.1f})")
            print(f"MTP{depth}: " + " | ".join(
                f"{k} {v['accepted']}/{v['proposed']} cycles {v['cycles']}" for k, v in pooled.items())
                + f" | gap pre {gap_pre} post {gap_post} margin {margin:.1f} -> {'pass' if passed else 'FAIL'}")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    result["status"] = "passed" if not errors else "failed"
    result["errors"] = errors
    output.write_text(json.dumps(result, indent=2))
    print("\n".join(errors) if errors else "compact requalification passed")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
