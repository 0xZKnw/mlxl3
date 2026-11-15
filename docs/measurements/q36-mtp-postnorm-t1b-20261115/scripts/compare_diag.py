"""Q36-MTP-POSTNORM-T1B: compare a diagnostic report with a reference report, case by case.

Usage: python -I compare_diag.py <actual diag.json> <reference diag.json> <same|differ> <variant>
`same`: every case has identical tokens, cycles, histogram, depth_blocks, accepted and proposed.
`differ`: tokens identical (exact verification), but at least one case has different cycles.
The variant must match `hidden_variant` of the actual report. Fails closed on any missing data.
"""

import json
import sys
from pathlib import Path

FIELDS = ("tokens", "cycles", "histogram", "depth_blocks", "accepted", "proposed")


def main():
    actual_path, reference_path, mode, variant = sys.argv[1:5]
    if mode not in ("same", "differ"):
        print("mode must be same or differ")
        return 2
    errors = []
    try:
        actual = json.loads(Path(actual_path).read_text())
        reference = json.loads(Path(reference_path).read_text())
        if actual.get("hidden_variant") != variant:
            errors.append(f"hidden_variant {actual.get('hidden_variant')!r} != {variant!r}")
        a = {(c["prompt"], c["depth"]): c for c in actual["cases"]}
        r = {(c["prompt"], c["depth"]): c for c in reference["cases"]}
        if len(a) != 12 or set(a) != set(r):
            errors.append(f"case sets differ or not 12 cases: {len(a)} vs {len(r)}")
        differing = 0
        for key in sorted(set(a) & set(r)):
            if not a[key]["tokens"] or len(a[key]["tokens"]) != len(r[key]["tokens"]):
                errors.append(f"{key}: empty or wrong-length tokens")
            if a[key]["tokens"] != r[key]["tokens"]:
                errors.append(f"{key}: tokens differ")
            changed = [f for f in FIELDS[1:] if a[key][f] != r[key][f]]
            if changed:
                differing += 1
                if mode == "same":
                    errors.append(f"{key}: {', '.join(changed)} differ")
        if mode == "differ" and differing == 0:
            errors.append("no case differs: the mutation or variant was not detected")
        print(f"{actual_path}: {len(a)} cases, {differing} with different cycles/counters")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    print("\n".join(errors) if errors else f"compare {mode} passed")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
