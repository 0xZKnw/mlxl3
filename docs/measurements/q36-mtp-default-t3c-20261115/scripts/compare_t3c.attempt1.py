"""Q36-MTP-DEFAULT-T3C O1: the default (variables unset) diagnostic must equal a reference exactly.

Usage: python -I compare_t3c.py <actual diag.json> <reference diag.json> <variant> <evict 0|1>
Every case (prompt, depth) must match on tokens, cycles, histogram, depth_blocks, accepted, proposed
and the whole trace (levels and EVICT records). Fails closed on missing or empty data.
"""

import json
import sys
from pathlib import Path

FIELDS = ("tokens", "cycles", "histogram", "depth_blocks", "accepted", "proposed", "trace")


def main():
    actual_path, reference_path, variant, evict = sys.argv[1:5]
    errors = []
    try:
        a_doc = json.loads(Path(actual_path).read_text())
        r_doc = json.loads(Path(reference_path).read_text())
        if a_doc.get("hidden_variant") != variant or r_doc.get("hidden_variant") != variant:
            errors.append(f"variants {a_doc.get('hidden_variant')!r} / {r_doc.get('hidden_variant')!r} != {variant!r}")
        if a_doc.get("evict") is not (evict == "1") or r_doc.get("evict") is not (evict == "1"):
            errors.append(f"evict flags {a_doc.get('evict')!r} / {r_doc.get('evict')!r} != {evict}")
        if not a_doc.get("compact_head"):
            errors.append("compact head not active")
        a = {(c["prompt"], c["depth"]): c for c in a_doc["cases"]}
        r = {(c["prompt"], c["depth"]): c for c in r_doc["cases"]}
        if len(a) != 12 or set(a) != set(r):
            errors.append(f"case sets differ or not 12 cases: {len(a)} vs {len(r)}")
        evictions = 0
        for key in sorted(set(a) & set(r)):
            x, y = a[key], r[key]
            if not x["tokens"]:
                errors.append(f"{key}: no tokens")
            changed = [f for f in FIELDS if x.get(f) != y.get(f)]
            if changed:
                errors.append(f"{key}: {', '.join(changed)} differ")
            evictions += len((x.get("trace") or {}).get("evictions", []))
        if evict == "1" and evictions == 0:
            errors.append("no EVICT records: EVICT not exercised")
        print(f"{actual_path}: {len(a)} cases, {evictions} EVICT records")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    print("\n".join(errors) if errors else "compare exact passed")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
