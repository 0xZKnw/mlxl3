"""Summarize compare_native results.json files: medians, change, per-run values and A/B control drift."""
import json
import sys
from pathlib import Path

for path in sys.argv[1:]:
    r = json.loads((Path(path) / "results.json").read_text())
    print(f"== {path}: status={r['status']} parity={r['parity']}")
    for name, s in r.get("summary", {}).items():
        d = s["decode_tps"]
        print(f"  {name:9s} decode {d['A']:.2f} -> {d['B']:.2f} ({d['change_percent']:+.2f}%)"
              + "".join(f"  {k} {v['change_percent']:+.2f}%" for k, v in s.items() if k != "decode_tps"))
    per = {}
    for p in r.get("passes", []):
        for name, pr in p["prompts"].items():
            vals = [row["stats"]["decode_tps"] for row in pr["runs"]]
            per.setdefault((p["label"], name), []).append(vals)
            print(f"    pass {p['label']} {name:9s} " + " ".join(f"{v:.2f}" for v in vals))
    for (label, name), groups in sorted(per.items()):
        if len(groups) == 2:
            a, b = (sum(g) / len(g) for g in groups)
            print(f"  drift {label} {name}: {100 * (b - a) / a:+.2f}% between its two passes")
