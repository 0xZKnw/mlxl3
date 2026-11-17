#!/usr/bin/env python3
"""Summarize a d2.py report: decode rate and MTP acceptance per label, case and arm."""
import json
import statistics as st
import sys

report = json.load(open(sys.argv[1]))
print("status", report["status"], "runs", len(report["runs"]))
table = {}
for run in report["runs"]:
    for row in run["rows"]:
        table.setdefault((run["label"], row["case"], row["arm"]), []).append(row["stats"])
labels = sorted({key[0] for key in table})
arms = sorted({key[2] for key in table}, key=lambda arm: (arm != "G", arm))
cases = report["cases"]
out = {}


def rate(stats):
    return [value["decode_tps"] for value in stats]


def accepted(stats):
    proposed = sum(value.get("mtp_proposed_tokens") or 0 for value in stats)
    if not proposed:
        return None
    return {
        "accepted": sum(value.get("mtp_accepted_tokens") or 0 for value in stats),
        "proposed": proposed,
        "blocks": sum(value.get("mtp_blocks") or 0 for value in stats),
        "tokens": sum(value["generated_tokens"] for value in stats),
    }


for label in labels:
    print(f"-- {label}")
    out[label] = {}
    for case in cases:
        base = st.fmean(rate(table[(label, case, "G")]))
        cells = []
        for arm in arms:
            values = rate(table[(label, case, arm)])
            cells.append(
                f"{arm} {st.fmean(values):6.2f} [{min(values):.1f}-{max(values):.1f}]"
                f" ({(st.fmean(values) / base - 1) * 100:+.1f}%)"
            )
        print(f"  {case:6s} " + "  ".join(cells))
    for arm in arms:
        stats = [value for case in cases for value in table[(label, case, arm)]]
        values = rate(stats)
        ratios = [
            st.fmean(rate(table[(label, case, arm)])) / st.fmean(rate(table[(label, case, "G")])) - 1
            for case in cases
        ]
        entry = {
            "n": len(values),
            "mean_tps": st.fmean(values),
            "median_tps": st.median(values),
            "vs_G_mean_of_cases": st.fmean(ratios),
            "vs_G_min": min(ratios),
            "vs_G_max": max(ratios),
        }
        mtp = accepted(stats)
        if mtp:
            entry["mtp_acceptance"] = mtp["accepted"] / mtp["proposed"]
            entry["tokens_per_block"] = mtp["tokens"] / mtp["blocks"]
            entry.update({f"mtp_{k}": v for k, v in mtp.items()})
        out[label][arm] = entry
        print(f"  arm {arm}:", json.dumps({k: round(v, 4) for k, v in entry.items()}))
json.dump(out, open(sys.argv[1].replace("results.json", "summary.json"), "w"), indent=2)
