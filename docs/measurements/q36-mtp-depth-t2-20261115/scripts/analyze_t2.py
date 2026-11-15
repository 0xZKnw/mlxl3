"""Q36-MTP-DEPTH-T2 analysis. Fails closed on any missing campaign, divergence or wrong protocol.

Usage: python -I analyze_t2.py <runs dir> <output.json>
"""

import json
import statistics
import sys
from pathlib import Path

DOCS = ("b-code", "c-prose-fr", "d-doc-summary")
ARMS = ("d2", "d3", "ada")
ORDERS = ("BAAB", "ABBA")
TIE = 1.5


def check(runs, doc, arm, order, errors):
    name = f"{doc}-{arm}-{order}"
    path = runs / name / "results.json"
    if not path.exists():
        errors.append(f"{name}: missing results.json")
        return None
    r = json.loads(path.read_text())
    if r.get("status") != "complete" or r.get("parity") is not True:
        errors.append(f"{name}: status {r.get('status')} parity {r.get('parity')}")
        return None
    want_b = {"MLXL3_MTP_POST_NORM": "1"}
    if arm == "ada":
        want_b["MLXL3_MTP_ADAPTIVE"] = "1"
    env = {k: v["environment"] for k, v in r["binaries"].items()}
    if env != {"A": {"MLXL3_MTP_POST_NORM": "1"}, "B": want_b}:
        errors.append(f"{name}: environment {env}")
    if r["binaries"]["A"]["sha256"] != r["binaries"]["B"]["sha256"]:
        errors.append(f"{name}: binaries differ")
    depth_b = {"d2": 2, "d3": 3, "ada": 1}[arm]
    if [p["label"] for p in r["passes"]] != list(order):
        errors.append(f"{name}: order")
    for p in r["passes"]:
        want = 1 if p["label"] == "A" else depth_b
        if p["mtp_depth"] != want:
            errors.append(f"{name}: pass {p['index']} depth {p['mtp_depth']} != {want}")
        if "mtppostnorm-v1=1" not in (p["ready"].get("mtp_tuning_key") or ""):
            errors.append(f"{name}: pass {p['index']} tuning key lacks post-norm")
    out = {"prompts": {}}
    for prompt in r["summary"]:
        sig, blocks, per_pass = {}, {}, []
        for p in r["passes"]:
            for run in p["prompts"][prompt]["runs"]:
                s = run["stats"]
                if s["generated_tokens"] != 256:
                    errors.append(f"{name}: {prompt} generated {s['generated_tokens']}")
                sig.setdefault(p["label"], set()).add(
                    (run["token_hash"], s["mtp_accepted_tokens"], s["mtp_proposed_tokens"], s["mtp_blocks"])
                )
                blocks.setdefault(p["label"], set()).add(tuple(s["mtp_depth_blocks"]))
            per_pass.append((p["label"], p["prompts"][prompt]["medians"]["decode_tps"]))
        hashes = {h for v in sig.values() for (h, *_rest) in v}
        if len(hashes) != 1:
            errors.append(f"{name}: {prompt} token hashes {hashes}")
        if arm != "ada" and any(len(v) != 1 for v in sig.values()):
            errors.append(f"{name}: {prompt} non-deterministic fixed-depth signature {sig}")
        d = r["summary"][prompt]["decode_tps"]
        out["prompts"][prompt] = {
            "A": d["A"],
            "B": d["B"],
            "change_percent": d["change_percent"],
            "prefill_change_percent": r["summary"][prompt]["prefill_tps"]["change_percent"],
            "ttft_change_percent": r["summary"][prompt]["ttft_seconds"]["change_percent"],
            "per_pass": per_pass,
            "signatures": {k: sorted(v) for k, v in sig.items()},
            "depth_blocks": {k: sorted(v) for k, v in blocks.items()},
        }
    return out


def main():
    runs, output = Path(sys.argv[1]), Path(sys.argv[2])
    errors, result = [], {"campaigns": {}, "ratios": {}, "decision": {}}
    for doc in DOCS:
        for arm in ARMS:
            for order in ORDERS:
                c = check(runs, doc, arm, order, errors)
                if c is not None:
                    result["campaigns"][f"{doc}-{arm}-{order}"] = c
    if not errors:
        # ratio B/A per prompt and arm, per order; 'short' pooled over the three documents.
        prompts = {"short": [(doc, "short") for doc in DOCS], **{doc: [(doc, "document")] for doc in DOCS}}
        for label, sources in prompts.items():
            row = {}
            for arm in ARMS:
                changes = {
                    order: [result["campaigns"][f"{doc}-{arm}-{order}"]["prompts"][p]["change_percent"] for doc, p in sources]
                    for order in ORDERS
                }
                mean = statistics.mean(x for v in changes.values() for x in v)
                stable = all(x > 0 for v in changes.values() for x in v) or all(x < 0 for v in changes.values() for x in v)
                row[arm] = {"changes": changes, "mean": mean, "stable": stable}
            fixed = {"d1": 0.0}
            for arm in ("d2", "d3"):
                fixed[arm] = row[arm]["mean"] if row[arm]["stable"] and abs(row[arm]["mean"]) >= TIE else 0.0
            oracle = max(fixed, key=fixed.get)
            row["oracle"] = oracle
            row["oracle_gain"] = fixed[oracle]
            row["gap"] = {
                a: (1 + fixed[oracle] / 100) / (1 + (fixed[a] if a in fixed else row[a]["mean"]) / 100) * 100 - 100
                for a in ("d1", "d2", "d3", "ada")
            }
            # raw gaps (no tie rule) for transparency
            row["gap_raw"] = {
                a: (1 + max(0, row["d2"]["mean"], row["d3"]["mean"]) / 100)
                / (1 + (0 if a == "d1" else row[a]["mean"]) / 100) * 100 - 100
                for a in ("d1", "d2", "d3", "ada")
            }
            result["ratios"][label] = row
        labels = list(prompts)
        mean_gap = {a: statistics.mean(result["ratios"][p]["gap_raw"][a] for p in labels) for a in ("d1", "d2", "d3", "ada")}
        best_fixed = min(("d1", "d2", "d3"), key=mean_gap.get)
        ada_no_regress = all(
            x >= -TIE
            for doc in DOCS
            for order in ORDERS
            for x in [result["campaigns"][f"{doc}-ada-{order}"]["prompts"][p]["change_percent"] for p in ("short", "document")]
        )
        fixed_beats = {
            arm: all(
                result["ratios"][p][arm]["stable"] and min(x for v in result["ratios"][p][arm]["changes"].values() for x in v) >= TIE
                for p in labels
            )
            for arm in ("d2", "d3")
        }
        result["decision"] = {
            "mean_gap_raw_percent": mean_gap,
            "best_fixed": best_fixed,
            "ada_no_regression": ada_no_regress,
            "ada_gap_le_best_fixed": mean_gap["ada"] <= mean_gap[best_fixed],
            "ada_default_candidate": ada_no_regress and mean_gap["ada"] <= mean_gap[best_fixed],
            "fixed_beats_d1_everywhere": fixed_beats,
        }
    result["status"] = "passed" if not errors else "failed"
    result["errors"] = errors
    output.write_text(json.dumps(result, indent=2))
    for name, c in result["campaigns"].items():
        for prompt, v in c["prompts"].items():
            print(
                f"{name:24s} {prompt:8s} A {v['A']:6.2f} B {v['B']:6.2f} ({v['change_percent']:+6.2f} %)"
                f" pf {v['prefill_change_percent']:+5.2f} ttft {v['ttft_change_percent']:+5.2f}"
                f" passes {[(lab, round(x, 2)) for lab, x in v['per_pass']]}"
            )
            print(f"{'':24s} {'':8s} sig {v['signatures']} blocks {v['depth_blocks']}")
    for label, row in result["ratios"].items():
        print(
            label,
            {a: (round(row[a]["mean"], 2), row[a]["stable"], {o: [round(x, 2) for x in xs] for o, xs in row[a]["changes"].items()}) for a in ARMS},
            "oracle", row["oracle"], round(row["oracle_gain"], 2),
            "gap_raw", {a: round(g, 2) for a, g in row["gap_raw"].items()},
        )
    print("decision", json.dumps(result["decision"]))
    print("\n".join(errors) if errors else "all T2 checks passed")
    print("t2_status", result["status"])
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
