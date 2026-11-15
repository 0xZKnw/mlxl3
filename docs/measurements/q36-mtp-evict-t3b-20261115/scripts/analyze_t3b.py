"""Q36-MTP-EVICT-T3B speed analysis. Fails closed on any missing campaign, divergence or wrong protocol.

Usage: python -I analyze_t3b.py <speed dir> <output.json>
Arm d1: A = D1, B = D3 + EVICT (criterion). Arm d3: A = D3, B = D3 + EVICT (descriptive).
"""

import json
import sys
from pathlib import Path

DOCS = ("b-code", "c-prose-fr")
ARMS = ("d1", "d3")
ORDERS = ("BAAB", "ABBA")
GAIN, SIDE = 1.5, 2.0


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
    env = {k: v["environment"] for k, v in r["binaries"].items()}
    want = {
        "A": {"MLXL3_MTP_POST_NORM": "1", "MLXL3_MTP_EVICT": "0"},
        "B": {"MLXL3_MTP_POST_NORM": "1", "MLXL3_MTP_EVICT": "1"},
    }
    if env != want:
        errors.append(f"{name}: environment {env}")
    if r["binaries"]["A"]["sha256"] != r["binaries"]["B"]["sha256"]:
        errors.append(f"{name}: binaries differ")
    if [p["label"] for p in r["passes"]] != list(order):
        errors.append(f"{name}: order")
    for p in r["passes"]:
        depth = 3 if p["label"] == "B" or arm == "d3" else 1
        if p["mtp_depth"] != depth:
            errors.append(f"{name}: pass {p['index']} depth {p['mtp_depth']} != {depth}")
        flag = "1" if p["label"] == "B" else "0"
        if f"mtppostnorm-v1=1:mtpevict-v1={flag}:" not in (p["ready"].get("mtp_tuning_key") or ""):
            errors.append(f"{name}: pass {p['index']} tuning key lacks post-norm/evict={flag}")
    out = {"prompts": {}}
    if set(r["summary"]) != {"short", "document"}:
        errors.append(f"{name}: prompts {sorted(r['summary'])}")
    for prompt in r["summary"]:
        sig, per_pass = {}, []
        for p in r["passes"]:
            runs_ = p["prompts"][prompt]["runs"]
            if len(runs_) != 2:
                errors.append(f"{name}: {prompt} pass {p['index']} has {len(runs_)} runs")
            for run in runs_:
                s = run["stats"]
                if s["generated_tokens"] != 256:
                    errors.append(f"{name}: {prompt} generated {s['generated_tokens']}")
                sig.setdefault(p["label"], set()).add(
                    (run["token_hash"], s["mtp_accepted_tokens"], s["mtp_proposed_tokens"], s["mtp_blocks"])
                )
            per_pass.append((p["label"], p["prompts"][prompt]["medians"]["decode_tps"]))
        hashes = {h for v in sig.values() for (h, *_rest) in v}
        if len(hashes) != 1:
            errors.append(f"{name}: {prompt} token hashes {hashes}")
        if any(len(v) != 1 for v in sig.values()):
            errors.append(f"{name}: {prompt} non-deterministic signature {sig}")
        s = r["summary"][prompt]
        out["prompts"][prompt] = {
            "A": s["decode_tps"]["A"],
            "B": s["decode_tps"]["B"],
            "change_percent": s["decode_tps"]["change_percent"],
            "prefill_change_percent": s["prefill_tps"]["change_percent"],
            "ttft_change_percent": s["ttft_seconds"]["change_percent"],
            "per_pass": per_pass,
            "signatures": {k: sorted(v) for k, v in sig.items()},
        }
    return out


def main():
    runs, output = Path(sys.argv[1]), Path(sys.argv[2])
    errors, result = [], {"campaigns": {}, "decision": {}}
    for doc in DOCS:
        for arm in ARMS:
            for order in ORDERS:
                c = check(runs, doc, arm, order, errors)
                if c is not None:
                    result["campaigns"][f"{doc}-{arm}-{order}"] = c
    if not errors:
        for arm in ARMS:
            cells = [
                (doc, order, prompt, result["campaigns"][f"{doc}-{arm}-{order}"]["prompts"][prompt])
                for doc in DOCS
                for order in ORDERS
                for prompt in ("short", "document")
            ]
            result["decision"][arm] = {
                "changes": {f"{d} {o} {p}": round(v["change_percent"], 2) for d, o, p, v in cells},
                "min_change": min(v["change_percent"] for *_k, v in cells),
                "all_ge_gain": all(v["change_percent"] >= GAIN for *_k, v in cells),
                "prefill_ok": all(v["prefill_change_percent"] >= -SIDE for *_k, v in cells),
                "ttft_ok": all(v["ttft_change_percent"] <= SIDE for *_k, v in cells),
            }
        d = result["decision"]["d1"]
        result["decision"]["validated_option"] = d["all_ge_gain"] and d["prefill_ok"] and d["ttft_ok"]
    result["status"] = "passed" if not errors else "failed"
    result["errors"] = errors
    output.write_text(json.dumps(result, indent=2))
    for name, c in result["campaigns"].items():
        for prompt, v in c["prompts"].items():
            print(
                f"{name:22s} {prompt:8s} A {v['A']:6.2f} B {v['B']:6.2f} ({v['change_percent']:+6.2f} %)"
                f" pf {v['prefill_change_percent']:+5.2f} ttft {v['ttft_change_percent']:+5.2f}"
                f" passes {[(lab, round(x, 2)) for lab, x in v['per_pass']]}"
            )
            print(f"{'':22s} {'':8s} sig {v['signatures']}")
    print("decision", json.dumps(result["decision"]))
    print("\n".join(errors) if errors else "all T3b speed checks passed")
    print("t3b_speed_status", result["status"])
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
