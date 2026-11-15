"""Q36-MTP-EVICT-T3A: offline replay of verify-width truncation on recorded MTP confidence traces.

Cycles are replayed independently (a truncation would shift later cycle starts; not modelled).
Fails closed on any misaligned trace. Usage:
  python -I simulate_evict.py <output.json> <post diag.json> <pre diag.json> <t0b dir>
"""

import json
import math
import sys
from pathlib import Path

DRAFT = {0: 0.0, 1: 1.31, 2: 2.45, 3: 3.60}  # ms, T0b marker medians
VERIFY = {2: 21.3, 3: 29.0, 4: 33.3}  # ms by verified rows, T0b marker medians
TAIL = 0.75  # greedy + commit/repair tail, ms
V1_GRID = (15.0, 17.0, 19.0)  # single-row verify, not measured
BINS = (0.5, 0.8, 0.95, 0.99)
TAUS = [round(0.05 * i, 2) for i in range(2, 20)]
FR = ("a-short-fr", "c-prose-fr")
CODE = ("b-code",)
GAIN = 1.5


def band(p):
    return sum(p >= b for b in BINS)


def cost(width, k, v1):
    rows = k + 1
    return DRAFT[width] + (v1 if rows == 1 else VERIFY[rows]) + TAIL


def load(path, errors):
    d = json.loads(Path(path).read_text())
    if not d.get("confidence"):
        errors.append(f"{path}: confidence capture off")
    cases = {}
    for c in d["cases"]:
        levels = c["trace"]["levels"]
        if c["trace"].get("overflowed"):
            errors.append(f"{c['prompt']} D{c['depth']}: trace overflowed")
        if len(levels) != sum(w for w, _ in c["cycles"]) or len(levels) != c["proposed"]:
            errors.append(f"{c['prompt']} D{c['depth']}: {len(levels)} levels vs {c['proposed']} proposed")
            continue
        if any(not lv["finite"] or lv["proposal"] != lv["host_proposal"] for lv in levels):
            errors.append(f"{c['prompt']} D{c['depth']}: non-finite level or host/device mismatch")
        cycles, at = [], 0
        for w, a in c["cycles"]:
            if not 0 <= a <= w <= 3:
                errors.append(f"{c['prompt']} D{c['depth']}: impossible cycle {(w, a)}")
            cycles.append((w, a, [math.exp(lv["top1_log_prob"]) for lv in levels[at : at + w]]))
            at += w
        if sum(a for _, a, _ in cycles) != c["accepted"]:
            errors.append(f"{c['prompt']} D{c['depth']}: accepted mismatch")
        cases[(c["prompt"], c["depth"])] = cycles
    return cases


def calibrate(cases, prompts):
    hit = [[1] * (len(BINS) + 1) for _ in range(3)]
    tested = [[2] * (len(BINS) + 1) for _ in range(3)]
    for (prompt, _), cycles in cases.items():
        if prompt not in prompts:
            continue
        for w, a, ps in cycles:
            for i in range(w):
                if a < i:
                    break
                tested[i][band(ps[i])] += 1
                hit[i][band(ps[i])] += a > i
    return [[h / t for h, t in zip(hr, tr, strict=True)] for hr, tr in zip(hit, tested, strict=True)]


def replay(cycles, choose, v1):
    tokens = seconds = 0.0
    for w, a, ps in cycles:
        k = choose(w, a, ps) if w else 0
        tokens += min(a, k) + 1
        seconds += cost(w, k, v1) / 1000
    return tokens / seconds


def evict(table, v1):
    def choose(w, a, ps):
        best, best_k, survive, expected = -1.0, 0, 1.0, 0.0
        for k in range(w + 1):
            if k:
                survive *= table[k - 1][band(ps[k - 1])]
                expected += survive
            value = (1 + expected) / cost(w, k, v1)
            if value > best:
                best, best_k = value, k
        return best_k

    return choose


def threshold(tau):
    def choose(w, a, ps):
        k = 0
        while k < w and ps[k] >= tau:
            k += 1
        return k

    return choose


def simulate(cases, label):
    prompts = sorted({p for p, _ in cases})
    out = {}
    for v1 in V1_GRID:
        rows = {}
        for prompt in prompts:
            others = [p for p in prompts if p != prompt]
            table = calibrate(cases, others)
            row = {}
            for depth in (1, 2, 3):
                row[f"fixed{depth}"] = replay(cases[(prompt, depth)], lambda w, a, ps: w, v1)
            for depth in (2, 3):
                cyc = cases[(prompt, depth)]
                row[f"evict{depth}"] = replay(cyc, evict(table, v1), v1)
                tau = max(
                    TAUS,
                    key=lambda t: sum(replay(cases[(o, depth)], threshold(t), v1) for o in others),
                )
                row[f"tau{depth}"] = replay(cyc, threshold(tau), v1)
                row[f"tau{depth}_value"] = tau
                row[f"oracle{depth}"] = replay(cyc, lambda w, a, ps: a, v1)
            best_fixed = max(("fixed1", "fixed2", "fixed3"), key=row.get)
            best_policy = max(("evict2", "evict3", "tau2", "tau3"), key=row.get)
            row["best_fixed"] = best_fixed
            row["best_policy"] = best_policy
            row["gain_percent"] = (row[best_policy] / row[best_fixed] - 1) * 100
            row["oracle_gain_percent"] = (max(row["oracle2"], row["oracle3"]) / row[best_fixed] - 1) * 100
            rows[prompt] = row
        out[str(v1)] = rows
    return out


def validate(t0b, errors):
    # loop seconds without the dylib for 127 decode tokens, from t0b/analysis.log ("dylib X vs Y").
    measured = {
        ("a-short-fr", 1): 1.576, ("a-short-fr", 2): 1.767, ("a-short-fr", 3): 1.707,
        ("b-code", 1): 1.615, ("b-code", 2): 1.710, ("b-code", 3): 1.737,
    }
    d = json.loads((Path(t0b) / "nodylib" / "diag.json").read_text())
    rows = {}
    for c in d["cases"]:
        key = (c["prompt"], c["depth"])
        cycles = [(w, a, [1.0] * w) for w, a in c["cycles"]]
        sim = replay(cycles, lambda w, a, ps: w, 17.0)
        real = 127 / measured[key]
        rows[f"{key[0]}-D{key[1]}"] = {"simulated": sim, "loop": real, "error_percent": (sim / real - 1) * 100}
        if abs(sim / real - 1) > 0.05:
            errors.append(f"cost model off by more than 5% on {key}")
    return rows


def main():
    output, post, pre, t0b = sys.argv[1:5]
    errors = []
    result = {"model": {"draft": DRAFT, "verify": VERIFY, "tail": TAIL, "v1_grid": V1_GRID}}
    result["validation"] = validate(t0b, errors)
    post_cases, pre_cases = load(post, errors), load(pre, errors)
    if not errors:
        result["post"] = simulate(post_cases, "post")
        result["pre"] = simulate(pre_cases, "pre")
        verdict = {}
        for v1, rows in result["post"].items():
            verdict[v1] = all(rows[p]["gain_percent"] >= GAIN for p in FR + CODE)
        result["passes_criterion_by_v1"] = verdict
        result["passes_criterion"] = all(verdict.values())
    result["status"] = "passed" if not errors else "failed"
    result["errors"] = errors
    Path(output).write_text(json.dumps(result, indent=2))
    for k, v in result["validation"].items():
        print(f"validate {k:14s} sim {v['simulated']:6.2f} loop {v['loop']:6.2f} ({v['error_percent']:+.2f} %)")
    for wiring in ("post", "pre"):
        for v1, rows in result.get(wiring, {}).items():
            for prompt, r in rows.items():
                print(
                    f"{wiring} V1={v1} {prompt:14s} "
                    + " ".join(f"{k} {r[k]:.2f}" for k in ("fixed1", "fixed2", "fixed3", "evict2", "evict3", "tau2", "tau3", "oracle2", "oracle3"))
                    + f" tau {r['tau2_value']}/{r['tau3_value']} | best {r['best_policy']} vs {r['best_fixed']} {r['gain_percent']:+.2f} % (oracle {r['oracle_gain_percent']:+.2f} %)"
                )
    print("criterion", result.get("passes_criterion_by_v1"), result.get("passes_criterion"))
    print("\n".join(errors) if errors else "all T3a checks passed")
    print("t3a_status", result["status"])
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
