#!/usr/bin/env python3
"""DFLASH2-2026-11-25-INK2M-R0 (b): replays block-width policies on the trace of
`inkling2_dflash2::tests::policy_trace_t1` (teacher forcing on real text, ratio rule at T = 1).

A row of the trace is one generation of `anchors` tokens. At anchor a a policy picks a width w in 0..7:
w = 0 is a plain step (1 token, cost 1); w >= 1 is a block whose proposal e is kept with the traced ratio
acceptance of slot e (0 for a slot the real text does not reach), until the first rejection; it delivers
kept + 1 tokens and costs c0 + c1 * w plain steps. The score is tokens per plain-step cost (1.0 = no drafter).

Exactness constraint on the policies: the width of a block never reads the draws of that block. A truncation
before slot e may read q of slot e (a function of the proposals before it) but not the proposal of slot e.

CPU only, deterministic (fixed seed). Usage:
  inkling2_dflash2_policy_sim.py TRACE.json [--c0 1.40 --c1 0.375 --drafter 0.38 --sync 0.10 --replicas 200]
"""
import argparse
import json
import math
import random
import sys

MAX_W = 7
OLD_TABLE = [1.0, 2.46, 2.65, 2.83, 3.02, 3.2, 3.39, 3.57]  # WidthPolicy::DFLASH2_COSTS in service


class Ema:
    """Replica of `inkling2_draft::WidthPolicy` (measured costs, exponential acceptance by position)."""

    ALPHA, PRIOR, PROBE_EVERY, PROBE_MAX = 0.15, 0.75, 8, 64

    def __init__(self, costs, probes="narrow", alpha=None, probe_width=None, probe_max=None, floor=None, streak=False, restart=None):
        self.costs = costs
        self.acc = [self.PRIOR] * MAX_W
        self.probes = probes
        self.alpha = self.ALPHA if alpha is None else alpha
        self.probe_width = probe_width  # None: best + 1 (code); int: that width when the best is narrower
        self.probe_max = self.PROBE_MAX if probe_max is None else probe_max
        self.floor = floor  # None, or a minimum width once a block pays (never 1..floor-1)
        self.streak = streak  # a probe whose proposals were all kept is followed by another probe
        self.restart = restart  # smallest gap after a probe with a rejection (None: 1, so 2 after a streak)
        self.blocks = 0
        self.gap = self.PROBE_EVERY
        self.since = 0
        self.probing = False

    def expected(self, w, acc=None):
        acc = self.acc if acc is None else acc
        reach, total = 1.0, 1.0
        for a in acc[:w]:
            reach *= a
            total += reach
        return total

    def best(self, acc=None):
        best, rate = 0, -1e300
        for w in range(MAX_W + 1):
            r = self.expected(w, acc) / self.costs[w]
            if r > rate + 1e-12:
                best, rate = w, r
        return best

    def next_width(self):
        self.blocks += 1
        best = self.best()
        if self.probes == "full":
            return MAX_W if self.blocks % self.PROBE_EVERY == 0 else best
        self.since += 1
        if self.since < self.gap:
            return best
        self.since = 0
        self.probing = True
        if self.probe_width is not None:
            return max(min(best + 1, MAX_W), min(self.probe_width, MAX_W))
        return min(best + 1, MAX_W)

    def observe(self, w, kept):
        for i in range(min(w, MAX_W)):
            if i > kept:
                break
            hit = 1.0 if kept > i else 0.0
            self.acc[i] = (1 - self.alpha) * self.acc[i] + self.alpha * hit
        if self.probing:
            self.probing = False
            if self.streak and w > 0 and kept >= w:
                self.gap = 1
            else:
                self.gap = min(max(self.gap * 2, self.restart or 1), self.probe_max) if self.best() == 0 else self.PROBE_EVERY


def reach_slots(row, a, count_key):
    """Ratio acceptances of the block at anchor a (unreached slots: 0)."""
    return [max(v, 0.0) for v in row["blocks"][count_key]["ratio"][a]]


def data_key(w, fidelity):
    """Which traced block gives the slots of a block of w proposals: the 7-proposal block when the drafter always
    runs its whole block ("full"), else the nearest traced count at or above w ("narrow": 1, 3 or 7)."""
    if fidelity == "full":
        return "7"
    return "1" if w == 1 else ("3" if w <= 3 else "7")


def bins_fit(pairs, edges):
    """Mean of y per bin of x (pairs of (x, y)); empty bins take the global mean."""
    sums = [[0.0, 0] for _ in range(len(edges) + 1)]
    for x, y in pairs:
        b = sum(1 for e in edges if x >= e)
        sums[b][0] += y
        sums[b][1] += 1
    total = sum(s for s, _ in sums) / max(1, sum(n for _, n in sums))
    return [s / n if n >= 8 else total for s, n in sums]


def bins_eval(table, edges, x):
    return table[sum(1 for e in edges if x >= e)]


Q_EDGES = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
SLOT_GROUP = [0, 1, 1, 2, 2, 2, 2]
LP_EDGES = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.6, -0.3, -0.1]


def fit_q(rows):
    """Acceptance of a slot from the drafter's own q_max, by slot group (calibration rows only)."""
    tables = []
    for group in range(3):
        pairs = []
        for row in rows:
            block = row["blocks"]["7"]
            for ratios, qmaxs in zip(block["ratio"], block["q_max"]):
                for e in range(MAX_W):
                    if SLOT_GROUP[e] == group and ratios[e] >= 0:
                        pairs.append((qmaxs[e], ratios[e]))
        tables.append(bins_fit(pairs, Q_EDGES))
    return tables


def fit_lp(rows, span):
    """First-slot acceptance from the target's mean log p of the last `span` delivered tokens."""
    pairs = []
    for row in rows:
        ratios = row["blocks"]["7"]["ratio"]
        for a in range(span, len(ratios)):
            pairs.append((sum(row["lp_real"][a - span:a]) / span, max(ratios[a][0], 0.0)))
    return bins_fit(pairs, LP_EDGES)


def simulate(row, policy, truth, rng, fidelity, anchors):
    """One replica. `policy(a, history)` returns (w, drafted) where drafted says the drafter ran (a width-0 block
    after a drafter run still pays it). Returns (tokens, cost, widths histogram, kept, proposed)."""
    c0, c1, drafter, sync, dead = truth
    a, tokens, cost = 0, 0, 0.0
    hist = [0] * (MAX_W + 1)
    kept_total = proposed = 0
    while a < anchors:
        w, drafted, synced, alive = policy.choose(row, a)
        w = min(w, anchors - a - 1) if anchors - a - 1 >= 0 else 0
        if alive is not None and alive > w:
            alive = None
        if w <= 0:
            cost += 1.0 + (drafter + sync if drafted else 0.0)
            policy.observe(row, a, 0, 0)
            hist[0] += 1
            tokens += 1
            a += 1
            continue
        slots = reach_slots(row, a, data_key(w, fidelity))
        # alive rows L <= w: rows 0..L-1 test proposals 1..L and the rows after them cost `dead` each (no expert
        # read); when every tested proposal is kept the last one is the next anchor (no bonus token)
        tested = w if alive is None else alive
        kept = 0
        for e in range(tested):
            if rng.random() < slots[e]:
                kept += 1
            else:
                break
        if alive is None:
            cost += c0 + c1 * w + (sync if synced else 0.0)
            delivered = kept + 1
        else:
            cost += c0 + c1 * (alive - 1) + dead * (w + 1 - alive)
            delivered = min(kept + 1, alive)
        hist[tested] += 1
        kept_total += kept
        proposed += tested
        policy.observe(row, a, tested, kept)
        tokens += delivered
        a += delivered
    return tokens, cost, hist, kept_total, proposed


class Never:
    def choose(self, row, a):
        return 0, False, False, None

    def observe(self, row, a, w, kept):
        pass


class EmaPolicy:
    """The engine's policy with a cost table (the table is a belief; the truth is c0 + c1 w)."""

    def __init__(self, costs, on_off=False, **kw):
        self.costs, self.kw, self.on_off = costs, kw, on_off
        self.ema = Ema(costs, **kw)

    def fresh(self):
        return EmaPolicy(self.costs, self.on_off, **self.kw)

    def choose(self, row, a):
        w = self.ema.next_width()
        return (MAX_W if w > 0 and self.on_off else w), False, False, None

    def observe(self, row, a, w, kept):
        self.ema.observe(w, kept)


class LpGate:
    """EMA policy whose first-slot estimate is the target's recent confidence (free signal)."""

    def __init__(self, costs, table, span):
        self.costs, self.table, self.span = costs, table, span
        self.ema = Ema(costs)

    def fresh(self):
        return LpGate(self.costs, self.table, self.span)

    def choose(self, row, a):
        if a < self.span:
            return self.ema.best(), False, False, None
        first = bins_eval(self.table, LP_EDGES, sum(row["lp_real"][a - self.span:a]) / self.span)
        acc = [first] + self.ema.acc[1:]
        return self.ema.best(acc), False, False, None

    def observe(self, row, a, w, kept):
        self.ema.observe(w, kept)


class QTrunc:
    """The drafter runs (always, or when `gate` wants a block); the width is the best one under the acceptances
    predicted from q_max of each slot along the path (read before the verification: one sync). Slot e's q is read
    given the proposals before it, never its own draw; an unreached slot keeps the previous estimate."""

    def __init__(self, truth, tables, gate=None, margin=0.0):
        self.truth, self.tables, self.gate, self.margin = truth, tables, gate, margin

    def fresh(self):
        return QTrunc(self.truth, self.tables, self.gate.fresh() if self.gate else None, self.margin)

    def choose(self, row, a):
        if self.gate is not None:
            w, _, _, _ = self.gate.choose(row, a)
            if w == 0:
                return 0, False, False, None
        c0, c1, drafter, sync, _ = self.truth
        qmaxs = row["blocks"]["7"]["q_max"][a]
        acc, previous = [], 0.5
        for e in range(MAX_W):
            if qmaxs[e] >= 0:
                previous = bins_eval(self.tables[SLOT_GROUP[e]], Q_EDGES, qmaxs[e])
            acc.append(previous)
        best, rate = 0, (1.0 / (1.0 + drafter + sync)) + self.margin
        reach, total = 1.0, 1.0
        for w in range(1, MAX_W + 1):
            reach *= acc[w - 1]
            total += reach
            r = total / (c0 + c1 * w + sync)
            if r > rate + 1e-12:
                best, rate = w, r
        return best, True, True, None

    def observe(self, row, a, w, kept):
        if self.gate is not None:
            self.gate.observe(row, a, w, kept)


class LiveRows:
    """Splash's "rows without a lane" transposed to a chain: the host picks the graph (W proposals, from `gate`; 0
    = plain step, no drafter), the GPU keeps row j alive while the estimated reach of proposal j (product of the
    acceptances predicted from q of slots 1..j, known once proposal j is drawn) is at least `floor`; the rows
    after the last alive one read no expert (cost `dead` each). No host read between drafter and verification."""

    def __init__(self, tables, gate, floor, full=True):
        self.tables, self.gate, self.floor, self.full = tables, gate, floor, full

    def fresh(self):
        return LiveRows(self.tables, self.gate.fresh() if self.gate else None, self.floor, self.full)

    def choose(self, row, a):
        w = MAX_W
        if self.gate is not None:
            w, _, _, _ = self.gate.choose(row, a)
            if w == 0:
                return 0, False, False, None
            if self.full:
                w = MAX_W
        qmaxs = row["blocks"]["7"]["q_max"][a]
        reach, previous, alive = 1.0, 0.5, 1
        for e in range(w):
            if qmaxs[e] >= 0:
                previous = bins_eval(self.tables[SLOT_GROUP[e]], Q_EDGES, qmaxs[e])
            reach *= previous
            if reach < self.floor:
                break
            alive = e + 2
        return w, True, False, alive

    def observe(self, row, a, w, kept):
        if self.gate is not None:
            self.gate.observe(row, a, w, kept)


class OracleRows:
    """Bound of the alive-row rule: the alive rows from the true slot acceptances (a perfect estimator), the
    drafter and the dead rows paid on every block the true best width keeps."""

    def __init__(self, truth, floor):
        self.truth, self.floor = truth, floor

    def fresh(self):
        return self

    def choose(self, row, a):
        slots = reach_slots(row, a, "7")
        reach, alive = 1.0, 1
        for e in range(MAX_W):
            reach *= slots[e]
            if reach < self.floor:
                break
            alive = e + 2
        return MAX_W, True, False, alive

    def observe(self, row, a, w, kept):
        pass


class Oracle:
    """Upper bound: the best width for the true slot acceptances of each anchor (no drafter charge at width 0)."""

    def __init__(self, truth, fidelity):
        self.truth, self.fidelity = truth, fidelity

    def fresh(self):
        return self

    def choose(self, row, a):
        c0, c1, _, _, _ = self.truth
        slots = reach_slots(row, a, "7")
        best, rate = 0, 1.0
        reach, total = 1.0, 1.0
        for w in range(1, MAX_W + 1):
            reach *= slots[w - 1]
            total += reach
            r = total / (c0 + c1 * w)
            if r > rate + 1e-12:
                best, rate = w, r
        return best, False, False, None

    def observe(self, row, a, w, kept):
        pass


def run(rows, make, truth, fidelity, replicas, seed, anchors):
    """Totals over rows and replicas; `make(index)` builds the policy for row `index` (cross-fitted tables)."""
    out = {"tokens": 0, "cost": 0.0, "hist": [0] * (MAX_W + 1), "kept": 0, "proposed": 0, "by_row": {}}
    for index, row in enumerate(rows):
        rng = random.Random(seed * 1000003 + index)
        tokens = cost = 0
        for _ in range(replicas):
            policy = make(index)
            t, c, hist, kept, proposed = simulate(row, policy, truth, rng, fidelity, anchors)
            tokens += t
            cost += c
            out["hist"] = [x + y for x, y in zip(out["hist"], hist)]
            out["kept"] += kept
            out["proposed"] += proposed
        out["tokens"] += tokens
        out["cost"] += cost
        out["by_row"][row["name"]] = tokens / cost
    out["rate"] = out["tokens"] / out["cost"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--c0", type=float, default=1.40)
    parser.add_argument("--c1", type=float, default=0.375)
    parser.add_argument("--drafter", type=float, default=0.38)
    parser.add_argument("--sync", type=float, default=0.10)
    parser.add_argument("--dead", type=float, default=0.067, help="cost of a row that reads no expert, in steps")
    parser.add_argument("--replicas", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    with open(args.trace, encoding="utf-8") as handle:
        trace = json.load(handle)
    rows = trace["rows"]
    anchors = trace["anchors"]
    if len(rows) < 6 or any(len(r["blocks"]["7"]["ratio"]) != anchors for r in rows):
        print("invalid trace: rows or anchors missing", file=sys.stderr)
        return 2
    for row in rows:
        values = [v for block in row["blocks"]["7"]["ratio"] for v in block]
        if len(values) != anchors * MAX_W or not all(math.isfinite(v) and -1.0 <= v <= 1.0 + 1e-9 for v in values):
            print(f"invalid trace values in {row['name']}", file=sys.stderr)
            return 2
    truth = (args.c0, args.c1, args.drafter, args.sync, args.dead)
    new_table = [1.0] + [args.c0 + args.c1 * w for w in range(1, MAX_W + 1)]

    # R0-a: the same slots in the blocks of 1, 3 and 7 proposals (reached in both)
    print("R0-a slot acceptance (ratio rule), narrow block against the 7-proposal block, slots reached in both")
    for key in ("1", "3"):
        count = int(key)
        sums = [[0.0, 0.0, 0] for _ in range(count)]
        for row in rows:
            for narrow, wide in zip(row["blocks"][key]["ratio"], row["blocks"]["7"]["ratio"]):
                for e in range(count):
                    if narrow[e] >= 0 and wide[e] >= 0:
                        sums[e][0] += narrow[e]
                        sums[e][1] += wide[e]
                        sums[e][2] += 1
        text = ", ".join(f"slot {e + 1}: {s[0] / max(1, s[2]):.3f} vs {s[1] / max(1, s[2]):.3f} (n {s[2]})" for e, s in enumerate(sums))
        print(f"  block of {count}: {text}")

    def others(index):
        return [r for i, r in enumerate(rows) if i != index]

    q_tables = [fit_q(others(i)) for i in range(len(rows))]
    lp_tables = {span: [fit_lp(others(i), span) for i in range(len(rows))] for span in (1, 2, 4, 8)}
    print("R0-b q_max calibration (all rows; slot group 0 / 1-2 / 3-6), bins", Q_EDGES)
    for group, table in enumerate(fit_q(rows)):
        print(f"  group {group}: " + " ".join(f"{v:.2f}" for v in table))
    print("R0-b target lp calibration of slot 1 (span 4), bins", LP_EDGES)
    print("  " + " ".join(f"{v:.2f}" for v in fit_lp(rows, 4)))

    policies = [
        ("P0 never", lambda i: Never(), "narrow"),
        ("P1 current (old table, narrow blocks)", lambda i: EmaPolicy(OLD_TABLE), "narrow"),
        ("P1f current, full block always", lambda i: EmaPolicy(OLD_TABLE), "full"),
        ("P2 recalibrated table (narrow blocks)", lambda i: EmaPolicy(new_table), "narrow"),
        ("P2f recalibrated table, full block", lambda i: EmaPolicy(new_table), "full"),
        ("P3a P2f, alpha 0.3", lambda i: EmaPolicy(new_table, alpha=0.3), "full"),
        ("P3b P2f, probe width 3", lambda i: EmaPolicy(new_table, probe_width=3), "full"),
        ("P3c P2f, probe width 3, gap max 16", lambda i: EmaPolicy(new_table, probe_width=3, probe_max=16), "full"),
        ("P3d P2f, alpha 0.3, probe 3, gap max 16", lambda i: EmaPolicy(new_table, alpha=0.3, probe_width=3, probe_max=16), "full"),
        ("P3e P2f, full-width probes", lambda i: EmaPolicy(new_table, probes="full"), "full"),
        ("S1 P2f + probe streak", lambda i: EmaPolicy(new_table, streak=True), "full"),
        ("S2 P2f + probe streak, alpha 0.3", lambda i: EmaPolicy(new_table, streak=True, alpha=0.3), "full"),
        ("S3 P2f + probe streak, gap max 16", lambda i: EmaPolicy(new_table, streak=True, probe_max=16), "full"),
        ("S4 P2f + probe streak, gap max 32", lambda i: EmaPolicy(new_table, streak=True, probe_max=32), "full"),
        ("S0 old table + full block + probe streak", lambda i: EmaPolicy(OLD_TABLE, streak=True), "full"),
        ("S5 P2f + probe streak, gap >= 8 after a rejected probe", lambda i: EmaPolicy(new_table, streak=True, restart=8), "full"),
        ("S6 P2f + probe streak, gap >= 4 after a rejected probe", lambda i: EmaPolicy(new_table, streak=True, restart=4), "full"),
    ]
    for span in (1, 2, 4, 8):
        policies.append((f"P4 target lp gate span {span}, full block", lambda i, s=span: LpGate(new_table, lp_tables[s][i], s), "full"))
    policies += [
        ("P5 drafter always + q truncation", lambda i: QTrunc(truth, q_tables[i]), "full"),
        ("P6a EMA gate (new table) + q truncation", lambda i: QTrunc(truth, q_tables[i], EmaPolicy(new_table)), "full"),
        ("P6b EMA gate (alpha 0.3, probe 3, gap 16) + q truncation", lambda i: QTrunc(truth, q_tables[i], EmaPolicy(new_table, alpha=0.3, probe_width=3, probe_max=16)), "full"),
        ("P6c lp gate span 4 + q truncation", lambda i: QTrunc(truth, q_tables[i], LpGate(new_table, lp_tables[4][i], 4)), "full"),
    ]
    for floor in (0.25, 0.35, 0.45, 0.55):
        policies += [
            (f"L1 drafter always, alive rows floor {floor}", lambda i, f=floor: LiveRows(q_tables[i], None, f), "full"),
            (f"L2 EMA gate on/off (new table) + alive rows floor {floor}", lambda i, f=floor: LiveRows(q_tables[i], EmaPolicy(new_table), f), "full"),
            (f"L3 EMA gate (alpha 0.3, probe 3, gap 16) + alive rows floor {floor}", lambda i, f=floor: LiveRows(q_tables[i], EmaPolicy(new_table, alpha=0.3, probe_width=3, probe_max=16), f), "full"),
            (f"L4 EMA gate width (graph = gate width) + alive rows floor {floor}", lambda i, f=floor: LiveRows(q_tables[i], EmaPolicy(new_table), f, full=False), "full"),
            (f"BOUND alive rows from true acceptances, floor {floor}", lambda i, f=floor: OracleRows(truth, f), "full"),
        ]
    policies += [
        ("BOUND oracle width per anchor", lambda i: Oracle(truth, "full"), "full"),
    ]
    print(f"R0-b truth: block of w = {args.c0} + {args.c1} w steps, drafter {args.drafter}, q read {args.sync}, dead row {args.dead}; "
          f"{len(rows)} rows x {anchors} anchors x {args.replicas} replicas")
    results = {}
    for name, make, fidelity in policies:
        r = run(rows, make, truth, fidelity, args.replicas, args.seed, anchors)
        results[name] = r
        blocks = sum(r["hist"])
        hist = " ".join(f"{100 * h / blocks:.0f}" for h in r["hist"])
        accepted = r["kept"] / r["proposed"] if r["proposed"] else 0.0
        print(f"  {name:58s} x{r['rate']:.3f}  widths% [{hist}]  kept/proposed {accepted:.2f}")
    categories = {}
    for name in results:
        for row_name, rate in results[name]["by_row"].items():
            categories.setdefault(name, {}).setdefault(row_name.rstrip("0123456789"), []).append(rate)
    print("R0-b by category (mean of the rows' rates)")
    for name, cats in categories.items():
        text = "  ".join(f"{c} {sum(v) / len(v):.3f}" for c, v in sorted(cats.items()))
        print(f"  {name:58s} {text}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"truth": truth, "results": {k: {"rate": v["rate"], "hist": v["hist"], "kept": v["kept"], "proposed": v["proposed"], "by_row": v["by_row"]} for k, v in results.items()}}, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
