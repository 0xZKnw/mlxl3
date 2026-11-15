"""Q36-MTP-EVICT-T3B: compare a diagnostic report with a reference report.

Usage: python -I compare_t3b.py <actual diag.json> <reference diag.json> <same|evict> <variant>
`same`: EVICT off; every case has identical tokens, cycles, histogram, depth_blocks, accepted,
proposed and, when both captured confidence, identical per-level host readings.
`evict`: EVICT on, confidence on; tokens identical to the AR-checked reference; the device top-1
bits equal the host top-1 of the full compact vector; every recorded k equals an independent
float64 recomputation of the pre-registered policy from the host top-1 values; MTP1 cycles are
unchanged (k = 1 always); at least one MTP2/MTP3 case truncates. Fails closed on missing data.
"""

import json
import math
import struct
import sys
from pathlib import Path

FIELDS = ("tokens", "cycles", "histogram", "depth_blocks", "accepted", "proposed")
LEVEL_FIELDS = ("proposal", "host_proposal", "top1_log_prob", "top2_log_prob", "top1_ties", "finite")
# Pre-registered constants (opti.md, Q36-MTP-EVICT-T3B), typed independently of contracts.rs.
EDGES = [math.log(p) for p in (0.5, 0.8, 0.95, 0.99)]
SURVIVAL = [[0.368, 0.645, 0.905, 0.968, 0.998], [0.256, 0.631, 0.924, 0.958, 0.995],
            [0.172, 0.685, 0.864, 0.923, 0.986]]
DRAFT = [0.0, 1310.0, 2450.0, 3600.0]
VERIFY = [17000.0, 21300.0, 29000.0, 33300.0]
TAIL = 750.0


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def policy(tops):
    w = len(tops)
    best, best_k, survive, expected, near = -1.0, 0, 1.0, 0.0, False
    for k in range(w + 1):
        if k:
            lp = tops[k - 1]
            near |= any(abs(lp - e) < 1e-6 for e in EDGES)
            survive *= SURVIVAL[k - 1][sum(lp >= e for e in EDGES)]
            expected += survive
        value = (1 + expected) / (DRAFT[w] + VERIFY[k] + TAIL)
        if value > best:
            best, best_k = value, k
    return best_k, near


def main():
    actual_path, reference_path, mode, variant = sys.argv[1:5]
    errors, summary = [], {}
    if mode not in ("same", "evict"):
        print("mode must be same or evict")
        return 2
    try:
        actual = json.loads(Path(actual_path).read_text())
        reference = json.loads(Path(reference_path).read_text())
        if actual.get("hidden_variant") != variant:
            errors.append(f"hidden_variant {actual.get('hidden_variant')!r} != {variant!r}")
        if actual.get("evict") is not (mode == "evict"):
            errors.append(f"evict flag {actual.get('evict')!r} does not match mode {mode}")
        if not actual.get("compact_head"):
            errors.append("compact head not active")
        a = {(c["prompt"], c["depth"]): c for c in actual["cases"]}
        r = {(c["prompt"], c["depth"]): c for c in reference["cases"]}
        if len(a) != 12 or set(a) != set(r):
            errors.append(f"case sets differ or not 12 cases: {len(a)} vs {len(r)}")
        truncating = 0
        for key in sorted(set(a) & set(r)):
            x, y = a[key], r[key]
            if not x["tokens"] or len(x["tokens"]) != len(y["tokens"]) or x["tokens"] != y["tokens"]:
                errors.append(f"{key}: tokens empty, wrong length or different")
            trace = x.get("trace") or {}
            evictions = trace.get("evictions", [])
            if mode == "same":
                changed = [f for f in FIELDS[1:] if x[f] != y[f]]
                if changed:
                    errors.append(f"{key}: {', '.join(changed)} differ")
                if evictions:
                    errors.append(f"{key}: EVICT records while off")
                if x.get("trace", {}).get("confidence") and y.get("trace", {}).get("confidence"):
                    lx, ly = x["trace"]["levels"], y["trace"]["levels"]
                    if len(lx) != len(ly) or not lx:
                        errors.append(f"{key}: level count {len(lx)} vs {len(ly)}")
                    elif any(p[f] != q[f] for p, q in zip(lx, ly, strict=True) for f in LEVEL_FIELDS):
                        errors.append(f"{key}: host level readings differ")
                continue
            levels = trace.get("levels", [])
            if not trace.get("confidence") or not evictions:
                errors.append(f"{key}: confidence or EVICT records missing")
                continue
            drafted = sum(e["drafted"] for e in evictions)
            if len(levels) != drafted or sum(e["kept"] for e in evictions) != x["proposed"]:
                errors.append(f"{key}: {len(levels)} levels, {drafted} drafted, proposed {x['proposed']}")
                continue
            # Width-0 target steps (output budget 1) draft nothing and record no EVICT decision.
            # Only the final step may be width 0; EVICT moves cycle boundaries, so whether it occurs
            # can differ from the reference.
            if any(c[0] == 0 for c in x["cycles"][:-1]):
                errors.append(f"{key}: width-0 step before the end")
            widths = [c[0] for c in x["cycles"] if c[0] > 0]
            if widths != [e["kept"] for e in evictions]:
                errors.append(f"{key}: cycle widths differ from recorded k")
            offset, kept_hist, mism_bits, mism_k, near = 0, [0, 0, 0, 0], 0, 0, 0
            for index, e in enumerate(evictions):
                # The output budget may shorten only the final drafted cycle (mtp_width).
                short_ok = index == len(evictions) - 1 and 1 <= e["drafted"] < key[1]
                if (e["drafted"] != key[1] and not short_ok) or len(e["top_bits"]) != e["drafted"]:
                    errors.append(f"{key}: drafted {e['drafted']} at depth {key[1]}")
                    break
                host = [lv["top1_log_prob"] for lv in levels[offset : offset + e["drafted"]]]
                offset += e["drafted"]
                if any(not math.isfinite(h) for h in host):
                    errors.append(f"{key}: non-finite host top-1")
                    break
                mism_bits += sum(bits_to_f32(b) != f32(h) for b, h in zip(e["top_bits"], host, strict=True))
                k, is_near = policy(host)
                near += is_near
                mism_k += k != e["kept"]
                kept_hist[e["kept"]] += 1
            if mism_bits or mism_k:
                errors.append(f"{key}: {mism_bits} device/host bit mismatches, {mism_k} k mismatches ({near} near edges)")
            if key[1] == 1 and (x["cycles"] != y["cycles"] or x["histogram"] != y["histogram"]):
                errors.append(f"{key}: MTP1 cycles changed under EVICT")
            truncating += key[1] > 1 and any(e["kept"] < e["drafted"] for e in evictions)
            summary[f"{key[0]} MTP{key[1]}"] = {
                "cycles": len(x["cycles"]), "ref_cycles": len(y["cycles"]),
                "accepted": x["accepted"], "ref_accepted": y["accepted"],
                "proposed": x["proposed"], "ref_proposed": y["proposed"], "drafted": drafted,
                "kept_histogram": kept_hist, "near_edge_levels": near,
            }
        if mode == "evict" and truncating == 0:
            errors.append("no MTP2/MTP3 case truncates: EVICT not exercised")
        print(f"{actual_path}: {len(a)} cases, mode {mode}")
        for k, v in summary.items():
            print(k, json.dumps(v))
    except (OSError, KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as error:
        errors.append(f"{type(error).__name__}: {error}")
    print("\n".join(errors) if errors else f"compare {mode} passed")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
