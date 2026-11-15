"""Q36-GPU-ATTRIBUTION-P8: attribute CB GPU time to the timeline blocks of q36_decode_block_timeline."""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cb_records  # noqa: E402

COMPLETED = 4
BANDWIDTH = 153e9  # bytes/s, M5 measured peak used in RESEARCH-...-AR-BUBBLE-SOURCE
MIB = 1 << 20
# Weight bytes per token from the safetensors headers (RESEARCH-2026-11-14-Q36-AR-BUBBLE-SOURCE).
# gdn_recurrence: f32 recurrent state 32x128x128 read+written per GDN layer (4 MiB x 30), weights none.
FLOOR_MIB = {
    "gdn_in": 368.4, "gdn_recurrence": 120.0, "gdn_out": 120.4, "attn_mixer": 90.3 + 40.1,
    "moe_router": 40.0, "moe_experts": 295.7, "moe_shared": 60.6, "head": 364.2, "embed": 0.0,
}


def union(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def analyze(run):
    timeline = json.loads((run / "timeline.json").read_text())
    records = cb_records.read(run / "cb.bin")
    warmup, measured = timeline["warmup_steps"], timeline["measured_steps"]
    rows = timeline["rows"]
    if not rows or measured < 1:
        raise ValueError("empty timeline")
    for a, b in zip(rows, rows[1:]):
        if not (a["t0_ns"] <= a["t1_ns"] <= b["t0_ns"]):
            raise ValueError(f"overlapping or unordered rows {a} {b}")
    starts = [r["t0_ns"] for r in rows]
    per_step = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(int)
    assigned = 0
    import bisect
    buckets = defaultdict(list)
    for rec in records:
        commit = rec["commit"] * 1e9
        i = bisect.bisect_right(starts, commit) - 1
        if i < 0 or commit > rows[i]["t1_ns"]:
            continue
        if rec["status"] != COMPLETED or rec["gpu_end"] < rec["gpu_start"]:
            raise ValueError(f"CB not completed or negative duration: {rec}")
        buckets[i].append((rec["gpu_start"], rec["gpu_end"]))
        assigned += 1
    for i, row in enumerate(rows):
        if row["step"] < warmup:
            continue
        ms = union(buckets[i]) * 1e3
        per_step[row["step"]][row["block"]] += ms
        counts[row["block"]] += len(buckets[i])
    steps = sorted(per_step)
    if len(steps) != measured:
        raise ValueError(f"expected {measured} measured steps, found {len(steps)}")
    blocks = sorted({b for s in steps for b in per_step[s]})
    out = {"run": str(run), "cb_records": len(records), "cb_assigned": assigned, "blocks": {}}
    total = [sum(per_step[s].values()) for s in steps]
    for block in blocks:
        values = [per_step[s].get(block, 0.0) for s in steps]
        mean = statistics.fmean(values)
        floor = FLOOR_MIB.get(block, 0.0) * MIB / BANDWIDTH * 1e3
        out["blocks"][block] = {
            "ms_per_token_mean": mean, "ms_per_token_min": min(values), "ms_per_token_max": max(values),
            "cbs_per_token": counts[block] / measured, "floor_ms": floor, "excess_ms": mean - floor,
            "effective_gbps": FLOOR_MIB.get(block, 0.0) * MIB / (mean / 1e3) / 1e9 if mean > 0 else None,
        }
    out["total_ms_per_token_mean"] = statistics.fmean(total)
    out["total_ms_per_token_min"] = min(total)
    out["total_floor_ms"] = sum(FLOOR_MIB.values()) * MIB / BANDWIDTH * 1e3
    return out


if __name__ == "__main__":
    results = [analyze(Path(p)) for p in sys.argv[1:]]
    Path(Path(__file__).parent / "analysis.json").write_text(json.dumps(results, indent=1) + "\n")
    for r in results:
        print(r["run"], f"total {r['total_ms_per_token_mean']:.3f} ms/token (min {r['total_ms_per_token_min']:.3f}),"
              f" floor {r['total_floor_ms']:.2f}, CBs {r['cb_assigned']}/{r['cb_records']}")
        for block, v in sorted(r["blocks"].items(), key=lambda kv: -kv[1]["ms_per_token_mean"]):
            gbps = f"{v['effective_gbps']:.0f} GB/s" if v["effective_gbps"] else "-"
            print(f"  {block:15s} {v['ms_per_token_mean']:6.3f} ms  floor {v['floor_ms']:5.3f}  "
                  f"excess {v['excess_ms']:6.3f}  {gbps:>9s}  CB/token {v['cbs_per_token']:.1f}")
