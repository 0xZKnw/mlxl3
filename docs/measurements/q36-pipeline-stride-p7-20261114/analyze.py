"""Q36-CB-GPU-TIME-P4 analysis: GPU busy union, idle gaps (CPU-late vs launch latency), per-token figures."""
import json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cb_records

def window_stats(rows, a, b, tokens):
    w = [r for r in rows if a <= r["commit"] <= b]
    busy = [r for r in w if r["gpu_end"] > r["gpu_start"]]
    busy.sort(key=lambda r: r["gpu_start"])
    union = 0.0; cpu_late = 0.0; launch = 0.0; end = None; late_gaps = []
    for r in busy:
        s, e = r["gpu_start"], r["gpu_end"]
        if end is None:
            union += e - s; end = e; continue
        if s > end:
            idle = s - end
            late = min(idle, max(0.0, r["commit"] - end))  # GPU idle because the CB was not committed yet
            cpu_late += late; launch += idle - late
            if late > 0: late_gaps.append(late)
            union += e - s; end = e
        elif e > end:
            union += e - end; end = e
    wall = b - a
    # GPU idle before the first and after the last busy CB inside the window
    edge = wall - union - cpu_late - launch
    lead = [r["gpu_start"] - r["commit"] for r in busy]
    return {"wall_ms_per_token": wall / tokens * 1e3, "gpu_union_ms_per_token": union / tokens * 1e3,
            "gpu_busy_fraction": union / wall, "idle_cpu_late_ms_per_token": cpu_late / tokens * 1e3,
            "idle_launch_ms_per_token": launch / tokens * 1e3, "idle_edges_ms_per_token": edge / tokens * 1e3,
            "cbs_per_token": len(w) / tokens, "nonempty_cbs_per_token": len(busy) / tokens,
            "cpu_late_gaps_per_token": len(late_gaps) / tokens,
            "cpu_late_gap_us_median": statistics.median(late_gaps) * 1e6 if late_gaps else 0.0,
            "commit_to_gpu_start_us_median": statistics.median(lead) * 1e6,
            "gpu_sum_ms_per_token": sum(r["gpu_end"] - r["gpu_start"] for r in busy) / tokens * 1e3}

out = {}
for d in sorted(p for p in Path(__file__).parent.iterdir() if p.is_dir() and p.name.startswith("p")):
    run = json.loads((d / "run.json").read_text())
    assert run["status"] == "complete", d
    entry = {"flag": run["flag"], "inject": run["inject"], "requests": {}}
    rows = cb_records.read(d / "cb.bin") if run["inject"] else None
    if rows is not None:
        assert all(r["status"] == 4 for r in rows)
        # every record must belong to a commit after load and before shutdown
        assert all(r["commit"] <= r["gpu_start"] <= r["gpu_end"] <= r["handler"] for r in rows if r["gpu_end"] > r["gpu_start"])
    for name, req in run["requests"].items():
        s = req["stats"]; e = {"decode_tps": s["decode_tps"], "token_hash": req["token_hash"], "text_len": len(req["text"])}
        if rows is not None:
            e |= window_stats(rows, req["deltas"][0], req["complete"], s["decode_tokens"])
        entry["requests"][name] = e
    out[d.name] = entry
hashes = {e["token_hash"] for v in out.values() for e in v["requests"].values()}
texts = {(e["token_hash"], e["text_len"]) for v in out.values() for e in v["requests"].values()}
out["_parity"] = {"distinct_token_hashes": sorted(hashes), "identical": len(hashes) == 1 and len(texts) == 1}
Path(Path(__file__).parent / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
keys = ["decode_tps", "wall_ms_per_token", "gpu_union_ms_per_token", "gpu_busy_fraction", "idle_cpu_late_ms_per_token",
        "idle_launch_ms_per_token", "idle_edges_ms_per_token", "cbs_per_token", "nonempty_cbs_per_token", "cpu_late_gaps_per_token",
        "cpu_late_gap_us_median", "commit_to_gpu_start_us_median", "gpu_sum_ms_per_token"]
for name, v in out.items():
    if name.startswith("_"): print(name, v); continue
    for r, e in v["requests"].items():
        if r == "warmup": continue
        print(name, r, " ".join(f"{k}={e[k]:.3f}" for k in keys if k in e))
