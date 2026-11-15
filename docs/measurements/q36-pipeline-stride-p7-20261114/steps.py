"""Split each measured decode window at the per-token GPU bubble; per-step structure and bubble anatomy."""
import json, statistics, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import cb_records
res = {}
for d in sorted(p for p in Path(__file__).parent.iterdir() if p.is_dir() and p.name.startswith("p") and (p / "cb.bin").exists()):
    run = json.loads((d / "run.json").read_text()); rows = cb_records.read(d / "cb.bin")
    agg = {"bubble_us": [], "commits": [], "busy_ms": [], "first_cb_us": [], "delta_to_commit_us": [], "top": Counter()}
    for name, req in run["requests"].items():
        if name == "warmup": continue
        a, b = req["deltas"][0], req["complete"]
        w = [r for r in rows if a <= r["commit"] <= b]
        # bubble = first commit whose commit time is >300 us after the latest GPU end of all earlier CBs
        end = None; starts = []
        for i, r in enumerate(w):
            if end is not None and r["commit"] - end > 300e-6:
                starts.append((i, r["commit"] - end, end))
            if r["gpu_end"] > (end or 0): end = r["gpu_end"]
        dl = req["deltas"]
        for (i, bub, prev_end), (j, _, _) in zip(starts, starts[1:]):
            step = w[i:j]
            agg["bubble_us"].append(bub * 1e6); agg["commits"].append(len(step))
            agg["busy_ms"].append(sum(max(0, r["gpu_end"] - r["gpu_start"]) for r in step) * 1e3)
            nxt = [t for t in dl if t >= prev_end]
            if nxt: agg["delta_to_commit_us"].append((step[0]["commit"] - nxt[0]) * 1e6)
            for k, r in enumerate(step):
                agg["top"][k] += (r["gpu_end"] - r["gpu_start"]) * 1e3
    n = len(agg["commits"])
    q = lambda v: [round(x, 1) for x in statistics.quantiles(v, n=4)]
    res[d.name] = {"steps": n, "bubble_us_quartiles": q(agg["bubble_us"]), "commits_per_step_mode": Counter(agg["commits"]).most_common(3),
                   "gpu_sum_ms_per_step_quartiles": q(agg["busy_ms"]),
                   "delta_seen_to_next_first_commit_us_quartiles": q(agg["delta_to_commit_us"]),
                   "top_positions_ms_per_step": [(k, round(v / n, 3)) for k, v in agg["top"].most_common(12)]}
Path(Path(__file__).parent / "steps.json").write_text(json.dumps(res, indent=1) + "\n")
for k, v in res.items(): print(k, json.dumps(v))
