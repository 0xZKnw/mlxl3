#!/usr/bin/env python3
"""F1 L1-EXACT-FEASIBILITY: can a certified int8 head reproduce the exact sampled law?

On the 512 real LFM3 rows of D4 (CPU): candidates under a tile-max threshold,
uniqueness of the top-k F16 log-probability set, and whether a rigorous
interval of the log-sum-exp has a single F16 rounding.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

d4_dir = Path(sys.argv[3])
spec = importlib.util.spec_from_file_location("d4", d4_dir / "d4.py")
d4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d4)
model, data = sys.argv[1], Path(sys.argv[2])
weights = d4.load_head(model)
rows, width = weights.shape
logits = np.concatenate(
    [
        np.load(data / f"{case['name']}.npz")["bits"].view(np.float16).astype(np.float32)
        for case in json.loads((data / "collect.json").read_text())["cases"]
    ]
)
wide = weights.astype(np.float64)
gram = np.zeros((width + 1, width + 1))
gram[:width, :width] = wide.T @ wide
gram[:width, width] = gram[width, :width] = wide.sum(0)
gram[width, width] = rows
rhs = np.concatenate([logits.astype(np.float64) @ wide, logits.sum(1, keepdims=True)], 1)
solution = np.linalg.solve(gram, rhs.T).T
states = solution[:, :width].astype(np.float32)
residual = np.abs(states @ weights.T + solution[:, width : width + 1].astype(np.float32) - logits).max(1)
del wide, gram
keep = residual <= 0.05
logits, states = logits[keep], states[keep]
norm = np.linalg.norm(states.astype(np.float64), axis=1)
error = d4.quantized_error(weights, "int8_row")
error_norm = np.linalg.norm(error.astype(np.float64), axis=1)
approx = logits - states @ error.T
del error
bound = (norm[:, None] * error_norm[None, :] * 1.002 + 1e-3).astype(np.float32)
lower, upper = approx - bound, approx + bound
slop = 2.0**-12
report = {"positions": int(len(logits)), "vocab": int(rows), "bound_median": float(np.median(bound))}
lg = logits.astype(np.float64)
m = lg.max(1, keepdims=True)
lse = (m[:, 0] + np.log(np.exp(lg - m).sum(1)))
c16 = lse.astype(np.float16).astype(np.float64)
lp16 = (logits.astype(np.float64) - c16[:, None]).astype(np.float16).astype(np.float32)
for k in (80, 50, 20):
    exact_tau = -np.sort(-lower, axis=1)[:, k - 1]
    tiles = lower.reshape(len(lower), -1, 64).max(2)
    tile_tau = -np.sort(-tiles, axis=1)[:, k - 1]
    tiles8 = lower.reshape(len(lower), -1, 8).max(2)
    tile8_tau = -np.sort(-tiles8, axis=1)[:, k - 1]
    entry = {}
    for name, tau in (("exact", exact_tau), ("tile64", tile_tau), ("tile8", tile8_tau)):
        cand = upper >= tau[:, None]
        counts = cand.sum(1)
        tail_lo = np.where(cand, 0.0, np.exp(lower.astype(np.float64) - m)).sum(1)
        tail_hi = np.where(cand, 0.0, np.exp(upper.astype(np.float64) - m)).sum(1)
        exact_part = np.where(cand, np.exp(lg - m), 0.0).sum(1)
        lo = m[:, 0] + np.log(exact_part + tail_lo) - slop
        hi = m[:, 0] + np.log(exact_part + tail_hi) + slop
        same = lo.astype(np.float16) == hi.astype(np.float16)
        width_lse = hi - lo
        ordered = -np.sort(-lp16, axis=1)
        unique = ordered[:, k - 1] != ordered[:, k]
        # Every true top-k row must be a candidate (sanity of the bounds).
        topk_rows = np.argsort(-logits, axis=1)[:, :k]
        covered = np.take_along_axis(cand, topk_rows, 1).all(1)
        entry[name] = {
            "candidates_median": float(np.median(counts)),
            "candidates_p90": float(np.quantile(counts, 0.9)),
            "candidates_max": int(counts.max()),
            "topk_covered": float(covered.mean()),
            "lse_interval_median": float(np.median(width_lse)),
            "lse_interval_p90": float(np.quantile(width_lse, 0.9)),
            "c16_certified": float(same.mean()),
            "kth_unique": float(unique.mean()),
            "both": float((same & unique).mean()),
        }
    report[f"k{k}"] = entry
print(json.dumps(report, indent=1))
json.dump(report, open(Path(sys.argv[4]), "w"), indent=1)

# F1b (same preregistration, design check): log-sum-exp *estimate* from the
# exact candidates plus the screening approximations of the other rows.
est = {}
for k in (80, 50):
    tiles = lower.reshape(len(lower), -1, 64).max(2)
    tau = -np.sort(-tiles, axis=1)[:, k - 1]
    cand = upper >= tau[:, None]
    a32 = approx.astype(np.float32)
    mm = m.astype(np.float32)
    s = np.where(cand, np.exp((logits - mm).astype(np.float32)), np.exp((a32 - mm).astype(np.float32))).sum(1, dtype=np.float32)
    c_est = (mm[:, 0] + np.log(s)).astype(np.float32)
    same = c_est.astype(np.float16) == lse.astype(np.float16)
    err = np.abs(c_est.astype(np.float64) - lse)
    # TV between the T0.2 top-k laws built with c16 and with fl16(c_est).
    tv = []
    for p in range(len(logits)):
        top = np.argsort(-logits[p])[:k]
        for c in (c16[p], float(c_est[p].astype(np.float16))):
            pass
        a = (logits[p, top].astype(np.float64) - c16[p]).astype(np.float16).astype(np.float64)
        b = (logits[p, top].astype(np.float64) - float(c_est[p].astype(np.float16))).astype(np.float16).astype(np.float64)
        wa = np.exp((a - a.max()) / 0.2); wa /= wa.sum()
        wb = np.exp((b - b.max()) / 0.2); wb /= wb.sum()
        tv.append(0.5 * np.abs(wa - wb).sum())
    tv = np.asarray(tv)
    est[f"k{k}"] = {
        "c16_estimate_equal": float(same.mean()),
        "lse_abs_error_median": float(np.median(err)),
        "lse_abs_error_max": float(err.max()),
        "tv_T0.2_mean": float(tv.mean()),
        "tv_T0.2_max": float(tv.max()),
        "tv_zero_share": float((tv == 0).mean()),
    }
print(json.dumps(est, indent=1))
report["estimate"] = est
json.dump(report, open(Path(sys.argv[4]), "w"), indent=1)
