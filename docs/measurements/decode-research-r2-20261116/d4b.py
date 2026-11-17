#!/usr/bin/env python3
"""D4b SAMPLER-S2-TV-20261116: law change of a top-K normalized sampler on real logits (CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
manifest = json.loads((out / "collect.json").read_text())
rows, histories = [], []
for case in manifest["cases"]:
    data = np.load(out / f"{case['name']}.npz")
    logits = data["bits"].view(np.float16)
    targets = data["targets"]
    for index in range(len(logits)):
        rows.append(logits[index])
        # Row `index` predicts targets[index]; the penalty window is the 20
        # generated tokens before it.
        histories.append(targets[max(0, index - 20) : index])


def law(values, history, temperature, top_k, penalty):
    values = values.copy()
    if penalty != 1 and len(history):
        unique = np.unique(history)
        chosen = values[unique]
        values[unique] = np.where(chosen < 0, chosen * penalty, chosen / penalty)
    order = np.lexsort((np.arange(len(values)), -values))[:top_k]
    kept = values[order].astype(np.float64)
    weights = np.exp((kept - kept.max()) / temperature)
    return order, weights / weights.sum()


def reference(logits, history, temperature, top_k, penalty):
    wide = logits.astype(np.float64)
    c = np.float16(wide.max() + np.log(np.exp(wide - wide.max()).sum()))
    values = (logits - c).astype(np.float16).astype(np.float32)  # fl16(l - c16)
    return law(values, history, temperature, top_k, penalty)


def candidate(logits, history, temperature, top_k, penalty, keep):
    wide = logits.astype(np.float64)
    best = np.sort(wide)[-keep:]
    c = best.max() + np.log(np.exp(best - best.max()).sum())
    return law(wide - c, history, temperature, top_k, penalty)


report = {}
for name, (temperature, top_k, penalty) in {
    "app_default": (0.2, 80, 1.05),
    "lfm3_general": (0.1, 50, 1.1),
}.items():
    for keep in (top_k, 256):
        distances, set_changes, top1_changes = [], 0, 0
        for logits, history in zip(rows, histories):
            ids_a, p_a = reference(logits, history, temperature, top_k, penalty)
            ids_b, p_b = candidate(logits, history, temperature, top_k, penalty, keep)
            law_a = dict(zip(ids_a.tolist(), p_a.tolist()))
            law_b = dict(zip(ids_b.tolist(), p_b.tolist()))
            keys = set(law_a) | set(law_b)
            distances.append(0.5 * sum(abs(law_a.get(k, 0) - law_b.get(k, 0)) for k in keys))
            set_changes += set(law_a) != set(law_b)
            top1_changes += int(ids_a[np.argmax(p_a)] != ids_b[np.argmax(p_b)])
        distances = np.asarray(distances)
        report[f"{name}_K{keep}"] = {
            "rows": len(rows),
            "tv_median": float(np.median(distances)),
            "tv_p90": float(np.quantile(distances, 0.9)),
            "tv_p99": float(np.quantile(distances, 0.99)),
            "tv_max": float(distances.max()),
            "tv_mean": float(distances.mean()),
            "topk_set_differs": int(set_changes),
            "mode_differs": int(top1_changes),
        }
        print(f"{name}_K{keep}", json.dumps(report[f"{name}_K{keep}"]), flush=True)
(out / "sampler-s2-tv.json").write_text(json.dumps(report, indent=2))
