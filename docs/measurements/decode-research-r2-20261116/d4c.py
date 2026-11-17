#!/usr/bin/env python3
"""D4c: rows a certified *sampling* head must recompute exactly, on the D4 logits (CPU)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("d4", here / "d4.py")
d4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d4)
model, out = sys.argv[1], Path(sys.argv[2])
weights = d4.load_head(model)
rows, width = weights.shape
logits = np.concatenate(
    [
        np.load(out / f"{case['name']}.npz")["bits"].view(np.float16).astype(np.float32)
        for case in json.loads((out / "collect.json").read_text())["cases"]
    ]
)
gram = np.zeros((width + 1, width + 1))
wide = weights.astype(np.float64)
gram[:width, :width] = wide.T @ wide
gram[:width, width] = gram[width, :width] = wide.sum(0)
gram[width, width] = rows
rhs = np.concatenate([logits.astype(np.float64) @ wide, logits.sum(1, keepdims=True)], 1)
states = np.linalg.solve(gram, rhs.T).T[:, :width].astype(np.float32)
del wide, gram
norm = np.linalg.norm(states.astype(np.float64), axis=1)
ordered = np.sort(logits, axis=1)[:, ::-1]
thresholds = {
    "top80_set": ordered[:, 79],
    "top50_set": ordered[:, 49],
    "window_T0.2_k80": np.maximum(ordered[:, 0] - 6.0, ordered[:, 79]),
    "window_T0.1_k50": np.maximum(ordered[:, 0] - 3.0, ordered[:, 49]),
    "argmax": ordered[:, 0],
}
report = {}
for name in ("int8_row", "int4_g64"):
    error = d4.quantized_error(weights, name)
    error_norm = np.linalg.norm(error.astype(np.float64), axis=1)
    upper = logits - states @ error.T
    del error
    upper += (norm[:, None] * error_norm[None, :] * 1.002 + 1e-3).astype(np.float32)
    proxy = rows * d4.proxy_bytes(name, width)
    report[name] = {}
    for key, threshold in thresholds.items():
        counts = (upper >= threshold[:, None]).sum(1)
        total = proxy + counts * width * 2
        report[name][key] = {
            "candidates_median": float(np.median(counts)),
            "candidates_p90": float(np.quantile(counts, 0.9)),
            "candidates_p99": float(np.quantile(counts, 0.99)),
            "candidates_max": int(counts.max()),
            "bytes_median_MB": float(np.median(total) / 1e6),
            "bytes_p90_MB": float(np.quantile(total, 0.9) / 1e6),
            "bytes_mean_MB": float(total.mean() / 1e6),
        }
        print(name, key, json.dumps(report[name][key]), flush=True)
    del upper
(out / "topk-cert-candidates.json").write_text(json.dumps(report, indent=2))
