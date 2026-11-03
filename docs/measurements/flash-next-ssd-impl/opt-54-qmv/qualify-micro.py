"""Qualify fixed EXL3 micro measurements; never imply tokens/sec."""

import math
import statistics


def qualify(data):
    assert data["status"] == "complete" and data["parity"] == "byte_exact_independent"
    assert data["no_timed_io"] is True
    assert data["samples_per_cell"] == 8 and data["forwards_per_sample"] == 256
    cells = data["records"]
    assert isinstance(cells, list) and len(cells) == 6
    expected = {(layer, rows) for layer in [0, 23, 47] for rows in [1, 2]}
    seen = set()
    metrics = []
    for cell in cells:
        key = cell["layer"], cell["rows"]
        assert type(key[0]) is int and type(key[1]) is int
        assert key in expected and key not in seen
        seen.add(key)
        ids = cell["route_ids"]
        assert isinstance(ids, list) and len(ids) == key[1] * 10
        assert all(type(i) is int and 0 <= i < 512 for i in ids)
        assert all(len(set(ids[start : start + 10])) == 10 for start in range(0, len(ids), 10))
        assert key[1] == 1 or len(set(ids)) < len(ids)
        assert type(cell["io_bytes_after_warm"]) is int and cell["io_bytes_after_warm"] > 0
        samples = cell["samples"]
        assert len(samples) == 8
        indexed = {}
        for sample in samples:
            pair, candidate, seconds = sample["pair"], sample["candidate"], sample["seconds"]
            assert type(pair) is int and pair in range(4) and type(candidate) is bool
            assert (pair, candidate) not in indexed
            assert type(seconds) in [int, float] and math.isfinite(seconds) and seconds > 0
            indexed[pair, candidate] = seconds
        arms = [[indexed[pair, c] for pair in range(4)] for c in [False, True]]
        gain = 100 * (statistics.median(arms[0]) / statistics.median(arms[1]) - 1)
        drift = [100 * (max(arm) / min(arm) - 1) for arm in arms]
        metrics.append(
            {"layer": key[0], "rows": key[1], "gain_percent": gain, "drift_percent": drift}
        )
    assert seen == expected
    stable = all(max(cell["drift_percent"]) <= 5 for cell in metrics)
    thresholds = all(cell["gain_percent"] >= 1 for cell in metrics)
    return {
        "status": "eligible_micro_only"
        if stable and thresholds
        else ("rejected_micro" if stable else "inconclusive_drift"),
        "full_model_eligible": stable and thresholds,
        "throughput_gain_confirmed": False,
        "cells": metrics,
    }
