"""Qualify the preregistered CPU-known EXL3 micro; never infer generation speed."""

import math
import statistics
from collections import Counter


def qualify(data, gated=False):
    assert data["status"] == "complete" and data["parity"] == "byte_exact_independent"
    assert data["no_timed_io"] is True
    forwards = 1024 if gated else 128
    assert data["samples_per_cell"] == 8 and data["forwards_per_sample"] == forwards
    expected = {
        (layer, rows, mixed)
        for layer in [0, 23, 47]
        for rows in [1, 2, 4, 8]
        for mixed in [False, True]
    }
    assert len(data["records"]) == len(expected)
    seen, metrics = set(), []
    for cell in data["records"]:
        key = cell["layer"], cell["rows"], cell["mixed"]
        assert type(key[0]) is int and type(key[1]) is int and type(key[2]) is bool
        assert key in expected and key not in seen
        seen.add(key)
        ids, slots = cell["route_ids"], cell["known_slots"]
        assert len(ids) == key[1] * 10 and all(type(i) is int and 0 <= i < 512 for i in ids)
        assert all(len(set(ids[i : i + 10])) == 10 for i in range(0, len(ids), 10))
        assert 0 < len(slots) <= len(ids) and all(type(i) is int and 0 <= i < 53 for i in slots)
        assert type(cell["io_bytes_after_warm"]) is int and cell["io_bytes_after_warm"] > 0
        grouped = len(set(slots)) < len(slots)
        if gated:
            chunks = sum((count + 3) // 4 for count in Counter(slots).values())
            grouped = len(slots) >= 16 and (len(slots) - chunks) * 5 >= len(slots)
            assert type(cell["shared_enabled"]) is bool and cell["shared_enabled"] == grouped
        indexed = {}
        assert len(cell["samples"]) == 8
        for sample in cell["samples"]:
            pair, candidate, seconds = sample["pair"], sample["candidate"], sample["seconds"]
            assert type(pair) is int and pair in range(4) and type(candidate) is bool
            assert (pair, candidate) not in indexed
            assert type(seconds) in [int, float] and math.isfinite(seconds) and seconds > 0
            indexed[pair, candidate] = seconds
        arms = [[indexed[pair, candidate] for pair in range(4)] for candidate in [False, True]]
        gains = [100 * (indexed[pair, False] / indexed[pair, True] - 1) for pair in range(4)]
        drift = [100 * (max(arm) / min(arm) - 1) for arm in arms]
        metrics.append(
            {
                "layer": key[0],
                "rows": key[1],
                "mixed": key[2],
                "grouped": grouped,
                "median_baseline_seconds_per_forward": statistics.median(arms[0]) / forwards,
                "median_candidate_seconds_per_forward": statistics.median(arms[1]) / forwards,
                "median_gain_percent": 100
                * (statistics.median(arms[0]) / statistics.median(arms[1]) - 1),
                "paired_gain_percent": gains,
                "drift_percent": drift,
                "gain_spread_points": max(gains) - min(gains),
                "threshold_passed": min(gains) >= (1 if grouped else -1),
            }
        )
    assert seen == expected and any(cell["grouped"] for cell in metrics)
    stable = all(
        max(cell["drift_percent"]) <= 5 and cell["gain_spread_points"] <= 3 for cell in metrics
    )
    passed = stable and all(cell["threshold_passed"] for cell in metrics)
    return {
        "status": "eligible_micro_only"
        if passed
        else "rejected_micro"
        if stable
        else "inconclusive_drift",
        "full_model_eligible": passed,
        "throughput_gain_confirmed": False,
        "cells": metrics,
    }
