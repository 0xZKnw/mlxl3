"""Attribute validated Flash decode traces to target windows; no GPU timing claim."""

from collections import Counter, defaultdict

EXPERT_BYTES = 1_248_012


def summarize(events, expected_counts, emitted):
    """Cross-check logical occurrences and requested record bytes against backend snapshots."""
    assert type(emitted) is int and emitted > 0
    assert len(expected_counts) == 48
    assert all(
        len(row) == 3 and all(type(x) is int and x >= 0 for x in row) for row in expected_counts
    )
    assert events and events[0]["name"] == "decode"
    windows, layers, output_steps = [], [], []
    histogram = Counter()
    cohort_multiplicities = Counter()
    per_width = defaultdict(lambda: [0, 0, 0])
    counts = [[0, 0, 0] for _ in range(48)]
    routes = defaultdict(list)
    widths = {}
    ordinary = set()
    for index, event in enumerate(events):
        name, parent, data = event["name"], event["parent"], event["data"]
        assert parent is None or type(parent) is int and 0 <= parent < index
        window = windows[parent] if parent is not None else None
        layer = layers[parent] if parent is not None else None
        step = output_steps[parent] if parent is not None else None
        if name == "output_step":
            step = index
        if name == "verify_draft":
            width = data["rows"]
            assert type(width) is int and 1 <= width <= 8
            window = index
            widths[index] = width
            histogram[width] += 1
        if name.startswith("layer_"):
            layer = int(name.removeprefix("layer_"))
            assert 0 <= layer < 48
        windows.append(window)
        layers.append(layer)
        output_steps.append(step)
        if name not in ("moe_forward_rows", "disk_read_batch"):
            continue
        assert layer is not None
        if window is None:
            assert step is not None
            key, width = ("ordinary", step), 1
            ordinary.add(key)
        else:
            key, width = ("verify", window), widths[window]
        if name == "moe_forward_rows":
            rows = data["rows"]
            assert type(rows) is int and 1 <= rows <= width
            ids, slots = data["ids"], data["slots_hit"]
            assert len(ids) == len(slots) == rows * 10
            assert all(type(id_) is int and 0 <= id_ < 512 for id_ in ids)
            assert all(type(hit) is bool for _, hit in slots)
            hits = sum(hit for _, hit in slots)
            misses = len(ids) - hits
            assert data["hot_rows"] == hits and data["cold_rows"] == misses
            routes[key, layer].extend(ids)
            cohort_multiplicities.update(Counter(ids).values())
            counts[layer][0] += hits
            counts[layer][1] += misses
            per_width[width][0] += hits
            per_width[width][1] += misses
        else:
            assert data["ok"] is True
            jobs = data["jobs"]
            assert jobs and len({slot for _, slot in jobs}) == len(jobs)
            read_bytes = len(jobs) * EXPERT_BYTES
            counts[layer][2] += read_bytes
            per_width[width][2] += read_bytes
    assert histogram, "no verification windows"
    assert counts == expected_counts, "route/I/O counters differ from backend snapshots"
    multiplicities = Counter()
    for window, width in [(("verify", k), v) for k, v in widths.items()] + [
        (k, 1) for k in ordinary
    ]:
        for layer in range(48):
            ids = routes[window, layer]
            assert len(ids) == width * 10, "incomplete target window"
            assert all(len(set(ids[i : i + 10])) == 10 for i in range(0, len(ids), 10))
            multiplicities.update(Counter(ids).values())
    totals = [sum(row[i] for row in counts) for i in range(3)]
    target_rows = sum(width * n for width, n in histogram.items()) + len(ordinary)
    assert totals[0] + totals[1] == target_rows * 48 * 10
    return {
        "emitted_decode_tokens": emitted,
        "verified_windows_by_rows": dict(sorted(histogram.items())),
        "ordinary_target_steps": len(ordinary),
        "target_rows": target_rows,
        "logical_hits": totals[0],
        "logical_misses": totals[1],
        "logical_hit_fraction": totals[0] / (totals[0] + totals[1]),
        "requested_record_bytes": totals[2],
        "requested_bytes_per_token": totals[2] / emitted,
        "requested_bytes_per_target_row": totals[2] / target_rows,
        "by_target_width_hits_misses_bytes": dict(sorted(per_width.items())),
        "expert_multiplicity_per_window_layer": dict(sorted(multiplicities.items())),
        "expert_multiplicity_per_moe_cohort": dict(sorted(cohort_multiplicities.items())),
        "backend_counters_exact": True,
        "limits": "Logical hits depend on cohort boundaries; record bytes are not physical NVMe bytes. Width1 includes ordinary steps, listed separately. Multiplicities are upper bounds before separating Hot/Cold compute phases. No exclusive GPU or throughput claim.",
    }
