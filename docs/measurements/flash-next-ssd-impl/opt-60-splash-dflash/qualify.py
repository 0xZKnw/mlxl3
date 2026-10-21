"""Paired BQ/QR/RA gates; a completed benchmark is not itself a gain."""

import math


def qualify(rows, baseline, candidate):
    if len(rows) != 8 or baseline == candidate:
        raise ValueError("eight ordered paired measurements required")
    schedule = [
        (p, i, v)
        for p in ("MAIN", "HOLD")
        for i, v in enumerate((baseline, candidate, candidate, baseline))
    ]
    if [(r["prompt"], r["index"], r["variant"]) for r in rows] != schedule:
        raise ValueError("foreign or incomplete schedule")
    if len({r["binary_sha256"] for r in rows}) != 1:
        raise ValueError("mixed engine binaries")
    for row in rows:
        if not (
            row["status"] == "validated"
            and row["reaped"] is True
            and row["reader_joined"] is True
            and type(row["peak_bytes"]) is int
            and 0 < row["peak_bytes"] <= 12_000_000_000
            and type(row["swapouts_bytes"]) is int
            and row["swapouts_bytes"] == 0
            and type(row["tps"]) in (int, float)
            and math.isfinite(row["tps"])
            and row["tps"] > 0
            and type(row["seconds"]) in (int, float)
            and math.isfinite(row["seconds"])
            and row["seconds"] > 0
            and math.isclose(row["tps"], 127 / row["seconds"], rel_tol=1e-5)
            and type(row["binary_sha256"]) is str
            and len(row["binary_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in row["binary_sha256"])
            and type(row["token_hash"]) is str
            and len(row["token_hash"]) == 16
            and all(c in "0123456789abcdef" for c in row["token_hash"])
            and type(row["text"]) is str
            and row["text"]
        ):
            raise ValueError("incomplete, swapped or invalid measurement")
    pairs, drift = [], {}
    for prompt, group in zip(("MAIN", "HOLD"), (rows[:4], rows[4:]), strict=True):
        if len({(r["token_hash"], r["text"]) for r in group}) != 1:
            raise ValueError("target output divergence")
        gains = []
        for a, b in (group[:2], group[2:]):
            h, c = (a, b) if a["variant"] == baseline else (b, a)
            gain = 100 * (c["tps"] / h["tps"] - 1)
            gains.append(gain)
            pairs.append(
                {
                    "prompt": prompt,
                    "baseline_tps": h["tps"],
                    "candidate_tps": c["tps"],
                    "gain_percent": gain,
                }
            )
        controls = [group[0]["tps"], group[3]["tps"]]
        drift[prompt] = {
            "control_percent": 100 * (max(controls) / min(controls) - 1),
            "gain_spread_points": max(gains) - min(gains),
        }
    stable = all(v["control_percent"] <= 5 and v["gain_spread_points"] <= 3 for v in drift.values())
    gain = stable and all(p["gain_percent"] >= 1 for p in pairs)
    return {
        "status": "validated_gain" if gain else "rejected_gain" if stable else "inconclusive_drift",
        "confirmed_gain": gain,
        "baseline": baseline,
        "candidate": candidate,
        "pairs": pairs,
        "drift": drift,
        "memory_status": "requires_campaign_postflight_before_publication",
    }
