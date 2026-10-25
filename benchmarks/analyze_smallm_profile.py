"""Validate the physical factorial report and apply its preregistered speed gates."""

import argparse
import json
import math
import statistics
from pathlib import Path


def analyze(report):
    if report.get("status") != "complete" or report.get("parity") is not True:
        raise ValueError("incomplete or divergent factorial profile")
    samples = report.get("samples", [])
    if len(samples) != 160:
        raise ValueError("expected 32 warmups and 128 measurements")
    measured, warmups, sequence = {}, {}, []
    for sample in samples:
        rows, config, seconds = (sample.get(k) for k in ("rows", "config", "seconds"))
        if (
            type(rows) is not int
            or not 1 <= rows <= 4
            or type(config) is not int
            or not 0 <= config <= 3
            or type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds <= 0
            or sample.get("bit_exact") is not True
            or sample.get("blocks") != 4
            or type(sample.get("blocks")) is not int
            or sample.get("logits") != rows * 248320
            or sample.get("hidden") != 5120
            or sample.get("state_arrays") != 128
            or sample.get("batch") is not bool(config & 1)
            or sample.get("mb3") is not bool(config & 2)
        ):
            raise ValueError("invalid factorial sample")
        if sample.get("phase") == "warmup":
            warmups[rows, config] = warmups.get((rows, config), 0) + 1
        elif sample.get("phase") == "measure":
            round_ = sample.get("round")
            if type(round_) is not int or not 0 <= round_ < 8:
                raise ValueError("invalid factorial round")
            key = rows, config, round_
            if key in measured:
                raise ValueError("duplicate factorial measurement")
            measured[key] = seconds / 4
            sequence.append((rows, config, round_))
        else:
            raise ValueError("unknown factorial phase")
    if len(measured) != 128 or len(warmups) != 16 or set(warmups.values()) != {2}:
        raise ValueError("missing factorial cells or warmups")
    orders = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]
    if sequence != [(m, c, r) for r in range(8) for m in range(1, 5) for c in orders[r % 4]]:
        raise ValueError("measurements do not follow the balanced interleaved protocol")
    median = statistics.median
    result = []
    for rows in range(1, 5):
        baseline = [measured[rows, 0, r] for r in range(8)]
        for config in range(4):
            times = [measured[rows, config, r] for r in range(8)]
            gains = [a / b - 1 for a, b in zip(baseline, times, strict=True)]
            ab = [
                gains[r] for r in range(8) if orders[r % 4].index(0) < orders[r % 4].index(config)
            ]
            ba = [
                gains[r] for r in range(8) if orders[r % 4].index(0) > orders[r % 4].index(config)
            ]
            drift_a = median(baseline[4:]) / median(baseline[:4]) - 1
            drift_b = median(times[4:]) / median(times[:4]) - 1
            stable = max(abs(drift_a), abs(drift_b)) <= 0.03
            useful = bool(
                config
                and median(gains) >= 0.03
                and sum(g > 0 for g in gains) >= 7
                and median(ab) >= 0.03
                and median(ba) >= 0.03
            )
            result.append(
                {
                    "rows": rows,
                    "config": config,
                    "median_block_ms": median(times) * 1000,
                    "rho_vs_AR": median(times) / median([measured[1, config, r] for r in range(8)]),
                    "paired_gain_percent": median(gains) * 100,
                    "wins": sum(g > 0 for g in gains),
                    "AB_percent": median(ab) * 100 if ab else None,
                    "BA_percent": median(ba) * 100 if ba else None,
                    "A_drift_percent": drift_a * 100,
                    "candidate_drift_percent": drift_b * 100,
                    "stable": stable,
                    "speed_gate_passed": stable and useful,
                }
            )
    return {
        "status": "complete",
        "parity": True,
        "comparisons": result,
        "scope": "teacher-forced target blocks; bridge speed remains unverified",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(json.loads(args.report.read_text())), indent=2))


if __name__ == "__main__":
    main()
