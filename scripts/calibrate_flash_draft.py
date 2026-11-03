"""Fit greedy DSpark STS and a measured B1..B4 cost table; never run a model."""

import argparse
import hashlib
import json
import math
from pathlib import Path

TEMPERATURES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)


def calibrated_probability(probability, temperature):
    if (
        type(probability) not in (int, float)
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
        or type(temperature) not in (int, float)
        or not math.isfinite(temperature)
        or not 0.25 <= temperature <= 8
    ):
        raise ValueError("invalid probability or temperature")
    probability = min(max(probability, 1e-6), 1 - 1e-6)
    value = (math.log(probability) - math.log1p(-probability)) / temperature
    tail = math.exp(-abs(value))
    return 1 / (1 + tail) if value >= 0 else tail / (1 + tail)


def prefix_ece(probabilities, labels):
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("empty or incomplete calibration labels")
    totals, counts = [0.0] * 10, [0] * 10
    for probability, label in zip(probabilities, labels, strict=True):
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
            or type(label) is not int
            or label not in (0, 1)
        ):
            raise ValueError("invalid calibration observation")
        bucket = min(int(probability * 10), 9)
        totals[bucket] += probability - label
        counts[bucket] += 1
    return sum(abs(total) for total, count in zip(totals, counts, strict=True) if count) / len(
        labels
    )


def fit_temperatures(rows):
    if len(rows) < 32:
        raise ValueError("at least 32 complete blocks required for calibration")
    calibrated_survival = [1.0] * len(rows)
    raw_survival = [1.0] * len(rows)
    temperatures, diagnostics = [], []
    for position in range(3):
        labels = [int(row["accepted"] > position) for row in rows]
        raw = [row["confidence"][position] for row in rows]
        candidates = []
        for temperature in TEMPERATURES:
            prediction = [
                previous * calibrated_probability(p, temperature)
                for previous, p in zip(calibrated_survival, raw, strict=True)
            ]
            candidates.append(
                (
                    prefix_ece(prediction, labels),
                    abs(math.log(temperature)),
                    temperature,
                    prediction,
                )
            )
        error, _, temperature, prediction = min(candidates)
        raw_survival = [previous * p for previous, p in zip(raw_survival, raw, strict=True)]
        temperatures.append(temperature)
        diagnostics.append(
            {
                "position": position,
                "before_ece": prefix_ece(raw_survival, labels),
                "after_ece": error,
            }
        )
        calibrated_survival = prediction
    return temperatures, diagnostics


def build_profile(arms):
    if (
        len(arms) != 4
        or any(type(arm.get("forced_prefix")) is not int for arm in arms)
        or {arm.get("forced_prefix") for arm in arms} != set(range(4))
    ):
        raise ValueError("require exactly one arm for each prefix 0..3")
    identity = arms[0].get("identity")
    if not isinstance(identity, dict) or not identity:
        raise ValueError("missing calibration identity")
    costs, cost_counts = [0.0] * 4, [0] * 4
    calibration, prompts = [], set()
    for arm in arms:
        if arm.get("status") != "complete_calibration_arm" or arm.get("identity") != identity:
            raise ValueError("incomplete or foreign calibration arm")
        prefix = arm["forced_prefix"]
        measured = [run for run in arm["runs"] if run.get("warmup") is False]
        if len(measured) < 2:
            raise ValueError("missing measured repetitions")
        for run in measured:
            prompt = run.get("prompt_id")
            if (
                not isinstance(prompt, str)
                or len(prompt) != 16
                or any(c not in "0123456789abcdef" for c in prompt)
                or not run.get("trace")
            ):
                raise ValueError("empty calibration run")
            prompts.add(run["prompt_id"])
            for row in run["trace"]:
                if (
                    type(row["drafted"]) is not int
                    or not 1 <= row["drafted"] <= 3
                    or type(row["selected"]) is not int
                    or row["selected"] != min(prefix, row["drafted"])
                    or type(row["accepted"]) is not int
                    or not 0 <= row["accepted"] <= row["selected"]
                    or len(row["confidence"]) != row["drafted"]
                ):
                    raise ValueError("invalid calibration round shape or counters")
                if row["drafted"] != 3:
                    continue
                seconds = row["round_seconds"]
                if (
                    type(seconds) not in (int, float)
                    or not math.isfinite(seconds)
                    or not 1e-9 <= seconds <= 360
                    or type(row["accepted"]) is not int
                    or not 0 <= row["accepted"] <= prefix
                    or row["selected"] != prefix
                    or len(row["confidence"]) != 3
                ):
                    raise ValueError("invalid full-width calibration round")
                for probability in row["confidence"]:
                    calibrated_probability(probability, 1.0)
                costs[prefix] += seconds
                cost_counts[prefix] += 1
                if prefix == 3:
                    calibration.append(row)
    if len(prompts) < 2 or any(count < 16 for count in cost_counts):
        raise ValueError("insufficient prompts or cost observations")
    temperatures, diagnostics = fit_temperatures(calibration)
    return {
        "status": "calibrated_greedy_profile",
        "version": 1,
        "query_width": 3,
        "identity": identity,
        "temperatures": temperatures,
        "round_seconds": [total / count for total, count in zip(costs, cost_counts, strict=True)],
        "cost_observations": cost_counts,
        "calibration_blocks": len(calibration),
        "calibration_prompt_ids": sorted(prompts),
        "calibration_ece": diagnostics,
        "scope": "greedy only; calibration ECE is a fit diagnostic, not held-out speed validation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arms", nargs=4, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump({"status": "incomplete"}, stream)
    result = {"status": "failed_or_interrupted"}
    try:
        arms, sources = [], []
        for path in args.arms:
            with path.open("rb") as stream:
                payload = stream.read((16 << 20) + 1)
            if len(payload) > 16 << 20:
                raise ValueError("calibration arm exceeds bounded input size")
            arms.append(json.loads(payload))
            sources.append({"file": str(path), "sha256": hashlib.sha256(payload).hexdigest()})
        result = build_profile(arms)
        result["sources"] = sources
    except BaseException as error:
        result.update(status="failed_or_interrupted", error=type(error).__name__, reason=str(error))
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
