"""Bound cohort availability in an existing trace, never predict native speed."""

import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRACE = ROOT / "docs/measurements/decode-profile-20261031/natural-r3/attribution-partial"


def window(ends, k, prefix, start, end, frontier):
    """Availability bounds with an unknown reader epoch inside its parent span."""
    assert ends and all(type(v) is int and v >= 0 for v in ends)
    assert type(k) is int and k > 0 and type(prefix) is bool
    assert all(type(v) is int for v in (start, end, frontier))
    assert 0 <= start <= frontier <= end and end - start >= max(ends)
    if k >= len(ends):
        return 0, 0
    ready = max(ends[:k]) if prefix else sorted(ends)[k - 1]
    last = max(ends)
    return tuple(
        max(0, min(last - ready, epoch + last - frontier)) for epoch in (start, end - last)
    )


def self_check():
    checked = 0
    for ends in itertools.product(range(1, 5), repeat=3):
        for start in (0, 2):
            end = start + max(ends) + 2
            for frontier in range(start, end + 1):
                for k, prefix in itertools.product((1, 2, 3, 4), (False, True)):
                    actual = window(ends, k, prefix, start, end, frontier)
                    scores = []
                    for epoch in range(start, end - max(ends) + 1):
                        useful = 0
                        for t in range(start, end):
                            done = [i for i, finish in enumerate(ends) if epoch + finish <= t]
                            ready = all(i in done for i in range(k)) if prefix else len(done) >= k
                            useful += ready and len(done) < len(ends) and t >= frontier
                        scores.append(useful)
                    assert actual == (min(scores), max(scores)), (ends, k, prefix)
                    checked += 1
    for args in [
        ([], 1, False, 0, 1, 0),
        ([1], 0, False, 0, 1, 0),
        ([-1], 1, False, 0, 1, 0),
        ([2], 1, False, 0, 1, 0),
        ([1], 1, False, 0, 1, 2),
        ([True], 1, False, 0, 1, 0),
    ]:
        try:
            window(*args)
        except AssertionError:
            continue
        raise AssertionError("invalid bounds accepted")
    return checked


def analyze():
    paths = [
        TRACE / name
        for name in (
            "partial-attribution.json",
            "io-tail-batches.json",
            "profile/trace_1-decode.json",
        )
    ]
    hashes = {}
    for p in paths:
        with p.open("rb") as stream:
            hashes[str(p.relative_to(ROOT))] = hashlib.file_digest(stream, "sha256").hexdigest()
    attribution, batches, events = [json.loads(p.read_text()) for p in paths]
    assert attribution["pid"] == 27154 and attribution["complete_steps"] == 40
    assert attribution["clock_fit_halfwidth_ns"] == 1375
    steps = {e["data"]["step"]: e for e in events if e["name"] == "output_step"}
    shift = attribution["selected_start_ns"] - steps[76]["start_ns"]
    assert steps[115]["end_ns"] + shift == attribution["selected_end_ns"]
    selected = {
        (e["start_ns"] + shift, e["end_ns"] + shift): e
        for e in events
        if e["name"] == "disk_read_batch"
        and attribution["selected_start_ns"]
        <= e["start_ns"] + shift
        < e["end_ns"] + shift
        <= attribution["selected_end_ns"]
    }
    assert len(selected) == len(batches) == 1696
    totals = {
        (k, prefix): [0, 0, 0] for k, prefix in itertools.product((1, 2, 4, 8), (False, True))
    }
    histogram = Counter()
    for batch in batches:
        e = selected.pop((batch["start"], batch["end"]))
        data = e["data"]
        jobs, timings = data["jobs"], data["timings_job_worker_start_end_bytes_failed"]
        n = len(jobs)
        assert 1 <= n <= 10 and n == batch["jobs"] == len(timings)
        assert data["ok"] is True and data["schedule"] == "static" and data["workers"] == 4
        assert len({slot for _, slot in jobs}) == n
        for i, row in enumerate(timings):
            assert len(row) == 6 and all(type(v) is int and v >= 0 for v in row)
            assert row[0] == i and row[1] == i % min(n, 4) and row[2] <= row[3]
            assert row[4] == 1248012 and row[5] == 0
        ends = [row[3] for row in timings]
        start, end = batch["start"], batch["end"]
        frontier = end - batch["tail_after_last_independent_gpu_ns"]
        for (k, prefix), total in totals.items():
            # Widen only the native epoch bounds for the documented clock uncertainty.
            lo, _ = window(ends, k, prefix, max(0, start - 1375), end, frontier)
            _, hi = window(ends, k, prefix, start, end + 1375, frontier)
            total[0] += lo
            total[1] += hi
            total[2] += hi > 0
        histogram[n] += 1
    assert not selected and sum(n * count for n, count in histogram.items()) == 5985
    return {
        "status": "complete_availability_bounds_not_speed",
        "source_sha256": hashes,
        "outputs": 40,
        "batches": 1696,
        "jobs": 5985,
        "bytes": 5985 * 1248012,
        "job_count_histogram": dict(sorted(histogram.items())),
        "cohorts": [
            {
                "k": k,
                "selection": "job_prefix" if prefix else "first_ready",
                "lower_ms_per_output": lo / 40e6,
                "upper_ms_per_output": hi / 40e6,
                "eligible_batches_with_window": count,
            }
            for (k, prefix), (lo, hi, count) in totals.items()
        ],
        "limits": [
            "Instrumented partial AR trace, not release; unchanged historical schedule.",
            "Job ends cover complete read functions, not kernel-only NAND time.",
            "Reader epoch bounded by enclosing Rust span; clock uncertainty +/-1375ns.",
            "No CPU publication/encoding, GPU work, contention, or reconstruction cost deducted.",
            "Job prefix is not asserted to be an occurrence prefix with cache hits or duplicate routes.",
        ],
    }


if __name__ == "__main__":
    checks = self_check()
    if sys.argv[1:] == ["--self-check"]:
        print(
            json.dumps(
                {
                    "finite_discrete_cases": checks,
                    "invalid_cases": 6,
                    "status": "passed_not_formal_proof",
                }
            )
        )
    else:
        assert len(sys.argv) == 2, "pass a fresh output path or --self-check"
        report = analyze()
        report["finite_discrete_cases"] = checks
        with Path(sys.argv[1]).open("x") as output:
            output.write(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["cohorts"], indent=2))
