"""Inspect the preregistered full reports; warmups are never timed samples."""

import json
import math
import statistics
import sys
from pathlib import Path

LABELS = {"MAIN": 87, "HOLD": 82, "LONG": 246}
METRICS = ("ttft_seconds", "prefill_seconds", "elapsed_seconds", "decode_tps")


def number(value):
    assert type(value) in (float, int) and math.isfinite(value) and value > 0
    return value


def spread(values):
    assert values
    values = [number(v) for v in values]
    m = statistics.median(values)
    return {
        "median": m,
        "min": min(values),
        "max": max(values),
        "range_pct": 100 * (max(values) - min(values)) / m,
    }


def latency_gain(before, after):
    return 100 * (1 - number(after) / number(before))


def inspect(report):
    assert report["status"] == "validated" and report["reaped"] and report["reader_joined"]
    assert report["tokens"] == 128 and report["repeats"] == 4
    assert len(report["warmup"]) == 3 and len(report["runs"]) == 12
    measured = [x for x in report["runs"] if x["repeat"] in (2, 3, 4)]
    warmups = report["warmup"] + [x for x in report["runs"] if x["repeat"] == 1]
    assert len(measured) == 9 and len(warmups) == 6
    result = {"prompts": {}, "binary_sha256": report["binary_sha256"]}
    for label, tokens in LABELS.items():
        rows = [x for x in measured if x["prompt"] == label]
        warm = [x for x in warmups if x["prompt"] == label]
        assert len(rows) == 3 and len(warm) == 2
        sigs = []
        for row in warm + rows:
            s = row["stats"]
            assert s["generated_tokens"] == 128 and s["decode_tokens"] == 127
            assert (
                s["prompt_tokens"] == tokens
                and s["cached_prompt_tokens"] == 0
                and s["evaluated_prompt_tokens"] == tokens
            )
            assert row["text"] and row["token_hash"]
            for key in METRICS:
                number(s[key])
            assert (
                0 <= s["dflash_accepted_tokens"] <= s["dflash_proposed_tokens"]
                and s["dflash_blocks"] > 0
            )
            sigs.append(
                (
                    row["token_hash"],
                    row["text"],
                    s["dflash_proposed_tokens"],
                    s["dflash_accepted_tokens"],
                    s["dflash_blocks"],
                )
            )
            assert 0 < s["memory"]["mlx_peak_bytes"] <= 12 << 30
            assert 0 < s["memory"]["process_lifetime_peak_bytes"] <= 12 << 30
        assert all(s == sigs[0] for s in sigs)
        metrics = {key: spread([x["stats"][key] for x in rows]) for key in METRICS}
        result["prompts"][label] = {
            "metrics": metrics,
            "signature": sigs[0],
            "compression_median_bytes": statistics.median(
                [x["stats"]["memory"]["process_compressed_logical_bytes"] for x in rows]
            ),
        }
    result["stable"] = all(
        v["range_pct"] <= 3 for p in result["prompts"].values() for v in p["metrics"].values()
    )
    return result


def compare(h, c):
    assert h["binary_sha256"] == c["binary_sha256"]
    result = {"status": "screen", "stable": h["stable"] and c["stable"], "prompts": {}}
    for label in LABELS:
        a, b = h["prompts"][label], c["prompts"][label]
        assert a["signature"] == b["signature"], label
        gain = {
            key: latency_gain(a["metrics"][key]["median"], b["metrics"][key]["median"])
            for key in METRICS
            if key != "decode_tps"
        }
        gain["decode_tps"] = 100 * (
            b["metrics"]["decode_tps"]["median"] / a["metrics"]["decode_tps"]["median"] - 1
        )
        extra = b["compression_median_bytes"] - a["compression_median_bytes"]
        result["prompts"][label] = {
            "gain_pct": gain,
            "extra_compression_bytes": extra,
            "H": a["metrics"],
            "C": b["metrics"],
        }
    result["favorable"] = result["stable"] and all(
        p["gain_pct"]["ttft_seconds"] >= 1
        and p["gain_pct"]["elapsed_seconds"] >= 1
        and p["gain_pct"]["decode_tps"] >= -1
        and p["extra_compression_bytes"] <= 256 << 20
        for p in result["prompts"].values()
    )
    return result


if __name__ == "__main__":
    h = inspect(json.loads(Path(sys.argv[1]).read_text()))
    result = (
        h if len(sys.argv) == 2 else compare(h, inspect(json.loads(Path(sys.argv[2]).read_text())))
    )
    print(json.dumps(result, indent=2))
