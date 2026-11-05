"""Flash57: same immutable binary, compact dispatch OFF/ON, same 3 GiB cache and HCCH gates."""

import hashlib
import json
import math
import runpy
import statistics
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
JOURNAL = Path("opti.md")
OLD = runpy.run_path(str(BASE.parent / "opt-50-larger-cache/bench.txt"))
PROFILE = runpy.run_path("scripts/profile_flash.py")


def qualify_micro(data):
    assert data["status"] == "complete" and data["parity"] == "byte_exact_independent_slots"
    assert type(data["aa"]) is bool and data["forwards_per_sample"] == 256
    assert len(data["records"]) == 18
    seen, cells = set(), []
    for row in data["records"]:
        key = row["layer"], row["rows"], row["mask"]
        assert all(type(n) is int for n in key)
        assert key in {(l, r, m) for l in [0, 23, 47] for r in [1, 2] for m in range(3)}
        assert key not in seen
        seen.add(key)
        ids = row["route_ids"]
        assert len(ids) == key[1] * 10 and all(type(i) is int and 0 <= i < 512 for i in ids)
        assert all(len(set(ids[i : i + 10])) == 10 for i in range(0, len(ids), 10))
        ranks = {id_: rank for rank, id_ in enumerate(sorted(set(ids)))}
        hot = sum(key[2] == 2 or (key[2] == 1 and ranks[i] % 2 == 0) for i in ids)
        assert type(row["hot_routes"]) is int and row["hot_routes"] == hot
        assert len(row["samples"]) == 8
        samples = {}
        for sample in row["samples"]:
            pair, candidate, seconds = (sample[k] for k in ["pair", "candidate", "seconds"])
            assert type(pair) is int and pair in range(4) and type(candidate) is bool
            assert (pair, candidate) not in samples
            assert type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0
            samples[pair, candidate] = seconds
        arms = [[samples[p, c] for p in range(4)] for c in (False, True)]
        cells.append(
            {
                "layer": key[0],
                "rows": key[1],
                "mask": key[2],
                "hot_routes": hot,
                "H_us": statistics.median(arms[0]) * 1e6,
                "C_us": statistics.median(arms[1]) * 1e6,
                "gain_percent": 100 * (statistics.median(arms[0]) / statistics.median(arms[1]) - 1),
                "drift_percent": [100 * (max(a) / min(a) - 1) for a in arms],
            }
        )
    return {
        "status": "micro_diagnostic_only",
        "throughput_gain_confirmed": False,
        "aa": data["aa"],
        "cells": cells,
    }


def qualify(records, mode):
    assert mode in ("baseline", "mtp")
    assert len(records) == 8
    assert [(r["prompt"], r["index"], r["label"]) for r in records] == [
        (p, i, label) for p in ("MAIN", "HOLD") for i, label in enumerate("HCCH")
    ]
    for label in "HC":
        assert len({(r["engine"], r["binary_sha256"]) for r in records if r["label"] == label}) == 1
    assert records[0]["binary_sha256"] == records[1]["binary_sha256"]
    assert all(
        r["status"] == "validated"
        and r["mode"] == mode
        and r["cache_gib"] == 3
        and type(r["gpu_compact"]) is int
        and r["gpu_compact"] == int(r["label"] == "C")
        and r["swapouts_bytes"] == 0
        and type(r["tps"]) in (int, float)
        and math.isfinite(r["tps"])
        and r["tps"] > 0
        and type(r["peak_bytes"]) is int
        and 0 < r["peak_bytes"] <= OLD["MAX_FOOTPRINT"]
        for r in records
    )
    pairs, drifts = [], {}
    for prompt, group in zip(("MAIN", "HOLD"), (records[:4], records[4:]), strict=True):
        assert all(
            (r["token_hash"], r["text"], r["mtp_counts"])
            == (group[0]["token_hash"], group[0]["text"], group[0]["mtp_counts"])
            for r in group
        ), "paired output or MTP count divergence"
        gains = []
        for a, b in (group[:2], group[2:]):
            h, c = (a, b) if a["label"] == "H" else (b, a)
            gain = 100 * (c["tps"] / h["tps"] - 1)
            gains.append(gain)
            pairs.append(
                {"prompt": prompt, "H_tps": h["tps"], "C_tps": c["tps"], "gain_percent": gain}
            )
        controls = [group[0]["tps"], group[3]["tps"]]
        drifts[prompt] = {
            "control_percent": 100 * (max(controls) / min(controls) - 1),
            "gain_spread_points": max(gains) - min(gains),
        }
    stable = all(
        v["control_percent"] <= 5 and v["gain_spread_points"] <= 3 for v in drifts.values()
    )
    gain = stable and all(p["gain_percent"] >= 1 for p in pairs)
    return {
        "status": "validated_gain" if gain else "rejected_gain" if stable else "inconclusive_drift",
        "confirmed_gain": gain,
        "pairs": pairs,
        "drifts": drifts,
    }


def main(mode):
    assert mode in ("baseline", "mtp")
    output = BASE / f"{mode}-result.json"
    assert not output.exists(), "refusing to overwrite a campaign"
    result = {"status": "incomplete", "confirmed_gain": False, "records": []}

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        quality = json.loads((BASE / "quality-summary.json").read_text())
        assert quality["status"] == "validated_quality_not_speed" and quality["cache_gib"] == 3
        assert quality["gpu_hot"] == 1 and quality["gpu_compact"] == 1
        matrix = json.loads((BASE / f"{mode}-matrix.json").read_text())
        assert len(matrix) == 8
        assert [(i["prompt"], i["index"], i["label"]) for i in matrix] == [
            (p, n, label) for p in ("MAIN", "HOLD") for n, label in enumerate("HCCH")
        ]
        identity = json.loads((BASE / "speed-identity.json").read_text())
        assert identity
        baseline = json.loads((BASE / "campaign-baseline.json").read_text())["original_vm_baseline"]
        for item in matrix:
            report = Path(item["report"])
            plan = json.loads((BASE / item["plan"]).read_text())
            assert report.name == "report.json" and len(plan) == 1
            assert plan[0]["command"][1] == str(Path(__file__).resolve())
            assert plan[0]["command"][2:] == [
                "generate",
                item["engine"],
                item["model"],
                str(report.parent),
                mode,
                item["prompt"],
                item["golden"],
            ], "generator command and expected report disagree"
            assert plan[0]["env"]["MLXL3_SSD_GPU_COMPACT"] == str(item["gpu_compact"])
            assert plan[0]["env"]["MLXL3_SSD_GPU_HOT"] == "1"
        for item in matrix:
            assert item["mode"] == mode and item["cache_gib"] == 3
            assert item["gpu_compact"] == int(item["label"] == "C")
            for path, sha in identity.items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
            record = {**item, "status": "running"}
            result["records"].append(record)
            save()
            plan = BASE / item["plan"]
            command = json.loads(plan.read_text())
            assert len(command) == 1
            assert command[0]["env"]["MLXL3_SSD_GPU_COMPACT"] == str(item["gpu_compact"])
            PROFILE["run_sampler"](plan, baseline)
            process = json.loads(plan.with_name(plan.stem + "-result.json").read_text())[
                "commands"
            ][0]
            record["process"] = process
            assert (
                process["status"] == "completed_expected_outcome"
                and process["reaped"]
                and process["swapouts_bytes"] == 0
            )
            report = BASE / item["report"]
            record.update(
                OLD["check_report"](
                    json.loads(report.read_text()),
                    item,
                    report.with_name("stderr.log").read_text(),
                    json.loads(Path(item["golden"]).read_text()),
                )
            )
            record["swapouts_bytes"] = process["swapouts_bytes"]
            save()
            print(
                {k: record[k] for k in ("mode", "prompt", "label", "index", "tps", "peak_bytes")},
                flush=True,
            )
        result.update(qualify(result["records"], mode))
    except BaseException as error:
        result.update(status="failed_or_interrupted", confirmed_gain=False, error=repr(error))
        if result["records"] and result["records"][-1]["status"] == "running":
            record = result["records"][-1]
            record.update(status="failed_or_interrupted", error=repr(error))
            plan = BASE / record["plan"]
            evidence = plan.with_name(plan.stem + "-result.json")
            if evidence.exists():
                record["process"] = json.loads(evidence.read_text())["commands"][0]
        raise
    finally:
        save()
        with JOURNAL.open("a") as stream:
            stream.write(
                f"\n- **57P {mode} : {result['status']}**, gain confirmé={result['confirmed_gain']} ; {output}.\n"
            )


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        engine, model, dest, mode, prompt, golden = sys.argv[2:]
        OLD["generate"](engine, model, dest, mode, prompt, 3, golden)
    else:
        main(sys.argv[1])
