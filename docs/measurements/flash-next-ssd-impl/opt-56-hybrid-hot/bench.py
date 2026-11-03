"""Flash56: immutable original/candidate binaries, same 3 GiB cache and HCCH gates."""

import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
JOURNAL = Path("opti.md")
OLD = runpy.run_path(str(BASE.parent / "opt-50-larger-cache/bench.txt"))
PROFILE = runpy.run_path("scripts/profile_flash.py")


def qualify(records, mode):
    assert mode in ("baseline", "mtp")
    assert len(records) == 8
    assert [(r["prompt"], r["index"], r["label"]) for r in records] == [
        (p, i, label) for p in ("MAIN", "HOLD") for i, label in enumerate("HCCH")
    ]
    for label in "HC":
        assert len({(r["engine"], r["binary_sha256"]) for r in records if r["label"] == label}) == 1
    assert records[0]["binary_sha256"] != records[1]["binary_sha256"]
    assert all(
        r["status"] == "validated"
        and r["mode"] == mode
        and r["cache_gib"] == 3
        and type(r["gpu_hot"]) is int
        and r["gpu_hot"] == int(r["label"] == "C")
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
            assert plan[0]["env"]["MLXL3_SSD_GPU_HOT"] == str(item["gpu_hot"])
        for item in matrix:
            assert item["mode"] == mode and item["cache_gib"] == 3
            assert item["gpu_hot"] == int(item["label"] == "C")
            for path, sha in identity.items():
                assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
            record = {**item, "status": "running"}
            result["records"].append(record)
            save()
            plan = BASE / item["plan"]
            command = json.loads(plan.read_text())
            assert len(command) == 1
            assert command[0]["env"]["MLXL3_SSD_GPU_HOT"] == str(item["gpu_hot"])
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
                f"\n- **56B {mode} : {result['status']}**, gain confirmé={result['confirmed_gain']} ; {output}.\n"
            )


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        engine, model, dest, mode, prompt, golden = sys.argv[2:]
        OLD["generate"](engine, model, dest, mode, prompt, 3, golden)
    else:
        main(sys.argv[1])
