"""Export captured Flash53 data. No model runs, no exclusive GPU attribution."""

import csv
import gzip
import json
import math
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))


def clocks(events, summary, part, stats):
    roots = [e for e in events if e["parent"] is None]
    assert len(roots) == 1 and roots[0]["name"] == part
    duration = roots[0]["end_ns"] - roots[0]["start_ns"]
    wall = summary[part + "_seconds"]
    assert type(wall) in (float, int) and math.isfinite(wall) and wall > 0
    assert abs(duration / 1e9 - wall) <= max(0.02, wall * 0.01), "trace/wall clock mismatch"
    assert sum(c["host_self_ns"]["total"] for c in stats["components"]) == duration
    return duration


def output_steps(events, summary):
    stats = API["validate_events"](events)
    steps, layers, rows = [], [], {}
    for i, event in enumerate(events):
        parent = event["parent"]
        step = steps[parent] if parent is not None else None
        layer = layers[parent] if parent is not None else None
        name = event["name"]
        if name.startswith("layer_"):
            layer = int(name.removeprefix("layer_"))
        if name == "output_step":
            step = i
            data = event["data"]
            assert data == summary["steps"][len(rows)], "step metadata differs"
            rows[i] = {
                **data,
                "target_rows_sum": 0,
                "target_moe_calls": 0,
                "route_readback_ns": 0,
                "ssd_batch_wall_ns": 0,
                "ssd_jobs": 0,
                "returned_bytes": 0,
                "hits": 0,
                "misses": 0,
                "head_proposal_ns": 0,
                "verify_pair_ns": 0,
                "commit_pair_ns": 0,
                "head_cache_extend_ns": 0,
                "proposal_spans": 0,
                "verify_spans": 0,
                "commit_spans": 0,
            }
        steps.append(step)
        layers.append(layer)
        if step is None:
            continue
        row = rows[step]
        duration = event["end_ns"] - event["start_ns"]
        names = {
            "mtp_head_proposal": ("head_proposal_ns", "proposal_spans"),
            "verify_pair": ("verify_pair_ns", "verify_spans"),
            "commit_pair": ("commit_pair_ns", "commit_spans"),
            "mtp_head_cache_extend": ("head_cache_extend_ns", None),
        }
        if name in names:
            field, count = names[name]
            row[field] += duration
            if count:
                row[count] += 1
        if layer is None:
            continue
        if name == "moe_forward_rows":
            row["target_rows_sum"] += event["data"]["rows"]
            row["target_moe_calls"] += 1
            row["hits"] += event["data"]["hot_rows"]
            row["misses"] += event["data"]["cold_rows"]
        elif name == "route_readback_inclusive":
            row["route_readback_ns"] += duration
        elif name == "disk_read_batch":
            row["ssd_batch_wall_ns"] += duration
            row["ssd_jobs"] += len(event["data"]["jobs"])
            row["returned_bytes"] += sum(
                r[4] for r in event["data"]["timings_job_worker_start_end_bytes_failed"]
            )
    result = list(rows.values())
    assert len(result) == len(summary["tokens"]) - 1 == len(summary["steps"])
    offset = len(summary["prompt_ids"])
    for i, row in enumerate(result, 1):
        assert row["step"] == i and row["mtp_proposed"] in (0, 1)
        proposed, accepted = row["mtp_proposed"], row["mtp_accepted"]
        assert type(proposed) is int and type(accepted) is int
        assert accepted in (0, 1) and accepted <= proposed
        delta = row["offset"] - offset
        assert delta in (0, 1, 2)
        if proposed:
            assert summary["mtp"] and delta == 1 + accepted
            label, target_rows = ("accepted" if accepted else "rejected"), 2
        elif delta == 0:
            assert summary["mtp"]
            label, target_rows = "queued", 0
        else:
            assert delta == 1
            label, target_rows = ("fallback" if summary["mtp"] else "serial"), 1
        assert row["mtp_blocks"] == proposed
        assert row["proposal_spans"] == row["verify_spans"] == row["commit_spans"] == proposed
        assert row["target_rows_sum"] == 48 * target_rows
        assert row["target_moe_calls"] == (48 if target_rows else 0)
        assert row["hits"] + row["misses"] == 480 * target_rows
        assert row["returned_bytes"] == row["ssd_jobs"] * API["EXPERT_BYTES"]
        row.update(classification=label, offset_before=offset, target_rows=target_rows)
        offset = row["offset"]
    assert offset == len(summary["prompt_ids"]) + len(result)
    assert (
        sum(r["mtp_proposed"] for r in result) == summary["mtp_proposed"] == summary["mtp_blocks"]
    )
    assert sum(r["mtp_accepted"] for r in result) == summary["mtp_accepted"]
    return result, stats


def write_csv(path, rows):
    assert rows, "empty export"
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


def summarize(base, output):
    import numpy as np

    assert base.is_dir() and not output.exists(), "refusing missing input or overwritten export"
    output.mkdir()
    status = {"status": "incomplete", "parity": False, "cases": [], "events": 0, "disk_jobs": 0}
    phase_rows, components, layer_rows, worker_rows, step_rows, system_rows = [], [], [], [], [], []
    (output / "result.json").write_text(json.dumps(status, indent=2) + "\n")
    try:
        for directory in sorted(base.glob("*/profile")):
            result_path = directory.parent / "dataset-result.json"
            if not result_path.exists() or not json.loads(result_path.read_text()).get("parity"):
                continue
            complete = load_json(directory / "complete.json")
            kind, prompt = complete["kind"], complete["prompt"]
            mode = "MTP" if complete["mtp"] else "OFF"
            cache = json.loads((directory.parent / "run.json").read_text())[0]["env"][
                "MLXL3_SSD_CACHE_GIB"
            ]
            meta = {
                "case": directory.parent.name,
                "kind": kind,
                "mode": mode,
                "prompt": prompt,
                "cache_gib": int(cache),
            }
            golden = load_json(
                ROOT
                / f"docs/measurements/flash-next-ssd-impl/opt-39a-les-lru/confirmation/reference128-{prompt}.json"
            )
            for summary in complete["summaries"]:
                vm = API["validate_summary"](summary, kind, golden)
                phase = summary["phase"]
                times = summary["mtp_head_verify_repair_total_seconds"]
                assert len(times) == 4 and all(
                    type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in times
                )
                assert sum(times[:3]) <= times[3] + 1e-6 <= summary["decode_seconds"] + 2e-6
                decoded = len(summary["tokens"]) - 1
                counters = {
                    k: sum(
                        b[i] - a[i]
                        for a, b in zip(
                            summary["prefill_counts"], summary["after_counts"], strict=True
                        )
                    )
                    for i, k in enumerate(["decode_hits", "decode_misses", "decode_returned_bytes"])
                }
                phase_rows.append(
                    {
                        **meta,
                        "phase": phase,
                        "traced": summary["traced"],
                        "fenced": summary["fenced"],
                        "prompt_tokens": len(summary["prompt_ids"]),
                        "output_tokens": len(summary["tokens"]),
                        "decode_tokens": decoded,
                        "prefill_seconds": summary["prefill_seconds"],
                        "decode_seconds": summary["decode_seconds"],
                        "diagnostic_tps": summary["diagnostic_tps"],
                        "mtp_proposed": summary["mtp_proposed"],
                        "mtp_accepted": summary["mtp_accepted"],
                        "mtp_blocks": summary["mtp_blocks"],
                        "head_seconds": summary["mtp_head_verify_repair_total_seconds"][0],
                        "verify_seconds": summary["mtp_head_verify_repair_total_seconds"][1],
                        "repair_seconds": summary["mtp_head_verify_repair_total_seconds"][2],
                        "footprint_after_bytes": summary["after"]["memory"][
                            "process_footprint_bytes"
                        ],
                        "lifetime_peak_after_bytes": summary["after"]["memory"][
                            "process_lifetime_peak_bytes"
                        ],
                        **counters,
                        **{
                            "prefill_vm_" + k: v
                            for k, v in vm["prefill_vm_counter_pages_delta"].items()
                        },
                        **{
                            "decode_vm_" + k: v
                            for k, v in vm["decode_vm_counter_pages_delta"].items()
                        },
                    }
                )
                if not summary["traced"]:
                    continue
                for part in ("prefill", "decode"):
                    events = API["load_events"](directory / f"{phase}-{part}.json")
                    if part == "decode":
                        rows, stats = output_steps(events, summary)
                        step_rows.extend({**meta, "phase": phase, **r} for r in rows)
                    else:
                        stats = API["validate_events"](events)
                    clocks(events, summary, part, stats)
                    del events
                    status["events"] += stats["event_count"]
                    status["disk_jobs"] += len(stats["disk_jobs"])
                    for component in stats["components"]:
                        components.append(
                            {
                                **meta,
                                "phase": phase,
                                "part": part,
                                "layer": component["layer"],
                                "component": component["component"],
                                **component["wall_ns"],
                                "host_self_total_ns": component["host_self_ns"]["total"],
                            }
                        )
                    for layer in sorted(
                        {r["layer"] for r in stats["routes"]}, key=lambda n: -1 if n is None else n
                    ):
                        routes = [r for r in stats["routes"] if r["layer"] == layer]
                        batches = [r for r in stats["disk_batches"] if r["layer"] == layer]
                        jobs = [r for r in stats["disk_jobs"] if r["layer"] == layer]
                        pairs = [r["pairs"] for r in routes if r["rows"] == 2]
                        layer_rows.append(
                            {
                                **meta,
                                "phase": phase,
                                "part": part,
                                "layer": "head" if layer is None else layer,
                                "capacity": routes[0]["capacity"],
                                "cohorts": len(routes),
                                "all_ready_cohorts": sum(r["all_ready"] for r in routes),
                                "target_or_head_rows": sum(r["rows"] for r in routes),
                                "hits": sum(r["hot"] for r in routes),
                                "misses": sum(r["cold"] for r in routes),
                                "unique_experts_sum": sum(r["unique"] for r in routes),
                                "pair_cohorts": len(pairs),
                                "shared_pair_experts_sum": sum(pairs),
                                "disk_jobs": len(jobs),
                                "returned_bytes": sum(r["bytes"] for r in jobs),
                                "ssd_batch_wall_ns": sum(r["wall_ns"] for r in batches),
                                "batch_worker_finish_spread_p95_ns": API["distribution"](
                                    [r["worker_finish_spread_ns"] for r in batches]
                                )["p95"]
                                if batches
                                else None,
                            }
                        )
                        for worker in sorted({r["worker"] for r in jobs}):
                            selected = [r for r in jobs if r["worker"] == worker]
                            values = [r["duration_ns"] for r in selected]
                            dist = API["distribution"](values)
                            independent = np.percentile(
                                values, [50, 90, 95, 99], method="inverted_cdf"
                            ).tolist()
                            assert independent == [dist[k] for k in ("p50", "p90", "p95", "p99")]
                            worker_rows.append(
                                {
                                    **meta,
                                    "phase": phase,
                                    "part": part,
                                    "layer": "head" if layer is None else layer,
                                    "worker": worker,
                                    "returned_bytes": sum(r["bytes"] for r in selected),
                                    **dist,
                                }
                            )
                    assert sum(
                        r["bytes"] for r in stats["disk_jobs"] if r["layer"] is not None
                    ) == sum(
                        b[2] - a[2]
                        for a, b in zip(
                            summary["before_counts"]
                            if part == "prefill"
                            else summary["prefill_counts"],
                            summary["prefill_counts"]
                            if part == "prefill"
                            else summary["after_counts"],
                            strict=True,
                        )
                    )
            status["cases"].append(meta)
            (output / "result.json").write_text(json.dumps(status, indent=2) + "\n")
        assert len([c for c in status["cases"] if c["kind"] == "natural"]) == 4
        assert len([c for c in status["cases"] if c["kind"] == "fenced"]) == 4
        expected = {(mode, prompt, 3) for mode in ("OFF", "MTP") for prompt in ("MAIN", "HOLD")}
        for kind in ("natural", "fenced"):
            assert {
                (c["mode"], c["prompt"], c["cache_gib"])
                for c in status["cases"]
                if c["kind"] == kind
            } == expected
        for path in sorted(base.glob("*/system-samples.json")):
            samples = load_json(path)
            for sample in samples["samples"]:
                gpu, disk = sample.get("gpu") or {}, sample.get("disk") or {}
                system_rows.append(
                    {
                        "case": path.parent.name,
                        "sampler_status": samples["status"],
                        "epoch": sample["epoch"],
                        "errors": len(sample["errors"]),
                        "Device Utilization %": gpu.get("Device Utilization %"),
                        "Renderer Utilization %": gpu.get("Renderer Utilization %"),
                        "Tiler Utilization %": gpu.get("Tiler Utilization %"),
                        "disk_mb_total": disk.get("mb_total"),
                        "disk_transfers_total": disk.get("transfers_total"),
                    }
                )
        for name, rows in [
            ("phases", phase_rows),
            ("components", components),
            ("layers", layer_rows),
            ("workers", worker_rows),
            ("output-steps", step_rows),
            ("system-timeline", system_rows),
        ]:
            write_csv(output / (name + ".csv"), rows)
        acceptance = []
        for meta in status["cases"]:
            if meta["kind"] != "natural" or meta["mode"] != "MTP":
                continue
            for phase in ["trace_1", "trace_2", "trace_3"]:
                selected = [
                    r for r in step_rows if r["case"] == meta["case"] and r["phase"] == phase
                ]
                for start, end in [(1, 32), (33, 64), (65, 96), (97, 127)]:
                    segment = [r for r in selected if start <= r["step"] <= end]
                    acceptance.append(
                        {
                            **meta,
                            "phase": phase,
                            "decode_step_start": start,
                            "decode_step_end": end,
                            "proposed": sum(r["mtp_proposed"] for r in segment),
                            "accepted": sum(r["mtp_accepted"] for r in segment),
                            "queued_outputs": sum(r["classification"] == "queued" for r in segment),
                            "step_wall_seconds": sum(r["seconds"] for r in segment),
                            "verify_pair_ns": sum(r["verify_pair_ns"] for r in segment),
                        }
                    )
        write_csv(output / "acceptance-by-position.csv", acceptance)
        status.update(
            status="validated_exports_observational",
            parity=True,
            phase_rows=len(phase_rows),
            component_rows=len(components),
            layer_rows=len(layer_rows),
            worker_rows=len(worker_rows),
            output_step_rows=len(step_rows),
            system_rows=len(system_rows),
            limits="host wall/completion intervals, not exclusive GPU; global system counters; fenced33 and natural128 separated; six-GiB lot incomplete",
        )
    except BaseException as error:
        status.update(status="failed_or_interrupted", parity=False, error=repr(error))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
