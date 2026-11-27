#!/usr/bin/env python3
"""Exercise the same local inventory through Flash Next DFlash2 with and without live cache."""

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True
# Reuse the historical inventory, isolated registry and bounded bridge transport.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flashnext-agent-cache-2026-11-26"))
import flash_agent_trial as inventory  # noqa: E402


EXPERIMENT = "FLASHNEXT-AGENT-DFLASH2-LIVE-01"
DRAFT_REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"
DRAFT_FILES = {
    "config.json": (1236, "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9"),
    "model.safetensors": (
        1_206_140_008,
        "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f",
    ),
}
WIDTH = 2
MODE = 2
ENVIRONMENT = {
    "MLXL3_FLASH_DRAFT_ADAPTIVE": "0",
    "MLXL3_FLASH_DRAFT_PRECAP": "0",
    "MLXL3_FLASH_DRAFT_WIDTH": str(WIDTH),
    "MLXL3_FLASH_DRAFT_Q4": "1",
    "MLXL3_FLASH_DRAFT_RING": "1",
    "MLXL3_SSD_CACHE_GIB": "3",
}
SPEC_COUNTS = ("dflash_proposed_tokens", "dflash_accepted_tokens", "dflash_blocks")
MEMORY_COUNTS = ("process_lifetime_peak_bytes", "mlx_peak_bytes", "process_footprint_bytes")
require = inventory.require


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", nargs="+", type=Path, metavar="TRIAL_JSON")
    for name in ("engine", "registry", "helper", "dflash-draft"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--label", default="A1")
    parser.add_argument("--live-cache", action="store_true")
    parser.add_argument("--context-length", type=inventory.positive, default=4096)
    parser.add_argument("--max-tokens", type=inventory.positive, default=192)
    parser.add_argument("--ready-timeout", type=inventory.positive, default=900)
    parser.add_argument("--task-timeout", type=inventory.positive, default=1800)
    parser.add_argument("--env", action="append", type=inventory.environment_setting, default=[])
    parser.set_defaults(mtp=False, reasoning_effort="none")
    args = parser.parse_args()
    inputs = (args.engine, args.registry, args.helper, args.model, args.dflash_draft)
    if args.compare is None and not all(inputs):
        parser.error("trial needs --engine, --registry, --helper, --model and --dflash-draft")
    if args.compare is not None and (any(inputs) or args.live_cache or args.env):
        parser.error("--compare takes completed trial reports and an output path")
    if args.compare is not None:
        sources = args.compare
        outputs = [args.output]
    else:
        sources = [*inputs[:3], *(args.dflash_draft / name for name in DRAFT_FILES)]
        outputs = [args.output, args.output.with_suffix(".stderr.log")]
    sources = {path.resolve() for path in sources}
    if any(path.resolve() in sources for path in outputs):
        parser.error("--output must differ from all input paths")
    if len({path.resolve() for path in outputs}) != len(outputs):
        parser.error("--output must differ from the stderr log path")
    return args


def draft_fingerprints(directory):
    hashes = {}
    for name, (size, expected) in DRAFT_FILES.items():
        path = directory / name
        require(
            path.is_file() and path.stat().st_size == size, f"invalid pinned draft file: {name}"
        )
        hashes[name] = inventory.fingerprint(path)
        require(hashes[name] == expected, f"pinned draft checksum mismatch: {name}")
    return hashes


def run_task(bridge, request, timeout, task):
    started = time.monotonic()
    try:
        bridge.send(request)
        for event in inventory.events_until(bridge, timeout, "complete", request["request_id"]):
            kind = event["type"]
            if kind in ("generation_mode", "tool_start", "tool_result"):
                require(
                    event.get("request_id") == request["request_id"], f"wrong request for {kind}"
                )
            if kind == "generation_mode":
                require("generation_mode" not in task, "duplicate generation mode")
                task[kind] = event
            elif kind == "complete":
                task[kind] = event
            elif kind in ("tool_start", "tool_result"):
                task["tools"].append(event)
    finally:
        task["wall_seconds"] = time.monotonic() - started


def valid_memory(footprint: int, peak: int, previous_peak: int) -> bool:
    """Whether one process footprint sample can follow the previous lifetime maximum.

    post: __return__ == (footprint > 0 and peak >= max(footprint, previous_peak))
    """
    return 0 < footprint <= peak and peak >= previous_peak


def validate_task(task, request):
    require(
        inventory.duration(task.get("wall_seconds")) and task["wall_seconds"] > 0,
        "invalid wall time",
    )
    expected = {
        "request_id": request["request_id"],
        "dflash_requested": True,
        "dflash_active": True,
        "mtp_requested": False,
        "mtp_active": False,
        "dflash_mode": MODE,
        "dflash_proposals": WIDTH,
        "dflash_draft_path": request["dflash_draft_path"],
    }
    mode = task.get("generation_mode", {})
    require(
        all(
            type(mode.get(key)) is type(value) and mode[key] == value
            for key, value in expected.items()
        ),
        f"unexpected generation mode: {mode}",
    )
    live_expected = request["live_prompt_cache"]
    live_steps = inventory.check_task(task, require_live=live_expected)
    require(
        live_steps == (len(inventory.FILES) if live_expected else 0),
        "unexpected number of resumed rounds",
    )
    complete = task["complete"]
    require(
        isinstance(complete.get("token_hash"), str)
        and re.fullmatch(r"[0-9a-f]{16}", complete["token_hash"]),
        "missing final-round token hash",
    )
    history = json.loads(complete["turn_context"])
    require(
        isinstance(history, list)
        and len(history) == 1 + len(task["tools"])
        and all(isinstance(entry, dict) for entry in history),
        "incomplete turn history",
    )
    summary, rounds = complete["stats"], complete["round_stats"]
    require(inventory.duration(summary.get("draft_load_seconds")), "invalid draft load time")
    previous_peak = 0
    process_identity = None
    for index, data in enumerate(rounds, 1):
        require(
            all(inventory.count(data.get(field)) for field in SPEC_COUNTS),
            f"invalid DFlash2 counts in round {index}",
        )
        proposed, accepted, blocks = (data[field] for field in SPEC_COUNTS)
        require(
            accepted <= proposed <= WIDTH * blocks, f"invalid DFlash2 acceptance in round {index}"
        )
        require(bool(blocks) == bool(proposed), f"invalid DFlash2 blocks in round {index}")
        memory = data.get("memory")
        require(
            isinstance(memory, dict)
            and all(inventory.count(memory.get(field)) for field in MEMORY_COUNTS),
            f"missing process/MLX memory in round {index}",
        )
        peak, mlx_peak, footprint = (memory[field] for field in MEMORY_COUNTS)
        require(
            valid_memory(footprint, peak, previous_peak) and mlx_peak > 0,
            f"invalid memory sample in round {index}",
        )
        identity = (memory.get("process_id"), memory.get("process_start_abstime"))
        if process_identity is None:
            process_identity = identity
        require(identity == process_identity, "memory samples span multiple bridge processes")
        previous_peak = peak
    require(
        rounds[-1]["memory"] == summary.get("memory"),
        "aggregate memory differs from the final round",
    )
    for field in SPEC_COUNTS:
        require(
            inventory.count(summary.get(field))
            and summary[field] > 0
            and summary[field] == sum(r[field] for r in rounds),
            f"missing DFlash2 activity or inconsistent sum: {field}",
        )
    return live_steps


def collect(args, report):
    request = inventory.request_for(args)
    request.update(
        dflash2=True, dflash_draft_path=str(args.dflash_draft.resolve()), dflash_mode=MODE
    )
    environment = {**dict(args.env), **ENVIRONMENT}
    report.update(
        label=args.label,
        model=args.model,
        fixture=inventory.FILES,
        engine=str(args.engine.resolve()),
        helper=str(args.helper.resolve()),
        context_length=args.context_length,
        request=request,
        draft_revision=DRAFT_REVISION,
        environment={
            **{
                key: value
                for key, value in os.environ.items()
                if key.startswith(("MLXL3_", "MLX_")) or key == "DYLD_LIBRARY_PATH"
            },
            **environment,
        },
        task={"tools": []},
    )
    report["draft_sha256"] = draft_fingerprints(args.dflash_draft)
    report["engine_sha256"] = inventory.fingerprint(args.engine)
    report["helper_sha256"] = inventory.fingerprint(args.helper)
    stderr_path = args.output.with_suffix(".stderr.log")
    with tempfile.TemporaryDirectory(prefix="mlxl3-flash-dflash2-agent-") as temporary:
        registry, report["model_registry_entry_sha256"] = inventory.configure(Path(temporary), args)
        bridge = inventory.Bridge(
            report["engine"],
            registry,
            args.model,
            stderr_path,
            environment,
            str(args.context_length),
        )
        try:
            ready = report["ready"] = inventory.wait_for(bridge, args.ready_timeout, "ready")
            require(
                ready.get("agent_loop_supported") is True
                and ready.get("dflash_supported") is True
                and isinstance(ready.get("flash"), dict)
                and ready.get("context_limit") == args.context_length,
                "bridge did not advertise the Flash DFlash2 Agent and requested context",
            )
            bridge.send({"type": "set_mcp", "enabled": True, "request_id": "mcp"})
            mcp = report["mcp"] = inventory.wait_for(bridge, 90, "mcp_status", "mcp")
            catalog = mcp.get("mcp_tool_catalog")
            require(
                isinstance(catalog, list)
                and any(item.get("function", {}).get("name") == inventory.TOOL for item in catalog)
                and not mcp.get("mcp_errors"),
                f"{inventory.TOOL} did not load: {mcp.get('mcp_errors')}",
            )
            run_task(bridge, request, args.task_timeout, report["task"])
            report["live_steps"] = validate_task(report["task"], request)
        finally:
            try:
                report["engine_exit"] = bridge.close()
            finally:
                bridge.selector.close()
                bridge.process.stdin.close()
                bridge.process.stdout.close()
    require(report["engine_exit"] == 0, f"bridge exited with {report['engine_exit']}")


def stable_result(task):
    complete = task["complete"]
    return (
        [(event["type"], event.get("arguments"), event.get("text")) for event in task["tools"]],
        json.loads(complete["turn_context"]),
        complete["assistant_context"],
        complete["cache_context"],
        complete["token_hash"],
        [step["generated_tokens"] for step in complete["round_stats"]],
    )


def compare_trials(paths, comparison):
    require(len(set(path.resolve() for path in paths)) == len(paths), "duplicate trial input paths")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    require(
        all(
            item.get("experiment") == EXPERIMENT and item.get("status") == "complete"
            for item in records
        ),
        "every report must be a completed trial from this experiment",
    )
    arms = [item["request"]["live_prompt_cache"] for item in records]
    require(
        arms in ([False, True], [False, True, True, False]), "expected trials in AB or ABBA order"
    )
    require(len({item["label"] for item in records}) == len(records), "duplicate trial labels")
    baseline = records[0]
    invariant_fields = (
        "model",
        "context_length",
        "fixture",
        "engine_sha256",
        "helper_sha256",
        "model_registry_entry_sha256",
        "draft_revision",
        "draft_sha256",
        "environment",
    )
    invariant_ready = ("mlx_version", "runtime_commit", "runtime_profile", "dflash_tuning_key")
    rows = []
    for entry in records:
        label = entry["label"]
        require(
            all(entry[key] == baseline[key] for key in invariant_fields),
            f"profile differs in {label}",
        )
        require(
            all(entry["ready"][key] == baseline["ready"][key] for key in invariant_ready),
            f"runtime differs in {label}",
        )
        require(
            entry["request"]
            == {**baseline["request"], "live_prompt_cache": entry["request"]["live_prompt_cache"]},
            f"request differs in {label}",
        )
        validate_task(entry["task"], entry["request"])
        require(
            stable_result(entry["task"]) == stable_result(baseline["task"]),
            f"task output differs in {label}",
        )
        stats = entry["task"]["complete"]["stats"]
        rows.append(
            {
                "label": label,
                "live_cache": entry["request"]["live_prompt_cache"],
                "wall_seconds": entry["task"]["wall_seconds"],
                **{
                    key: stats[key]
                    for key in (
                        "prefill_seconds",
                        "decode_seconds",
                        "draft_load_seconds",
                        "evaluated_prompt_tokens",
                        *SPEC_COUNTS,
                    )
                },
                "live_prompt_tokens": stats.get("live_prompt_tokens", 0),
                **{key: stats["memory"][key] for key in MEMORY_COUNTS},
            }
        )
    grouped = [[row for row in rows if row["live_cache"] is flag] for flag in (False, True)]
    metrics = [key for key in rows[0] if key not in ("label", "live_cache")]
    medians = [
        {key: statistics.median(row[key] for row in group) for key in metrics} for group in grouped
    ]
    process_peaks = [max(row["process_lifetime_peak_bytes"] for row in group) for group in grouped]
    require(process_peaks[0] > 0 and process_peaks[1] > 0, "missing lifetime peak")
    comparison.update(
        parity=True,
        memory_gate_passed=process_peaks[1] <= process_peaks[0],
        process_peaks_bytes={"A": process_peaks[0], "B": process_peaks[1]},
        medians={"A": medians[0], "B": medians[1]},
        trials=rows,
    )


def main():
    args = options()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"experiment": EXPERIMENT, "status": "failed"}
    try:
        if args.compare is None:
            collect(args, report)
        else:
            report.update(parity=False, inputs=[str(path.resolve()) for path in args.compare])
            compare_trials(args.compare, report)
    except (Exception, KeyboardInterrupt) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if args.compare is None and "engine_exit" in report:
            if detail := inventory.stderr_tail(args.output.with_suffix(".stderr.log")):
                report["stderr_tail"] = detail
    else:
        report["status"] = (
            "complete" if report.get("memory_gate_passed", True) else "memory_regression"
        )
    inventory.write_report(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "label", "live_steps", "parity", "error")
                if key in report
            }
        )
    )
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
