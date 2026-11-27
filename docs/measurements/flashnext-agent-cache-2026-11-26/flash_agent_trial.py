#!/usr/bin/env python3
"""Measure a complete local inventory task through the native Studio bridge and tool helper."""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path

# Reuse the bridge transport whose unbuffered selector was exercised in the Inkling trials.
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent-live-cache-2026-11-26"))
from bridge import Bridge, BridgeError

TOOL = "sandbox.read"
FILES = {"stock.txt": "units=17", "returns.txt": "units=5", "dispatch.txt": "units=4"}
SYSTEM = "You are an inventory assistant working with local records. Use relative file paths and report the result."
PROMPT = (
    "Calculate the units currently in stock. Read stock.txt, returns.txt, then dispatch.txt via "
    "sandbox.read with one file per call in this order. Add the returned units to stock and subtract "
    "the dispatched units. When all three records have been read, answer only TOTAL=<number>."
)
ANSWER = "TOTAL=18"
TOKEN_FIELDS = (
    "prompt_tokens",
    "cached_prompt_tokens",
    "evaluated_prompt_tokens",
    "generated_tokens",
    "decode_tokens",
)
TIME_FIELDS = ("prepare_seconds", "prefill_seconds", "decode_seconds")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def environment_setting(value):
    name, separator, setting = value.partition("=")
    if not separator or not re.fullmatch(r"MLXL3_[A-Z0-9_]+", name):
        raise argparse.ArgumentTypeError("expected MLXL3_NAME=VALUE")
    return name, setting


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "registry", "helper", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default="A")
    parser.add_argument("--context-length", type=positive, default=4096)
    parser.add_argument("--max-tokens", type=positive, default=192)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--ready-timeout", type=positive, default=900)
    parser.add_argument("--task-timeout", type=positive, default=1800)
    parser.add_argument("--mtp", action="store_true")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--live-cache", action="store_true")
    cache.add_argument("--advertised-live-cache", action="store_true")
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--env", action="append", type=environment_setting, default=[])
    args = parser.parse_args()
    if args.require_live and not (args.live_cache or args.advertised_live_cache):
        parser.error("--require-live needs --live-cache or --advertised-live-cache")
    return args


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure(work, args):
    project_root = work / "files"
    project_root.mkdir()
    for name, content in FILES.items():
        (project_root / name).write_text(f"{content}\n", encoding="utf-8")
    project = work / "project.json"
    project.write_text(
        json.dumps(
            {
                "permissions": {
                    "read": True,
                    "write": False,
                    "run": False,
                    "commandSeconds": 20,
                    "toolHarness": "native",
                    "internet": False,
                    "internetHosts": [],
                }
            }
        ),
        encoding="utf-8",
    )

    registry = work / "models.json"
    entry = json.loads(args.registry.read_text(encoding="utf-8"))["models"][args.model]
    encoded_entry = json.dumps(entry, sort_keys=True)
    registry.write_text(json.dumps({"version": 1, "models": {args.model: entry}}), encoding="utf-8")
    (work / "mcp.json").write_text(
        json.dumps(
            {
                "version": 1,
                "mcpServers": {
                    "exa": {"enabled": False, "url": "https://mcp.exa.ai/mcp"},
                    "sandbox": {
                        "enabled": True,
                        "command": str(args.helper.resolve()),
                        "args": ["--agent-tools", str(project_root), str(project), "long-calls"],
                        "callTimeoutSeconds": 43,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return registry, hashlib.sha256(encoded_entry.encode()).hexdigest()


def request_for(args):
    request = {
        "type": "generate",
        "request_id": "inventory",
        "conversation_id": "flashnext-agent-cache-trial",
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "mtp": args.mtp,
        "dflash2": False,
        "dspark": False,
        "memory_saver": False,
        "reuse_prompt_cache": True,
        "live_prompt_cache": args.live_cache,
        "tool_rounds": 5,
        "mcp_enabled": True,
        "allowed_tools": [TOOL],
    }
    if args.reasoning_effort is not None:
        request["reasoning_effort"] = args.reasoning_effort
    return request


def events_until(bridge, seconds, target, request_id=None):
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BridgeError(f"no {target} event within {seconds} s")
        event = bridge.event(remaining)
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise BridgeError(f"invalid bridge event: {event!r}")
        if event["type"] in ("error", "cancelled"):
            raise BridgeError(f"{event['type']}: {event.get('message', target)}")
        if event.get("request_id") not in (None, request_id):
            raise BridgeError(f"unexpected request ID: {event.get('request_id')!r}")
        if event["type"] == target and event.get("request_id") != request_id:
            raise BridgeError(f"{target} event did not match the request")
        yield event
        if event["type"] == target:
            return


def wait_for(bridge, seconds, target, request_id=None):
    for event in events_until(bridge, seconds, target, request_id):
        if event["type"] == target:
            return event
    raise BridgeError(f"missing {target} event")


def run_task(bridge, request, seconds, report):
    task = report["task"]
    started = time.monotonic()
    try:
        bridge.send(request)
        for event in events_until(bridge, seconds, "complete", request["request_id"]):
            if event["type"] in ("tool_start", "tool_result"):
                if event.get("request_id") != request["request_id"]:
                    raise BridgeError(f"{event['type']} event did not match the request")
                task["tools"].append(event)
            elif event["type"] == "complete":
                task["complete"] = event
    finally:
        task["wall_seconds"] = time.monotonic() - started


def require(condition, message):
    if not condition:
        raise BridgeError(message)


def count(value):
    return type(value) is int and value >= 0


def duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def check_task(task, require_live=False):
    tools = task["tools"]
    complete = task["complete"]
    require(len(tools) == 2 * len(FILES), "expected exactly three tool calls and results")
    call_ids = set()
    for index, (name, content) in enumerate(FILES.items()):
        start, result = tools[2 * index : 2 * index + 2]
        call_id = start.get("tool_call_id")
        require(
            isinstance(call_id, str) and bool(call_id) and call_id not in call_ids,
            f"invalid call ID for {name}",
        )
        call_ids.add(call_id)
        arguments = start.get("arguments")
        require(
            start["type"] == "tool_start"
            and start.get("tool_name") == TOOL
            and isinstance(arguments, dict)
            and arguments.get("file_path") == name,
            f"unexpected call for {name}",
        )
        require(
            result["type"] == "tool_result"
            and result.get("tool_name") == TOOL
            and result.get("tool_call_id") == call_id
            and result.get("is_error") is False
            and isinstance(result.get("text"), str)
            and result["text"].strip() == f"1| {content}",
            f"unexpected tool result for {name}",
        )
    answer = complete.get("assistant_context")
    require(
        isinstance(answer, str) and answer.strip() == ANSWER, f"unexpected final answer: {answer!r}"
    )
    require(
        complete.get("stop_reason") == "eos" and complete.get("context_full") is False,
        "task did not end at the model's stop token",
    )
    require(complete.get("tool_rounds") == len(FILES), "expected one tool call per model round")
    steps = complete.get("round_stats")
    summary = complete.get("stats")
    require(
        isinstance(steps, list)
        and len(steps) == len(FILES) + 1
        and all(isinstance(step, dict) for step in steps)
        and isinstance(summary, dict),
        "missing per-round or aggregate statistics",
    )
    live_steps = 0
    for index, step in enumerate(steps, 1):
        require(
            all(count(step.get(field)) for field in TOKEN_FIELDS),
            f"missing or invalid token counts in round {index}",
        )
        require(
            step["prompt_tokens"] > 0
            and step["generated_tokens"] > 0
            and step["prompt_tokens"]
            == step["cached_prompt_tokens"] + step["evaluated_prompt_tokens"],
            f"inconsistent token counts in round {index}",
        )
        require(
            all(duration(step.get(field)) for field in TIME_FIELDS),
            f"invalid timings in round {index}",
        )
        live = step.get("live_prompt_tokens", 0)
        require(
            count(live) and live <= step["cached_prompt_tokens"],
            f"invalid live cache count in round {index}",
        )
        live_steps += int(live > 0)
    for field in TOKEN_FIELDS + ("live_prompt_tokens",):
        expected = sum(step.get(field, 0) for step in steps)
        actual = summary.get(field, 0) if field == "live_prompt_tokens" else summary.get(field)
        require(count(actual) and actual == expected, f"inconsistent total for {field}")
    for field in TIME_FIELDS:
        expected = sum(step[field] for step in steps)
        actual = summary.get(field)
        require(
            duration(actual) and math.isclose(actual, expected, abs_tol=1e-8),
            f"inconsistent total for {field}",
        )
    require(
        summary.get("round_count") == len(steps) and summary.get("tool_rounds") == len(FILES),
        "aggregate round counts do not match the task",
    )
    require(
        all(duration(summary.get(field)) for field in ("elapsed_seconds", "tool_seconds")),
        "invalid elapsed/tool duration",
    )
    if require_live:
        require(
            live_steps > 0 and steps[0].get("live_prompt_tokens", 0) == 0,
            "candidate resumed no live tokens between tool rounds",
        )
    return live_steps


def json_safe(record):
    """Retain malformed numeric fields as strings in a failed report."""
    if isinstance(record, dict):
        return {key: json_safe(value) for key, value in record.items()}
    if isinstance(record, list):
        return [json_safe(value) for value in record]
    if isinstance(record, float) and not math.isfinite(record):
        return repr(record)
    return record


def stderr_tail(path):
    if not path.is_file():
        return None
    with path.open("rb") as source:
        source.seek(max(0, path.stat().st_size - 2048))
        raw = source.read()
    return raw.decode("utf-8", errors="replace").strip() or None


def write_report(path, report):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as output:
        temp_path = Path(output.name)
        try:
            json.dump(json_safe(report), output, indent=2, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def main():
    args = options()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(args.env)
    request = request_for(args)
    stderr_path = args.output.with_suffix(".stderr.log")
    report = {
        "experiment": "FLASHNEXT-2026-11-26-AGENT-CACHE",
        "label": args.label,
        "model": args.model,
        "registry": str(args.registry.resolve()),
        "engine": str(args.engine.resolve()),
        "helper": str(args.helper.resolve()),
        "fixture": FILES,
        "context_length": args.context_length,
        "request": request,
        "environment": {
            **{
                key: value
                for key, value in os.environ.items()
                if key.startswith(("MLXL3_", "MLX_")) or key == "DYLD_LIBRARY_PATH"
            },
            **environment,
        },
        "status": "failed",
        "task_passed": False,
        "task": {"tools": []},
    }
    try:
        report["engine_sha256"] = fingerprint(args.engine)
        report["helper_sha256"] = fingerprint(args.helper)
        with tempfile.TemporaryDirectory(prefix="mlxl3-flash-agent-") as temporary:
            registry, report["model_registry_entry_sha256"] = configure(Path(temporary), args)
            bridge = Bridge(
                report["engine"],
                registry,
                args.model,
                stderr_path,
                environment,
                context=str(args.context_length),
            )
            try:
                report["ready"] = wait_for(bridge, args.ready_timeout, "ready")
                if args.advertised_live_cache:
                    require(
                        report["ready"].get("live_prompt_cache_supported") is True,
                        "the bridge did not advertise the live prompt cache",
                    )
                    request["live_prompt_cache"] = True
                bridge.send({"type": "set_mcp", "enabled": True, "request_id": "mcp"})
                mcp = wait_for(bridge, 90, "mcp_status", "mcp")
                report["mcp"] = mcp
                catalog = mcp.get("mcp_tool_catalog")
                require(
                    isinstance(catalog, list)
                    and any(item.get("function", {}).get("name") == TOOL for item in catalog)
                    and not mcp.get("mcp_errors"),
                    f"{TOOL} failed to load: {mcp.get('mcp_errors')}",
                )
                run_task(bridge, request, args.task_timeout, report)
                report["live_steps"] = check_task(report["task"], args.require_live)
            finally:
                try:
                    report["engine_exit"] = bridge.close()
                finally:
                    bridge.selector.close()
                    bridge.process.stdin.close()
                    bridge.process.stdout.close()
            require(report["engine_exit"] == 0, f"bridge exited with {report['engine_exit']}")
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - save partial evidence on any failed trial
        report["error"] = f"{type(error).__name__}: {error}"
        if detail := stderr_tail(stderr_path):
            report["stderr_tail"] = detail
    else:
        report["task_passed"] = True
        report["status"] = "complete"
    write_report(args.output, report)
    task = report["task"]
    stats = task.get("complete", {}).get("stats", {})
    if not isinstance(stats, dict):
        stats = {}
    console = {key: report[key] for key in ("label", "status", "task_passed")}
    console.update(
        wall_seconds=task.get("wall_seconds"),
        live_steps=report.get("live_steps"),
        cache_tokens=stats.get("cached_prompt_tokens"),
        read_tokens=stats.get("evaluated_prompt_tokens"),
        prefill_seconds=stats.get("prefill_seconds"),
        error=report.get("error"),
        stderr_tail=report.get("stderr_tail"),
    )
    print(json.dumps(console))
    return 0 if report["task_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
