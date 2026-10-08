"""Compare idle footprint and exact bridge outputs without reading user history."""

import argparse
import ctypes
import hashlib
import json
import math
import signal
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_json_process import JsonProcess


def footprint(pid):
    """Darwin rusage_info_v4: UUID[16], then ri_phys_footprint at u64 index 7."""
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    library.proc_pid_rusage.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(4096)
    if library.proc_pid_rusage(pid, 4, buffer) != 0:
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    value = int.from_bytes(buffer.raw[72:80], sys.byteorder)
    if value <= 0:
        raise ValueError("empty footprint")
    return value


def exchange(child, request, *, cancel=False, timeout=180):
    deadline = time.monotonic() + timeout
    child.send(request, deadline)
    cancelled = False
    while True:
        event = child.receive(deadline)
        if event.get("request_id") != request["request_id"]:
            raise ValueError("unexpected request identity")
        if cancel and not cancelled and event.get("type") == "delta":
            child.process.send_signal(signal.SIGUSR1)
            cancelled = True
        if event.get("type") in ("complete", "error", "cancelled", "pong"):
            if cancel and (not cancelled or event["type"] != "cancelled"):
                raise ValueError("cancellation was not exercised")
            return event


def quality(event):
    if event.get("type") != "complete":
        raise ValueError("missing completion")
    stats = event["stats"]
    if type(stats["generated_tokens"]) is not int or not 0 < stats["generated_tokens"] <= 32:
        raise ValueError("empty token output")
    for key in (
        "prompt_tokens",
        "cached_prompt_tokens",
        "evaluated_prompt_tokens",
        "context_used",
        "context_limit",
    ):
        if type(stats[key]) is not int or stats[key] < 0:
            raise ValueError("invalid context counters")
    if (
        stats["prompt_tokens"] <= 0
        or stats["cached_prompt_tokens"] + stats["evaluated_prompt_tokens"]
        != stats["prompt_tokens"]
        or stats["context_used"] != stats["prompt_tokens"] + stats["generated_tokens"]
        or stats["context_used"] > stats["context_limit"]
    ):
        raise ValueError("inconsistent context counters")
    if not event["cache_context"] or not event["token_hash"]:
        raise ValueError("empty output oracle")
    for key in ("decode_tps", "prefill_tps", "ttft_seconds"):
        if not math.isfinite(stats[key]) or stats[key] <= 0:
            raise ValueError("invalid timing")
    keys = (
        "generated_tokens",
        "prompt_tokens",
        "cached_prompt_tokens",
        "evaluated_prompt_tokens",
        "context_used",
        "context_limit",
        "mtp_proposed_tokens",
        "mtp_accepted_tokens",
        "mtp_blocks",
    )
    return {
        "token_hash": event["token_hash"],
        "text_sha256": hashlib.sha256(event["cache_context"].encode()).hexdigest(),
        "stats": {key: stats.get(key) for key in keys},
    }


def run(
    engine,
    model,
    head,
    *,
    measure=footprint,
    result=None,
    progress=None,
    timeout=180,
    memory_saver=False,
    short_only=False,
):
    result = {} if result is None else result
    records = result.setdefault("records", [])
    result.update(
        status="running",
        reaped=False,
        memory_saver=memory_saver,
        engine_sha256=hashlib.sha256(Path(engine).read_bytes()).hexdigest(),
    )
    with tempfile.TemporaryDirectory(prefix="idle-memory-") as directory:
        stderr = Path(directory) / "stderr.log"
        with stderr.open("w+") as log:
            command = [
                str(engine),
                "--registry",
                str(Path(directory) / "models.json"),
                "bridge",
                str(model),
                "--context-length",
                "4096" if short_only else "32768",
            ]
            try:
                with JsonProcess(command, stderr=log) as child:
                    deadline = time.monotonic() + timeout
                    while True:
                        event = child.receive(deadline)
                        if event["type"] == "error":
                            raise ValueError(event)
                        if event["type"] == "ready":
                            if memory_saver and event.get("memory_saver_supported") is not True:
                                raise ValueError("engine does not support memory saving")
                            break
                    ready_bytes = measure(child.process.pid)
                    result["ready_bytes"] = ready_bytes
                    short = "Explain this table.\n" + "1 2 3 4 5 6 7 8 9 10\n" * 30
                    long = "Summarize the data.\n" + "1 2 3 4 5 6 7 8 9 10\n" * 480
                    cases = [
                        ("short-normal", short, 0, False, False),
                        ("short-mtp", short, 2, False, False),
                        ("cache-cold", short, 2, True, False),
                        ("cache-warm", short, 3, True, False),
                        ("invalid-depth", short, 4, True, False),
                        ("cache-after-error", short, 3, True, False),
                        ("long-mtp", long, 2, True, False),
                        ("cancel", short, 2, False, True),
                        ("recovery", short, 2, False, False),
                    ]
                    if short_only:
                        cases = [case for case in cases if case[0] != "long-mtp"]
                    for index, (label, text, depth, reuse, cancel) in enumerate(cases):
                        request = {
                            "type": "generate",
                            "request_id": str(index),
                            "conversation_id": "idle-fixture",
                            "max_tokens": 256 if cancel else 32,
                            "messages": [{"role": "user", "content": text}],
                            "temperature": 0,
                            "top_k": 1,
                            "repetition_penalty": 1,
                            "mtp": depth > 0,
                            "mtp_depth": depth,
                            "mtp_head_path": str(head),
                            "reuse_prompt_cache": reuse,
                            "memory_saver": memory_saver,
                        }
                        output = exchange(child, request, cancel=cancel, timeout=timeout)
                        expected = "cancelled" if cancel else "error" if depth == 4 else "complete"
                        if output["type"] != expected:
                            raise ValueError(f"{label}: expected {expected}, got {output['type']}")
                        oracle = quality(output) if expected == "complete" else None
                        pong = exchange(child, {"type": "ping", "request_id": f"ping-{index}"})
                        if pong["type"] != "pong":
                            raise ValueError("missing idle acknowledgement")
                        if memory_saver:
                            memory = pong.get("memory", {})
                            if (
                                type(memory.get("mlx_cache_bytes")) is not int
                                or memory["mlx_cache_bytes"] != 0
                            ):
                                raise ValueError("unused allocator buffers retained")
                            if oracle is not None and oracle["stats"]["cached_prompt_tokens"] != 0:
                                raise ValueError("prompt cache retained in memory saving mode")
                        idle_bytes = measure(child.process.pid)
                        if (
                            label.startswith("cache-")
                            and not memory_saver
                            and label != "cache-cold"
                            and oracle["stats"]["cached_prompt_tokens"] < 256
                        ):
                            raise ValueError("cache reuse was not exercised")
                        if label == "long-mtp" and oracle["stats"]["prompt_tokens"] < 8192:
                            raise ValueError("long context was not exercised")
                        records.append(
                            {
                                "case": label,
                                "quality": oracle,
                                "idle_bytes": idle_bytes,
                                "idle_memory": pong.get("memory"),
                                "generation_memory": output.get("stats", {}).get("memory"),
                                "timings": {
                                    key: output.get("stats", {}).get(key)
                                    for key in ("decode_tps", "prefill_tps", "ttft_seconds")
                                },
                            }
                        )
                        print(f"{label}: idle {idle_bytes / 1e9:.3f} GB", flush=True)
                        if progress is not None:
                            progress()
                    child.send({"type": "shutdown", "request_id": "shutdown"}, time.monotonic() + 5)
                    if child.process.wait(timeout=10) != 0:
                        raise ValueError("engine failed to exit")
            finally:
                log.flush()
                log.seek(0)
                result["stderr"] = log.read()
            result.update(status="complete", reaped=True)
            return result


def validate_runs(runs, *, short_only=False):
    if not runs or len(runs) < 2:
        raise ValueError("need both comparison arms")
    reference = runs[0]["records"]
    names = [
        "short-normal",
        "short-mtp",
        "cache-cold",
        "cache-warm",
        "invalid-depth",
        "cache-after-error",
        "long-mtp",
        "cancel",
        "recovery",
    ]
    if short_only:
        names.remove("long-mtp")
    if [row["case"] for row in reference] != names or sum(
        row["quality"] is not None for row in reference
    ) != len(names) - 2:
        raise ValueError("incomplete workload")
    for run in runs:
        if run.get("status") != "complete" or run.get("reaped") is not True:
            raise ValueError("engine did not complete and exit")
        if len(run["records"]) != len(reference):
            raise ValueError("incomplete workload")
        for expected, actual in zip(reference, run["records"], strict=True):
            expected_quality, actual_quality = (
                deepcopy_quality(expected["quality"]),
                deepcopy_quality(actual["quality"]),
            )
            if run.get("memory_saver") or runs[0].get("memory_saver"):
                for quality_value in (expected_quality, actual_quality):
                    if quality_value is not None:
                        quality_value["stats"].pop("cached_prompt_tokens")
                        quality_value["stats"].pop("evaluated_prompt_tokens")
            if actual["case"] != expected["case"] or actual_quality != expected_quality:
                raise ValueError("output, state reuse or MTP counters changed")
            if run.get("memory_saver"):
                memory = actual.get("idle_memory", {})
                if type(memory.get("mlx_cache_bytes")) is not int or memory["mlx_cache_bytes"] != 0:
                    raise ValueError("unused allocator buffers retained")
                if (
                    actual["quality"] is not None
                    and actual["quality"]["stats"]["cached_prompt_tokens"] != 0
                ):
                    raise ValueError("prompt cache retained in memory saving mode")
            if type(actual["idle_bytes"]) is not int or actual["idle_bytes"] <= 0:
                raise ValueError("invalid footprint")
        short = run["records"][0]["quality"]
        for row in run["records"]:
            if (
                row["quality"] is not None
                and row["case"] != "long-mtp"
                and (
                    row["quality"]["token_hash"] != short["token_hash"]
                    or row["quality"]["text_sha256"] != short["text_sha256"]
                )
            ):
                raise ValueError("MTP or recovery differs from ordinary generation")


def deepcopy_quality(value):
    return None if value is None else {**value, "stats": dict(value["stats"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "candidate", "model", "head", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--order", choices=("AB", "BA"), default="AB")
    parser.add_argument("--candidate-memory-saver", action="store_true")
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--exchange-timeout", type=int, choices=(180, 300), default=180)
    parser.add_argument("--campaign-timeout", type=int, choices=(900, 1200), default=900)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    report = {
        "status": "running",
        "parity": False,
        "runs": [],
        "model": str(args.model),
        "head": str(args.head),
        "order": args.order,
        "candidate_memory_saver": args.candidate_memory_saver,
        "short_only": args.short_only,
        "exchange_timeout": args.exchange_timeout,
        "campaign_timeout": args.campaign_timeout,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("memory campaign interrupted")

    previous = {
        signum: signal.signal(signum, interrupted) for signum in (signal.SIGTERM, signal.SIGALRM)
    }
    signal.alarm(args.campaign_timeout)
    try:
        for arm in args.order:
            current = {"arm": arm}
            report["runs"].append(current)
            try:
                run(
                    args.baseline if arm == "A" else args.candidate,
                    args.model,
                    args.head,
                    result=current,
                    progress=save,
                    timeout=args.exchange_timeout,
                    memory_saver=args.candidate_memory_saver and arm == "B",
                    short_only=args.short_only,
                )
            finally:
                # JsonProcess's context has completed its terminate/kill/wait cleanup.
                current["reaped"] = True
            save()
        validate_runs(report["runs"], short_only=args.short_only)
        report.update(status="complete", parity=True)
    except BaseException as error:
        report.update(status="failed", parity=False, error=repr(error))
        raise
    finally:
        signal.alarm(0)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        save()


if __name__ == "__main__":
    main()
