"""Physical integration check; synthetic history and owned, bounded engine only."""

import importlib.util
import json
import math
import signal
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from native_json_process import JsonProcess

spec = importlib.util.spec_from_file_location("idle", ROOT / "benchmarks/benchmark_idle_memory.py")
idle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(idle)


def validate_tune_result(event, pong):
    if event.get("type") != "mtp_tune_complete":
        raise ValueError("missing Tune completion")
    rows = event["rows"]
    if (
        len(rows) != 4
        or any(type(row["depth"]) is not int for row in rows)
        or sorted(row["depth"] for row in rows) != list(range(4))
    ):
        raise ValueError("incomplete tuning grid")
    hashes = rows[0]["token_hashes"]
    if len(hashes) != 2 or not all(isinstance(h, str) and h for h in hashes):
        raise ValueError("empty Tune oracle")
    for row in rows:
        if (
            row["token_hashes"] != hashes
            or type(row["decode_tokens"]) is not int
            or row["decode_tokens"] != 190
        ):
            raise ValueError("Tune parity/count divergence")
        for key in ["decode_tps", "decode_seconds"]:
            if type(row[key]) not in (float, int) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError("invalid Tune metrics")
        for key in ["accepted_tokens", "proposed_tokens"]:
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError("invalid Tune counters")
        if row["accepted_tokens"] > row["proposed_tokens"]:
            raise ValueError("invalid Tune acceptance")
    if (
        pong.get("type") != "pong"
        or type(pong["memory"]["mlx_cache_bytes"]) is not int
        or pong["memory"]["mlx_cache_bytes"] != 0
    ):
        raise ValueError("Tune did not purge allocator cache")


def main():
    output = Path(sys.argv[1])
    if output.exists():
        raise ValueError("refusing to overwrite evidence")
    report = {"status": "running", "parity": False, "requests": [], "reaped": False}

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    save()

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt("transition check interrupted")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGALRM)}
    signal.alarm(900)
    try:
        with tempfile.TemporaryDirectory(prefix="memory-transition-") as directory:
            stderr = Path(directory) / "stderr.log"
            with stderr.open("w+") as log:
                try:
                    command = [
                        str(ROOT / "build/memory-saver/candidate-rs"),
                        "--registry",
                        str(Path(directory) / "registry.json"),
                        "bridge",
                        sys.argv[2],
                        "--context-length",
                        "4096",
                    ]
                    with JsonProcess(command, stderr=log) as child:
                        deadline = time.monotonic() + 180
                        while True:
                            ready = child.receive(deadline)
                            if ready.get("type") == "error":
                                raise ValueError(ready)
                            if ready.get("type") == "ready":
                                break
                        if ready.get("memory_saver_supported") is not True:
                            raise ValueError("missing capability")
                        text = "Explain this table.\n" + "1 2 3 4 5 6 7 8 9 10\n" * 30
                        reference = None
                        for index, (mode, reused) in enumerate(
                            [
                                (False, False),
                                (False, True),
                                (True, False),
                                (True, False),
                                (False, False),
                                (False, True),
                            ]
                        ):
                            request = {
                                "type": "generate",
                                "request_id": str(index),
                                "conversation_id": "transition",
                                "messages": [{"role": "user", "content": text}],
                                "max_tokens": 32,
                                "temperature": 0,
                                "top_k": 1,
                                "repetition_penalty": 1,
                                "mtp": True,
                                "mtp_depth": 2,
                                "mtp_head_path": sys.argv[3],
                                "reuse_prompt_cache": True,
                                "memory_saver": mode,
                            }
                            event = idle.exchange(child, request)
                            quality = idle.quality(event)
                            cached = quality["stats"]["cached_prompt_tokens"]
                            if (cached >= 256) != reused or (not reused and cached != 0):
                                raise ValueError("wrong reuse across memory policy transition")
                            semantic = idle.deepcopy_quality(quality)
                            semantic["stats"].pop("cached_prompt_tokens")
                            semantic["stats"].pop("evaluated_prompt_tokens")
                            if reference is None:
                                reference = semantic
                            if semantic != reference:
                                raise ValueError(
                                    "tokens/text/context/MTP changed across policy transition"
                                )
                            pong = idle.exchange(child, {"type": "ping", "request_id": f"p{index}"})
                            if pong["type"] != "pong" or (
                                mode and pong["memory"]["mlx_cache_bytes"] != 0
                            ):
                                raise ValueError("idle purge missing")
                            report["requests"].append(
                                {"memory_saver": mode, "quality": quality, "memory": pong["memory"]}
                            )
                            save()
                        request = {
                            "type": "tune_mtp",
                            "request_id": "tune",
                            "mtp_head_path": sys.argv[3],
                            "memory_saver": True,
                        }
                        deadline = time.monotonic() + 240
                        child.send(request, deadline)
                        while True:
                            event = child.receive(deadline)
                            if event.get("request_id") != "tune":
                                raise ValueError("unexpected Tune request identity")
                            if event["type"] in ("error", "cancelled"):
                                raise ValueError(event)
                            if event["type"] == "mtp_tune_complete":
                                break
                        pong = idle.exchange(child, {"type": "ping", "request_id": "after-tune"})
                        validate_tune_result(event, pong)
                        report["tune"] = {"result": event, "idle_memory": pong["memory"]}
                        child.send(
                            {"type": "shutdown", "request_id": "shutdown"}, time.monotonic() + 5
                        )
                        if child.process.wait(timeout=10) != 0:
                            raise ValueError("engine exit failed")
                finally:
                    log.flush()
                    log.seek(0)
                    report["stderr"] = log.read()
                    report["reaped"] = True
        report.update(status="complete", parity=True)
    except BaseException as error:
        report.update(status="failed", parity=False, error=repr(error))
        raise
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        save()


if __name__ == "__main__":
    main()
