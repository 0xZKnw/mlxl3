"""Check managed MTP selection, unload and greedy parity through the real CLI.

Uses disposable registries and bounded/reaped subprocesses. Run alone on an
Apple GPU with two local target checkpoints; timings are diagnostic only.
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from native_json_process import JsonProcess, positive_seconds


def matching_completions(expected: dict, actual: dict, budget: int) -> bool:
    """Reject empty/incomplete outputs before comparing the target token hashes.

    post: not __return__ or expected['token_hash'] == actual['token_hash']
    post: not __return__ or 0 < actual['stats']['generated_tokens'] <= budget
    post: not __return__ or expected['cache_context'] == actual['cache_context']
    """
    left = expected.get("stats")
    right = actual.get("stats")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    count = right.get("generated_tokens")
    return (
        type(count) is int
        and 0 < count <= budget
        and type(left.get("generated_tokens")) is int
        and count == left["generated_tokens"]
        and isinstance(expected.get("token_hash"), str)
        and bool(expected["token_hash"])
        and expected["token_hash"] == actual.get("token_hash")
        and isinstance(expected.get("cache_context"), str)
        and bool(expected["cache_context"])
        and expected["cache_context"] == actual.get("cache_context")
    )


def run(engine, models, timeout, report):
    heads = []
    for model in models:
        installed = subprocess.run(
            [engine, "mtp-head", "--target", model],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
        events = [json.loads(line) for line in installed.stdout.splitlines()]
        paths = [e.get("path") for e in events if e.get("type") == "installed"]
        if len(paths) != 1 or not isinstance(paths[0], str) or not paths[0]:
            raise RuntimeError("missing installed MTP head")
        heads.append(paths[0])
    if heads[0] == heads[1]:
        raise RuntimeError("dense and MoE selected the same MTP head")
    with tempfile.TemporaryDirectory(prefix="mtp-switch-check-") as directory:
        for index in (0, 1, 0):
            with JsonProcess(
                [
                    engine,
                    "--registry",
                    str(Path(directory) / "models.json"),
                    "bridge",
                    models[index],
                    "--context-length",
                    "4096",
                ]
            ) as process:

                def exchange(request=None, expected="ready", error=False):
                    deadline = time.monotonic() + timeout
                    if request is not None:
                        process.send(request, deadline)
                    while True:
                        event = process.receive(deadline)
                        if request and event.get("request_id") != request["request_id"]:
                            continue
                        if event.get("type") in ("error", "cancelled"):
                            if error and event.get("type") == "error":
                                return event
                            raise RuntimeError(event)
                        if event.get("type") == expected:
                            return event

                ready = exchange()
                if not ready.get("mtp_auto_download_supported") or not ready.get(
                    "mtp_configure_supported"
                ):
                    raise RuntimeError("managed MTP capabilities missing")

                configuration_id = 0

                def configure(enabled, head=heads[index], error=False):
                    nonlocal configuration_id
                    configuration_id += 1
                    return exchange(
                        {
                            "type": "set_mtp",
                            "request_id": f"configure-{configuration_id}",
                            "enabled": enabled,
                            "mtp_head_path": head,
                        },
                        "mtp_status",
                        error,
                    )

                before = configure(False)
                loaded = configure(True)
                again = configure(True)
                base = before["memory"]["mlx_active_bytes"]
                if (
                    before.get("mtp_active") is not False
                    or loaded.get("mtp_active") is not True
                    or loaded.get("mtp_head_path") != heads[index]
                    or loaded["memory"]["mlx_active_bytes"] <= base
                    or again["memory"]["mlx_active_bytes"] != loaded["memory"]["mlx_active_bytes"]
                ):
                    raise RuntimeError("MTP load/reuse memory mismatch")
                for invalid in (heads[1 - index], str(Path(directory) / "missing-head")):
                    failure = configure(True, invalid, error=True)
                    if (
                        failure.get("mtp_active") is not False
                        or failure["memory"]["mlx_active_bytes"] != base
                    ):
                        raise RuntimeError("failed MTP load retained the previous head")
                    configure(True)
                unloaded = configure(False)
                if (
                    unloaded.get("mtp_active") is not False
                    or unloaded["memory"]["mlx_active_bytes"] != base
                ):
                    raise RuntimeError("MTP OFF retained head allocations")
                cases = []
                for budget in (1, 3, 17):
                    baseline = None
                    for depth in range(4):
                        result = exchange(
                            {
                                "type": "generate",
                                "request_id": f"{budget}-{depth}",
                                "conversation_id": "switch-check",
                                "reuse_prompt_cache": False,
                                "messages": [
                                    {
                                        "role": "user",
                                        "content": "Écris une fonction Python de multiplication de matrices et explique-la.",
                                    }
                                ],
                                "max_tokens": budget,
                                "temperature": 0,
                                "top_k": 1,
                                "repetition_penalty": 1,
                                "mtp": depth > 0,
                                "mtp_depth": max(1, depth),
                                "mtp_head_path": heads[index],
                            },
                            "complete",
                        )
                        if baseline is None:
                            baseline = result
                        if not matching_completions(baseline, result, budget):
                            raise RuntimeError(
                                "MTP changed target IDs/history/budget or returned empty output"
                            )
                        cases.append(
                            {
                                "budget": budget,
                                "depth": depth,
                                "token_hash": result["token_hash"],
                                "stats": result["stats"],
                            }
                        )
                report["models"].append(
                    {
                        "model": models[index],
                        "pid": process.process.pid,
                        "ready": ready,
                        "before": before,
                        "loaded": loaded,
                        "unloaded": unloaded,
                        "cases": cases,
                    }
                )
            if process.process.poll() is None:
                raise RuntimeError("previous model process was not reaped")
            print(
                json.dumps(
                    {
                        "model": models[index],
                        "head": heads[index],
                        "cases": len(cases),
                        "parity": True,
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("dense")
    parser.add_argument("moe")
    parser.add_argument("--timeout", type=positive_seconds, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "parity": False, "models": []}
    args.output.write_text(json.dumps(report))
    try:
        run(args.engine, [args.dense, args.moe], args.timeout, report)
        report.update(status="passed", parity=True)
    except BaseException as error:
        report.update(status="failed", parity=False, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
