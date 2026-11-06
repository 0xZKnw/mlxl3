"""Bounded ordinary Qwen bridge migration gate; timings are not benchmarks."""

import argparse
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_json_process import JsonProcess


def completion(event, text, budget):
    count = event.get("stats", {}).get("generated_tokens")
    if type(count) is not int or not 0 < count <= budget:
        raise ValueError("invalid Qwen generated token count")
    if not text or not isinstance(event.get("token_hash"), str) or not event["token_hash"]:
        raise ValueError("empty Qwen text/hash")
    history = event.get("cache_context")
    if not isinstance(history, str) or not history:
        raise ValueError("empty Qwen history")
    return {"count": count, "hash": event["token_hash"], "text": text, "history": history}


def exercise(engine, model, output, *, native, timeout=180, mtp_head=None):
    if not 0 < timeout < float("inf"):
        raise ValueError("finite positive timeout required")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    report = {"status": "in_progress", "parity": False, "reaped": False, "results": results}
    child = None
    try:
        with tempfile.TemporaryDirectory(prefix="qwen-native-bridge-") as tmp:
            with (
                (output / "stderr.log").open("w") as err,
                (output / "events.jsonl").open("w") as events,
                JsonProcess(
                    [
                        str(engine),
                        "--registry",
                        str(Path(tmp) / "models.json"),
                        "bridge",
                        str(model),
                        "--context-length",
                        "4096",
                    ],
                    stderr=err,
                    env={**os.environ, "DYLD_PRINT_LIBRARIES": "1"},
                ) as child,
            ):

                def until(kind, rid=None, cancel=None):
                    deadline = time.monotonic() + timeout
                    text = ""
                    sent = False
                    while True:
                        event = child.receive(deadline)
                        events.write(json.dumps(event) + "\n")
                        events.flush()
                        if rid is not None and event.get("request_id") != rid:
                            continue
                        if event.get("type") == "error" and kind != "error":
                            raise RuntimeError(event)
                        if event.get("type") == "delta":
                            text += event.get("text", "")
                        if event.get("type") == cancel and not sent:
                            child.process.send_signal(signal.SIGUSR1)
                            sent = True
                        if event.get("type") == kind:
                            return event, text

                ready, _ = until("ready")
                if native and (
                    ready.get("backend") != "direct-metal"
                    or ready.get("mtp_supported") is not (mtp_head is not None)
                    or ready.get("dflash_supported") is not False
                    or ready.get("mtp_max_depth") != (3 if mtp_head else 0)
                    or ready.get("flash") is not None
                ):
                    raise RuntimeError("incorrect native Qwen capabilities")

                def request(rid, budget=8, *, cancel=None, error=False, **options):
                    data = {
                        "type": "generate",
                        "request_id": rid,
                        "conversation_id": "qwen-native",
                        "messages": [
                            {
                                "role": "user",
                                "content": "Compte de 1 à 20, séparés par des virgules.",
                            }
                        ],
                        "max_tokens": budget,
                        "temperature": 0,
                        "top_k": 1,
                        "repetition_penalty": 1,
                        "reuse_prompt_cache": False,
                        **options,
                    }
                    child.send(data, time.monotonic() + timeout)
                    event, text = until(
                        "error" if error else "cancelled" if cancel else "complete", rid, cancel
                    )
                    return (
                        event
                        if error or cancel
                        else (completion(event, text, budget), event["stats"])
                    )

                for budget in (1, 2, 8):
                    result, stats = request(f"cold-{budget}", budget)
                    if (
                        native
                        and ready.get("qwen_device_greedy") is True
                        and (
                            stats.get("native_gpu_greedy_selections") != budget
                            or stats.get("native_gpu_greedy_heads") != budget - 1
                        )
                    ):
                        raise RuntimeError("Qwen GPU greedy counters differ from output budget")
                    results.append(result)
                request("invalid-budget", 0, error=True)
                if native:
                    request("unsupported-dflash2", error=True, dflash2=True)
                    if mtp_head is None:
                        request("unsupported-mtp", error=True, mtp=True)
                    else:
                        opts = {"mtp": True, "mtp_head_path": str(mtp_head)}
                        request(
                            "mtp-invalid-head",
                            error=True,
                            **{**opts, "mtp_head_path": str(mtp_head / "missing")},
                        )
                        request("mtp-invalid-depth", error=True, mtp_depth=4, **opts)
                        selections = []
                        for depth in (1, 2, 3):
                            for budget, reference in zip((1, 2, 8), results, strict=True):
                                actual, stats = request(
                                    f"mtp-{depth}-{budget}", budget, mtp_depth=depth, **opts
                                )
                                counts = stats.get("mtp_depth_blocks")
                                proposed = stats.get("mtp_proposed_tokens")
                                accepted = stats.get("mtp_accepted_tokens")
                                if (
                                    actual != reference
                                    or not isinstance(counts, list)
                                    or len(counts) != 4
                                    or any(type(count) is not int or count < 0 for count in counts)
                                    or any(counts[depth + 1 :])
                                    or type(proposed) is not int
                                    or type(accepted) is not int
                                    or proposed != sum(i * n for i, n in enumerate(counts))
                                    or not 0 <= accepted <= proposed
                                    or (budget == 8 and proposed == 0)
                                ):
                                    raise RuntimeError(
                                        f"MTP output or counters diverged at depth {depth}, budget {budget}"
                                    )
                                selections.append(
                                    {
                                        "depth": depth,
                                        "budget": budget,
                                        "proposed": proposed,
                                        "accepted": accepted,
                                        "blocks": counts,
                                    }
                                )
                        if not any(row["accepted"] for row in selections):
                            raise RuntimeError("MTP accepted-prefix path was never exercised")
                        report["mtp_selections"] = selections
                        for phase in ("context_usage", "delta"):
                            request(f"mtp-cancel-{phase}", 64, cancel=phase, mtp_depth=3, **opts)
                            recovery, _ = request(f"mtp-recovery-{phase}", mtp_depth=3, **opts)
                            if recovery != results[-1]:
                                raise RuntimeError("Qwen MTP cancellation changed recovery")
                    cold, _ = request("cache-cold", reuse_prompt_cache=True)
                    warm, stats = request("cache-warm", reuse_prompt_cache=True)
                    if cold != warm or stats.get("cached_prompt_tokens", 0) <= 0:
                        raise RuntimeError("native Qwen prefix cache missed or changed output")
                    other, stats = request(
                        "cache-other", reuse_prompt_cache=True, conversation_id="other"
                    )
                    if cold != other or stats.get("cached_prompt_tokens") != 0:
                        raise RuntimeError("Qwen cache crossed conversation ownership")
                    for phase in ("context_usage", "delta"):
                        request(f"cancel-{phase}", 64, cancel=phase)
                        recovery, _ = request(f"recovery-{phase}")
                        if recovery != results[-1]:
                            raise RuntimeError("Qwen cancellation changed recovery")
                    # Deterministic top-k1 also exercises the non-unit penalty fallback.
                    _, sampling_stats = request("sampling", repetition_penalty=1.1)
                    if sampling_stats.get("native_gpu_greedy_heads") is not None:
                        raise RuntimeError("non-unit penalty entered Qwen GPU greedy")
                child.send({"type": "shutdown"}, time.monotonic() + timeout)
                if child.process.wait(timeout=10) != 0:
                    raise RuntimeError("Qwen bridge shutdown failed")
            if native:
                images = (output / "stderr.log").read_text().lower()
                if "/metal.framework/" not in images or any(
                    v in images for v in ("/libmlx", "/libjaccl", "/site-packages/mlx/")
                ):
                    raise RuntimeError("native Qwen loaded MLX or failed to report Metal")
        report.update(status="validated", parity=True)
        return results
    except BaseException as error:
        report.update(status="failed", parity=False, error=repr(error))
        raise
    finally:
        report["reaped"] = child is not None and child.process.poll() is not None
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stock", type=Path)
    parser.add_argument("native", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mtp-head", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "in_progress",
        "parity": False,
        "sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (("stock", args.stock), ("native", args.native))
        },
    }
    try:
        reference = exercise(
            args.stock.resolve(), args.model.resolve(), args.output / "stock", native=False
        )
        actual = exercise(
            args.native.resolve(),
            args.model.resolve(),
            args.output / "native",
            native=True,
            mtp_head=args.mtp_head.resolve() if args.mtp_head else None,
        )
        if reference != actual:
            raise ValueError("native Qwen bridge differs from independent MLX bridge")
        report.update(status="validated", parity=True, compared_completions=len(actual))
    except BaseException as error:
        report.update(status="failed", parity=False, error=repr(error))
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
