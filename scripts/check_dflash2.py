"""Bounded production-bridge checks for both DFlash2 families; no speed claims."""

import argparse
import json
import math
import signal
import tempfile
import time
from pathlib import Path

from native_json_process import JsonProcess, positive_seconds


def equal(reference, candidate):
    """Require nonempty target parity, including the committed chat history."""
    for event in (reference, candidate):
        stats = event["stats"]
        assert isinstance(event["token_hash"], str) and len(event["token_hash"]) == 16
        assert isinstance(event["cache_context"], str) and event["cache_context"]
        assert type(stats["generated_tokens"]) is int and stats["generated_tokens"] > 0
        assert all(
            type(stats[key]) in (int, float) and math.isfinite(stats[key]) and stats[key] >= 0
            for key in ("decode_seconds", "ttft_seconds", "elapsed_seconds")
        )
    for key in ("token_hash", "cache_context"):
        assert reference[key] == candidate[key], f"DFlash changed {key}"
    assert reference["stats"]["generated_tokens"] == candidate["stats"]["generated_tokens"]


def run(engine, model, draft, *, timeout=900, tune=True, output=None, foreign_draft=None):
    report = {"status": "running", "parity": None, "checks": 0, "events": []}

    def save():
        if output:
            Path(output).write_text(json.dumps(report, indent=2) + "\n")

    save()
    deadline = time.monotonic() + timeout
    try:
        with (
            tempfile.TemporaryDirectory(prefix="dflash2-check-") as directory,
            JsonProcess(
                [
                    engine,
                    "--registry",
                    str(Path(directory) / "models.json"),
                    "bridge",
                    model,
                    "--context-length",
                    "4096",
                ]
            ) as bridge,
        ):
            count = 0

            def until(kind, request=None, cancel_at=None):
                cancelled = False
                while True:
                    event = bridge.receive(deadline)
                    report["events"].append(event)
                    if request is not None and event.get("request_id") != request:
                        continue
                    if event["type"] == "error" and kind != "error":
                        raise RuntimeError(event.get("message", "bridge error"))
                    if cancel_at == event["type"] and not cancelled:
                        bridge.process.send_signal(signal.SIGUSR1)
                        cancelled = True
                    if event["type"] == kind:
                        return event

            def request(
                kind="generate", mode=0, budget=17, cancel_at=None, terminal=None, **options
            ):
                nonlocal count
                count += 1
                value = {
                    "type": kind,
                    "request_id": str(count),
                    "conversation_id": "dflash2-check",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write complete Python code to multiply two matrices with comments and examples.",
                        }
                    ],
                    "max_tokens": budget,
                    "temperature": 0,
                    "top_k": 1,
                    "repetition_penalty": 1,
                    "dflash2": mode > 0,
                    "dflash_mode": max(1, mode),
                    "dflash_draft_path": draft,
                    "reuse_prompt_cache": False,
                }
                value.update(options)
                bridge.send(value, deadline)
                result = until(
                    terminal
                    or (
                        "cancelled"
                        if cancel_at
                        else "dflash_tune_complete"
                        if kind == "tune_dflash"
                        else "dflash_status"
                        if kind == "set_dflash"
                        else "complete"
                    ),
                    str(count),
                    cancel_at,
                )
                report["checks"] += 1
                return result

            ready = until("ready")
            assert ready["dflash_supported"] is True and ready["dflash_tune_supported"] is True
            report["ready"] = ready
            assert request("set_dflash", enabled=True)["dflash_active"] is True
            for budget in (1, 2, 3, 8, 17, 64):
                baseline = request(budget=budget)
                for mode in (1, 2, 3):
                    actual = request(mode=mode, budget=budget)
                    equal(baseline, actual)
                    assert actual["stats"]["generated_tokens"] <= budget
            for mode in (0, 4):
                request(mode=1, dflash_mode=mode, terminal="error")
            request(mode=1, mtp=True, terminal="error")
            request(mode=1, repetition_penalty=1.1, terminal="error")
            request(
                "set_dflash",
                enabled=True,
                dflash_draft_path=str(Path(directory) / "missing"),
                terminal="error",
            )
            if foreign_draft:
                request(
                    "set_dflash", enabled=True, dflash_draft_path=foreign_draft, terminal="error"
                )
            golden = request()
            for phase in ("context_usage", "delta"):
                request(mode=3, budget=256, cancel_at=phase)
                equal(golden, request(mode=3))
            prompt = "Explain this table in detail.\n" + "1 2 3 4 5 6 7 8 9 10\n" * 18
            messages = [{"role": "user", "content": prompt}]
            cold = request(mode=2, messages=messages)
            request(mode=2, messages=messages, reuse_prompt_cache=True)
            warm = request(mode=3, messages=messages, reuse_prompt_cache=True)
            equal(cold, warm)
            assert warm["stats"]["cached_prompt_tokens"] >= 256
            miss = request(
                mode=3, messages=messages, reuse_prompt_cache=True, conversation_id="other"
            )
            equal(cold, miss)
            assert miss["stats"]["cached_prompt_tokens"] == 0
            assert request("set_dflash", enabled=False)["dflash_active"] is False
            if tune:
                request("tune_dflash", cancel_at="dflash_tune_progress")
                equal(golden, request(mode=3))
                result = request("tune_dflash")
                assert result["tuning_key"] == ready["dflash_tuning_key"]
                rows = result["rows"]
                assert [r["depth"] for r in rows] == [0, 1, 2, 3]
                assert all(r["token_hashes"] == rows[0]["token_hashes"] for r in rows)
                assert all(math.isfinite(r["decode_tps"]) and r["decode_tps"] > 0 for r in rows)
                baseline = int(rows[0]["decode_tps"] * 1000)
                candidates = [
                    r
                    for r in rows[1:]
                    if r["eligible"] and int(r["decode_tps"] * 1000) * 100 > baseline * 103
                ]
                best = (
                    max(candidates, key=lambda r: (int(r["decode_tps"] * 1000), -r["depth"]))[
                        "depth"
                    ]
                    if candidates
                    else 0
                )
                assert best == result["best_depth"]
                report["tuning"] = result
                equal(golden, request(mode=best))
        report.update(status="complete", parity=True)
    except BaseException as error:
        report.update(status="failed", parity=None, error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("model")
    parser.add_argument("draft")
    parser.add_argument("--timeout", type=positive_seconds, default=900)
    parser.add_argument("--skip-tune", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--foreign-draft")
    args = parser.parse_args()
    result = run(
        args.engine,
        args.model,
        args.draft,
        timeout=args.timeout,
        tune=not args.skip_tune,
        output=args.output,
        foreign_draft=args.foreign_draft,
    )
    print(
        json.dumps({key: value for key, value in result.items() if key not in ("events", "tuning")})
    )
