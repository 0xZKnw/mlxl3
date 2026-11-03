"""Qualify native AR JSONL generation, rejection and both cancellation phases."""

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import tempfile
import time
from pathlib import Path

from scripts.native_json_process import JsonProcess, positive_seconds

GOLDEN = "9f850af1cbba074b"  # Retained independent two-token bridge oracle, #104/#108.
_spec = importlib.util.spec_from_file_location(
    "flash_benchmark", Path(__file__).parents[1] / "scripts/benchmark-flash-bridge.py"
)
_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_benchmark)


def complete(event: dict, mode: dict, text: str) -> str:
    """Validate the actual production completion before accepting its token hash.

    raises: AssertionError, KeyError, TypeError, ValueError
    post: bool(__return__)
    """
    assert type(event["stats"]["generated_tokens"]) is int
    assert type(event["stats"]["decode_tokens"]) is int
    _benchmark.validate(event, mode, False, 2)
    assert event["token_hash"] == GOLDEN and text.strip() and event["cache_context"]
    return event["token_hash"]


def verify(engine, model, root, report, timeout=180.0):
    with (
        (root / "stderr.log").open("x") as error,
        (root / "events.jsonl").open("x") as raw,
        tempfile.TemporaryDirectory(prefix="mlxl3-ar-check-") as temp,
        JsonProcess(
            [
                str(engine),
                "--registry",
                str(Path(temp) / "models.json"),
                "bridge",
                str(model),
                "--context-length",
                "256",
            ],
            stderr=error,
            env=os.environ,
        ) as child,
    ):

        def until(wanted, identity, deadline, cancel_at=None):
            mode, text, sent = {}, "", False
            while True:
                event = child.receive(deadline)
                raw.write(json.dumps(event, allow_nan=False) + "\n")
                raw.flush()
                assert isinstance(event.get("type"), str), event
                if event.get("request_id", "") != identity:
                    continue
                kind = event["type"]
                if kind == "generation_mode":
                    mode = event
                if kind == "delta":
                    text += event.get("text", "")
                if cancel_at == kind and not sent:
                    child.process.send_signal(signal.SIGUSR1)
                    sent = True
                if kind == wanted:
                    assert not cancel_at or sent, "cancellation phase was not reached"
                    return event, mode, text
                assert kind not in {"error", "complete", "cancelled"}, event

        until("ready", "", time.monotonic() + timeout)
        messages = [{"role": "user", "content": "Combien font 2 + 2 ?"}]

        def generate(identity, cancel_at=None, invalid=False):
            deadline = time.monotonic() + timeout
            child.send(
                {
                    "type": "generate",
                    "request_id": identity,
                    "messages": [] if invalid else messages,
                    "max_tokens": 128 if cancel_at else 2,
                    "temperature": 0,
                    "top_k": 1,
                    "repetition_penalty": 1,
                    "mtp": False,
                    "dflash2": False,
                    "reuse_prompt_cache": False,
                },
                deadline,
            )
            return until(
                "error" if invalid else "cancelled" if cancel_at else "complete",
                identity,
                deadline,
                cancel_at,
            )

        first, mode, text = generate("baseline")
        complete(first, mode, text)
        baseline = (first["token_hash"], text, first["cache_context"])
        refused, _, _ = generate("invalid", invalid=True)
        assert refused.get("message"), "missing production rejection"
        for phase in (None, "context_usage", "delta"):
            if phase:
                generate("cancel-" + phase, cancel_at=phase)
            event, mode, text = generate("recover-" + str(phase))
            complete(event, mode, text)
            assert (event["token_hash"], text, event["cache_context"]) == baseline
        deadline = time.monotonic() + timeout
        child.send({"type": "flash_memory", "request_id": "ledger"}, deadline)
        memory, _, _ = until("flash_memory", "ledger", deadline)
        ledger = memory["ledger"]
        slots = ledger["expert_slots_by_layer"]
        assert len(slots) == 48 and all(type(n) is int and n == 53 for n in slots)
        depth = int(os.environ.get("MLXL3_SSD_LOOKAHEAD_DEPTH", "0"))
        if depth:
            assert ledger["prefetch"]["submitted"] > 0 and ledger["prefetch"]["published"] > 0
        report.update(ledger=ledger, requested_lookahead=depth)
        child.send({"type": "shutdown"}, time.monotonic() + 5)
        assert child.process.wait(timeout=10) == 0
    images = [
        s.lower() for s in (root / "stderr.log").read_text().splitlines() if s.startswith("dyld[")
    ]
    assert any("/metal.framework/" in s for s in images)
    assert not any(k in s for s in images for k in ("/libmlx", "/libjaccl", "/site-packages/mlx/"))
    report.update(
        status="verified",
        parity=True,
        token_hash=GOLDEN,
        completed=4,
        cancelled=["context_usage", "delta"],
        rejected=1,
        mlx_loaded=False,
        text_sha256=hashlib.sha256(baseline[1].encode()).hexdigest(),
        dyld_images=len(images),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "model", "report"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--timeout", type=positive_seconds, default=180.0)
    args = parser.parse_args()
    report = {
        "status": "started",
        "parity": False,
        "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
    }
    with args.report.open("x") as sink:
        try:
            root = args.report.with_suffix("")
            root.mkdir()
            verify(args.engine, args.model, root, report, args.timeout)
        except BaseException as error:
            report.update(status="failed", parity=False, error=repr(error))
            raise
        finally:
            json.dump(report, sink, indent=2)
            sink.write("\n")


if __name__ == "__main__":
    main()
