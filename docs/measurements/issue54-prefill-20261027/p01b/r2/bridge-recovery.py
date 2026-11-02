"""Real Rust bridge cancellation/recovery; existing bounded transport and validator."""

import hashlib
import json
import runpy
import signal
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "scripts"))
from native_json_process import JsonProcess


def run(engine, model, draft, output):
    output.mkdir(exist_ok=False)
    report = {
        "status": "incomplete",
        "records": [],
        "binary_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    prompt = json.loads((output.parent / "prompts.json").read_text())["LONG"]
    validate = runpy.run_path("scripts/benchmark-flash-bridge.py")["validate"]
    child = None
    try:
        with (
            tempfile.TemporaryDirectory(prefix="prefill-e2e-") as directory,
            (output / "stderr.log").open("w") as log,
        ):
            with JsonProcess(
                [
                    str(engine),
                    "--registry",
                    str(Path(directory) / "models.json"),
                    "bridge",
                    str(model),
                    "--context-length",
                    "4096",
                ],
                stderr=log,
            ) as child:
                deadline = time.monotonic() + 180
                while True:
                    event = child.receive(deadline)
                    if event.get("type") == "error":
                        raise RuntimeError(event)
                    if event.get("type") == "ready":
                        break

                def request(dflash, label, *, cancel_at=None, too_long=False):
                    rid = str(len(report["records"]))
                    deadline = time.monotonic() + 180
                    child.send(
                        {
                            "type": "generate",
                            "request_id": rid,
                            "messages": [
                                {"role": "user", "content": prompt * 32 if too_long else prompt}
                            ],
                            "max_tokens": 256 if cancel_at else 17,
                            "temperature": 0,
                            "top_k": 1,
                            "repetition_penalty": 1,
                            "mtp": False,
                            "dflash2": dflash,
                            "dflash_mode": 1,
                            "dflash_draft_path": str(draft),
                            "reuse_prompt_cache": False,
                            "mcp_enabled": False,
                        },
                        deadline,
                    )
                    mode = None
                    sent = False
                    while True:
                        event = child.receive(deadline)
                        kind = event.get("type")
                        if kind == "error" and event.get("request_id") not in (None, rid):
                            raise RuntimeError(event)
                        if event.get("request_id") != rid:
                            if kind == "error":
                                raise RuntimeError(event)
                            continue
                        if kind == "generation_mode":
                            mode = event
                        if kind == cancel_at and not sent:
                            child.process.send_signal(signal.SIGUSR1)
                            sent = True
                        if kind not in ("complete", "cancelled", "error"):
                            continue
                        if cancel_at:
                            assert sent and kind == "cancelled", event
                        elif too_long:
                            assert (
                                kind == "error" and "context" in event.get("message", "").lower()
                            ), event
                        else:
                            assert kind == "complete", event
                            validate(event, mode, False, 17, dflash)
                        report["records"].append(
                            {
                                "label": label,
                                "dflash": dflash,
                                "cancel_at": cancel_at,
                                "response": event,
                            }
                        )
                        save()
                        return event

                goldens = []
                for dflash in [False, True]:
                    golden = request(dflash, "golden")
                    goldens.append(golden)
                    for phase in ["context_usage", "delta"]:
                        request(dflash, "cancel", cancel_at=phase)
                        actual = request(dflash, "recover")
                        assert (actual["token_hash"], actual["cache_context"]) == (
                            golden["token_hash"],
                            golden["cache_context"],
                        )
                    request(dflash, "over-context", too_long=True)
                    actual = request(dflash, "recover-error")
                    assert (actual["token_hash"], actual["cache_context"]) == (
                        golden["token_hash"],
                        golden["cache_context"],
                    )
                assert (goldens[0]["token_hash"], goldens[0]["cache_context"]) == (
                    goldens[1]["token_hash"],
                    goldens[1]["cache_context"],
                )
                assert len(report["records"]) == 14
                child.send({"type": "shutdown"}, time.monotonic() + 10)
                assert child.process.wait(timeout=10) == 0
        report.update(status="validated", reaped=child.process.poll() is not None, parity=True)
        save()
        print(
            json.dumps(
                {
                    "status": "validated",
                    "requests": 14,
                    "cancellations": 4,
                    "errors": 2,
                    "parity": True,
                }
            ),
            flush=True,
        )
    except BaseException as error:
        report.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            reaped=child is not None and child.process.poll() is not None,
        )
        save()
        raise


if __name__ == "__main__":
    run(*(Path(x) for x in sys.argv[1:]))
