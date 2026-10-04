"""Exercise native MTP through the production Desktop JSON-lines bridge."""

import argparse
import json
import queue
import signal
import subprocess
import tempfile
import threading
from pathlib import Path


def run(engine, model, head, budgets, prompt, context=4096, check_cancel=True, check_prefix=False):
    with tempfile.TemporaryDirectory(prefix="mlxl3-mtp-check-") as directory:
        registry = str(Path(directory) / "models.json")
        process = subprocess.Popen(
            [engine, "--registry", registry, "bridge", model, "--context-length", str(context)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        events = queue.Queue()

        def read():
            try:
                for line in process.stdout:
                    events.put(json.loads(line))
            finally:
                events.put({"type": "error", "message": "bridge exited"})

        threading.Thread(target=read, daemon=True).start()

        def until(kind, request_id=None, cancel_at=None, expect_error=False):
            mode = None
            sent = False
            text = ""
            while True:
                event = events.get(timeout=60)
                if event["type"] == "error":
                    if request_id and event.get("request_id") not in (None, request_id):
                        continue
                    if expect_error:
                        return event, mode, text
                    raise RuntimeError(event)
                if request_id and event.get("request_id") != request_id:
                    continue
                if event["type"] == "generation_mode":
                    mode = event
                if event["type"] == "delta":
                    text += event.get("text", "")
                if event["type"] == cancel_at and not sent:
                    process.send_signal(signal.SIGUSR1)
                    sent = True
                if event["type"] == kind:
                    return event, mode, text

        count = 0

        def generate(enabled, budget, cancel_at=None, expect_error=False, **overrides):
            nonlocal count
            count += 1
            request_id = str(count)
            request = {
                "type": "generate",
                "request_id": request_id,
                "conversation_id": "mtp-fixture",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": budget,
                "temperature": 0,
                "top_k": 1,
                "repetition_penalty": 1,
                "mtp": enabled,
                "mtp_head_path": head,
                "reuse_prompt_cache": False,
                **overrides,
            }
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            result, mode, text = until(
                "cancelled" if cancel_at else "complete", request_id, cancel_at, expect_error
            )
            if expect_error:
                return result, text
            expected = (
                enabled
                and (request["temperature"] == 0 or request["top_k"] == 1)
                and request["repetition_penalty"] == 1
            )
            if mode is None or mode.get("mtp_active") is not expected:
                raise RuntimeError(f"engine mode mismatch: {mode}")
            return result, text

        try:
            ready, _, _ = until("ready")
            print(
                json.dumps({"runtime": ready, "head": head, "prompt": prompt, "context": context}),
                flush=True,
            )
            # Compile both paths before collecting the comparison. These are
            # explicitly warmups and are retained in the evidence.
            for enabled in (False, True):
                result, _ = generate(enabled, max(budgets))
                print(json.dumps({"warmup": enabled, "stats": result["stats"]}), flush=True)
            for budget in budgets:
                results = []
                for enabled in (False, True, True, False):
                    result, text = generate(enabled, budget)
                    results.append((enabled, result, text))
                hashes = {item[1].get("token_hash") for item in results}
                texts = {item[2] for item in results}
                contexts = {item[1]["cache_context"] for item in results}
                counts = {item[1]["stats"]["generated_tokens"] for item in results}
                if (
                    None in hashes
                    or len(hashes) != 1
                    or len(texts) != 1
                    or len(contexts) != 1
                    or len(counts) != 1
                ):
                    raise RuntimeError(
                        "MTP changed target token IDs, text, history or output budget"
                    )
                if any(item[1]["stats"]["generated_tokens"] > budget for item in results):
                    raise RuntimeError("MTP exceeded the output budget")
                print(
                    json.dumps(
                        {
                            "budget": budget,
                            "parity": True,
                            "token_hash": hashes.pop(),
                            "runs": [
                                {"mtp": enabled, "stats": item["stats"]}
                                for enabled, item, _ in results
                            ],
                        }
                    ),
                    flush=True,
                )
            if check_prefix:
                for suffix in ("", "\nGive a second example."):
                    messages = [{"role": "user", "content": prompt + suffix}]
                    cold, text = generate(True, 17, messages=messages, reuse_prompt_cache=False)
                    generate(True, 17, messages=messages, reuse_prompt_cache=True)
                    warm, actual = generate(True, 17, messages=messages, reuse_prompt_cache=True)
                    if cold["token_hash"] != warm["token_hash"] or text != actual:
                        raise RuntimeError("MTP prefix reuse changed target output")
                    if warm["stats"]["cached_prompt_tokens"] <= 0:
                        raise RuntimeError("MTP prefix was not reused; supply a longer prompt")
                    miss, _ = generate(
                        True,
                        17,
                        messages=messages,
                        reuse_prompt_cache=True,
                        conversation_id="different",
                    )
                    if (
                        miss["stats"]["cached_prompt_tokens"] != 0
                        or miss["token_hash"] != cold["token_hash"]
                    ):
                        raise RuntimeError("MTP reused a different conversation")
                    print(
                        json.dumps(
                            {
                                "prefix_parity": True,
                                "suffix": suffix,
                                "cold": cold["stats"],
                                "warm": warm["stats"],
                            }
                        ),
                        flush=True,
                    )
            if check_cancel:
                for invalid in (
                    {"mtp_head_path": ""},
                    {"mtp_head_path": "/missing-head"},
                    {"dflash2": True},
                ):
                    result, _ = generate(True, 8, expect_error=True, **invalid)
                    if result["type"] != "error":
                        raise RuntimeError("invalid MTP request was accepted")
                    print(json.dumps({"rejected_request": invalid}), flush=True)
                golden, text = generate(False, max(budgets))
                for phase in ("context_usage", "delta"):
                    generate(True, 256, cancel_at=phase)
                    result, actual = generate(True, max(budgets))
                    if result["token_hash"] != golden["token_hash"] or text != actual:
                        raise RuntimeError("MTP cancellation polluted the next generation")
                    print(json.dumps({"cancel_at": phase, "recovery_parity": True}), flush=True)
                # Sampling remains ordinary when MTP's v1 greedy contract is
                # unmet. Both requests use the same deterministic greedy mode
                # with a non-unit repetition penalty.
                plain, text = generate(False, 8, repetition_penalty=1.1)
                result, actual = generate(True, 8, repetition_penalty=1.1)
                if result["token_hash"] != plain["token_hash"] or actual != text:
                    raise RuntimeError("MTP fallback changed repetition sampling")
                print(json.dumps({"non_unit_penalty_fallback": True}), flush=True)
            process.stdin.write('{"type":"shutdown"}\n')
            process.stdin.flush()
            if process.wait(timeout=10) != 0:
                raise RuntimeError("bridge shutdown failed")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine")
    parser.add_argument("model")
    parser.add_argument("head")
    parser.add_argument("--tokens", default="1,2,3,17,64")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--skip-cancel", action="store_true")
    parser.add_argument("--check-prefix", action="store_true")
    args = parser.parse_args()
    budgets = [int(value) for value in args.tokens.split(",")]
    if not budgets or min(budgets) <= 0:
        parser.error("token budgets must be positive")
    prompt = (
        args.prompt_file.read_text()
        if args.prompt_file
        else "Explain lossless speculative decoding in one paragraph."
    )
    run(
        args.engine,
        args.model,
        args.head,
        budgets,
        prompt,
        args.context,
        not args.skip_cancel,
        args.check_prefix,
    )
