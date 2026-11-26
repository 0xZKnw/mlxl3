#!/usr/bin/env python3
"""CACHE-2026-11-26-INK2M-SNAPSHOT-BUDGET: does every turn of a growing conversation resume
from the prompt cache of the turn before it?

usage: cache_budget_trial.py ENGINE REGISTRY MODEL OUT.json [NAME=VALUE ...]

One conversation without tools on the real bridge, greedy, 16 tokens per answer. Each turn is
the turn before it, its answer, and a new user message whose size is chosen so the prefill
after the resume covers one chunk, then a span of five chunks or more, then one chunk again.
`NAME=VALUE` pairs go to the engine's environment (the cache budget, the trace).

For each turn the report keeps the prompt tokens, the tokens resumed, the resume expected
(the last multiple of 128 of the prompt before it; 0 on the first turn or with the cache
off), the token hash and the answer. `resumed_as_expected` is true only when every turn
resumed exactly there; `error` holds any failure. Every wait is bounded and the bridge is
shut down on any exit. The engine's stderr goes to OUT.stderr.log.
"""
import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

CHUNK = 128
engine, registry, model, out = sys.argv[1:5]
overrides = dict(pair.split("=", 1) for pair in sys.argv[5:])
cache_off = overrides.get("MLXL3_PROMPT_CACHE_MIB") == "0"


def filler(label, sentences):
    return " ".join(f"{label} note {i}: the value of row {i} is {i * 37 % 101}." for i in range(sentences))


# sizes in sentences (about 16 tokens each), tuned to the prefill pattern named above
TURNS = [
    ("Keep these notes in mind. " + filler("alpha", 173) + " Reply with the word ready.", "cold"),
    ("More notes. " + filler("beta", 9) + " Reply with the word ready.", "resume, one or two chunks"),
    ("More notes. " + filler("gamma", 52) + " Reply with the word ready.", "resume, a span of five chunks or more"),
    ("More notes. " + filler("delta", 6) + " Reply with the word ready.", "resume from the turn before"),
    ("More notes. " + filler("epsilon", 6) + " Reply with the word ready.", "resume from the turn before"),
]

report = {"engine": engine, "model": model, "environment": overrides, "turns": [],
          "resumed_as_expected": False}
stderr_path = Path(out).with_suffix(".stderr.log")
stderr = stderr_path.open("w")
process = subprocess.Popen([engine, "--registry", registry, "bridge", model, "--context-length", "1048576"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                           env={**os.environ, **overrides})
selector = selectors.DefaultSelector()
selector.register(process.stdout, selectors.EVENT_READ)
pending = bytearray()


def event(seconds):
    """The next bridge event. Bytes are read from the pipe itself and split here: a line
    waiting in a reader's buffer would never wake the selector."""
    deadline = time.monotonic() + seconds
    while b"\n" not in pending:
        left = deadline - time.monotonic()
        if left <= 0 or not selector.select(timeout=left):
            raise RuntimeError(f"no event within {seconds} s")
        data = os.read(process.stdout.fileno(), 65536)
        if not data:
            raise RuntimeError(f"bridge exited ({process.poll()})")
        pending.extend(data)
    line, _, rest = bytes(pending).partition(b"\n")
    pending[:] = rest
    try:
        return json.loads(line)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid bridge line: {line[:200]!r}") from error


def send(message):
    process.stdin.write((json.dumps(message) + "\n").encode())
    process.stdin.flush()


def generate(request_id, messages):
    send({"type": "generate", "request_id": request_id, "conversation_id": "cache-budget-trial",
          "messages": messages, "max_tokens": 16, "temperature": 0.0, "dspark": True})
    started = time.monotonic()
    while True:
        item = event(900)
        if item.get("request_id") not in (None, request_id):
            continue
        kind = item.get("type")
        if kind in ("error", "cancelled"):
            raise RuntimeError(f"{request_id}: {item.get('message', kind)}")
        if kind == "complete":
            item["wall_seconds"] = time.monotonic() - started
            return item


try:
    while (item := event(900)).get("type") != "ready":
        if item.get("type") == "error":
            raise RuntimeError(item.get("message"))
    report["ready"] = {key: item.get(key) for key in ("runtime_commit", "runtime_version", "dspark_supported",
                                                       "dspark_default") if key in item}
    messages, previous_prompt = [], 0
    for index, (text, purpose) in enumerate(TURNS, 1):
        messages.append({"role": "user", "content": text})
        done = generate(f"turn-{index}", messages)
        stats = done.get("stats") or {}
        prompt, cached = stats.get("prompt_tokens"), stats.get("cached_prompt_tokens")
        if not isinstance(prompt, int) or not isinstance(cached, int) or not done.get("token_hash"):
            raise RuntimeError(f"turn {index}: incomplete statistics {stats!r}")
        expected = 0 if cache_off else previous_prompt // CHUNK * CHUNK
        report["turns"].append({
            "turn": index, "purpose": purpose, "prompt_tokens": prompt, "cached_prompt_tokens": cached,
            "expected_cached": expected, "resumed": cached == expected,
            "prefilled_after_resume": prompt - cached,
            "token_hash": done["token_hash"], "answer": done.get("cache_context"),
            "generated_tokens": stats.get("generated_tokens"), "dspark_blocks": stats.get("dspark_blocks"),
            "ttft_seconds": stats.get("ttft_seconds"), "prefill_seconds": stats.get("prefill_seconds"),
            "prefill_tps": stats.get("prefill_tps"), "wall_seconds": done["wall_seconds"],
            "memory": stats.get("memory"),
        })
        messages.append({"role": "assistant", "content": done.get("cache_context") or done.get("assistant_context") or ""})
        previous_prompt = prompt
    report["resumed_as_expected"] = len(report["turns"]) == len(TURNS) and all(t["resumed"] for t in report["turns"])
except Exception as error:  # the report keeps the failure
    report["error"] = f"{type(error).__name__}: {error}"
finally:
    if process.poll() is None:
        try:
            send({"type": "shutdown"})
            process.wait(timeout=60)
        except Exception:
            process.kill()
            process.wait()
    stderr.close()
    report["engine_exit"] = process.returncode
    Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
for turn in report["turns"]:
    print(f"turn {turn['turn']}: prompt {turn['prompt_tokens']:>6} cached {turn['cached_prompt_tokens']:>6} "
          f"expected {turn['expected_cached']:>6} prefill {turn['prefill_seconds']} s  hash {turn['token_hash'][:12]}")
print(json.dumps({key: report.get(key) for key in ("resumed_as_expected", "error", "engine_exit")}))
sys.exit(1 if report.get("error") else 0)
