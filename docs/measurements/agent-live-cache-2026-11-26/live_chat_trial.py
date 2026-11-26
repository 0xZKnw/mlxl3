#!/usr/bin/env python3
"""AGENT-2026-11-26-LIVE-CACHE (3): a conversation without tools on the real bridge, with the
live prompt cache: does every turn go on from the state the turn before it decoded?

usage: live_chat_trial.py ENGINE REGISTRY MODEL OUT.json [live|exact] [NAME=VALUE ...]

Five turns, greedy, complete answers (512 tokens at most, low reasoning effort). With `live`,
each turn after the first must resume the previous prompt and all it generated (the end token
too when the model had already committed it): `cached_prompt_tokens` is the previous prompt
plus its generated tokens, or one more. With `exact` the request has no `live_prompt_cache`
and each turn resumes at the last multiple of 128 of the previous prompt.
`as_expected` is true only when all five turns finished on an end token and resumed there.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bridge import Bridge  # noqa: E402

engine, registry, model, out = sys.argv[1:5]
mode = sys.argv[5] if len(sys.argv) > 5 else "live"
environment = dict(pair.split("=", 1) for pair in sys.argv[6:])
assert mode in ("live", "exact"), mode


def filler(label, sentences):
    return " ".join(f"{label} note {i}: the value of row {i} is {i * 37 % 101}." for i in range(sentences))


TURNS = [
    "Keep these notes in mind. " + filler("alpha", 173) + " Reply with the single word ready.",
    "More notes. " + filler("beta", 9) + " Reply with the single word ready.",
    "More notes. " + filler("gamma", 52) + " Reply with the single word ready.",
    "What is the value of row 3 in the alpha notes? Reply with the number only.",
    "And the value of row 5 in the gamma notes? Reply with the number only.",
]
report = {"engine": engine, "model": model, "mode": mode, "environment": environment, "turns": [],
          "as_expected": False}
bridge = Bridge(engine, registry, model, Path(out).with_suffix(".stderr.log"), environment)
try:
    ready = bridge.ready()
    report["ready"] = {key: ready.get(key) for key in ("runtime_commit", "runtime_version",
                                                        "live_prompt_cache_supported", "dspark_default")}
    messages, previous = [], None
    for index, text in enumerate(TURNS, 1):
        messages.append({"role": "user", "content": text})
        request = {"type": "generate", "request_id": f"turn-{index}", "conversation_id": "live-chat-trial",
                   "messages": messages, "max_tokens": 512, "temperature": 0.0, "dspark": True,
                   "reasoning_effort": "low"}
        if mode == "live":
            request["live_prompt_cache"] = True
        done, _, _, _ = bridge.generate(request)
        stats = done.get("stats") or {}
        prompt, cached, generated = (stats.get(key) for key in ("prompt_tokens", "cached_prompt_tokens", "generated_tokens"))
        if not all(isinstance(value, int) for value in (prompt, cached, generated)) or not done.get("token_hash"):
            raise RuntimeError(f"turn {index}: incomplete statistics {stats!r}")
        if previous is None:
            expected = [0]
        elif mode == "live":
            held = previous["prompt_tokens"] + previous["generated_tokens"]
            expected = [held, held + 1]
        else:
            expected = [previous["prompt_tokens"] // 128 * 128]
        turn = {"turn": index, "prompt_tokens": prompt, "cached_prompt_tokens": cached,
                "live_prompt_tokens": stats.get("live_prompt_tokens"), "expected_cached": expected,
                "resumed": cached in expected, "evaluated_prompt_tokens": prompt - cached,
                "generated_tokens": generated, "stop_reason": done.get("stop_reason"),
                "token_hash": done["token_hash"], "answer": done.get("cache_context"),
                "shown": done.get("assistant_context"),
                "prefill_seconds": stats.get("prefill_seconds"), "ttft_seconds": stats.get("ttft_seconds"),
                "decode_tps": stats.get("decode_tps"), "memory": stats.get("memory")}
        report["turns"].append(turn)
        messages.append({"role": "assistant", "content": done.get("cache_context") or ""})
        previous = turn
    report["as_expected"] = len(report["turns"]) == len(TURNS) and all(
        turn["resumed"] and turn["stop_reason"] == "eos" for turn in report["turns"])
except Exception as error:  # the report keeps the failure
    report["error"] = f"{type(error).__name__}: {error}"
finally:
    report["engine_exit"] = bridge.close()
    Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
for turn in report["turns"]:
    print(f"turn {turn['turn']}: prompt {turn['prompt_tokens']:>5} cached {turn['cached_prompt_tokens']:>5} "
          f"(live {turn['live_prompt_tokens']}) expected {turn['expected_cached']} read {turn['evaluated_prompt_tokens']:>5} "
          f"generated {turn['generated_tokens']:>4} {turn['stop_reason']} prefill {turn['prefill_seconds']:.2f} s "
          f"shown {turn['shown']!r}")
print(json.dumps({key: report.get(key) for key in ("as_expected", "error", "engine_exit")}))
sys.exit(1 if report.get("error") else 0)
