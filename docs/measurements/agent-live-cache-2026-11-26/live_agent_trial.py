#!/usr/bin/env python3
"""AGENT-2026-11-26-LIVE-CACHE (4): two Agent turns with real tools on the real bridge, with or
without the live prompt cache. Indicative: sampled at the app's temperature, one pass per run.

usage: live_agent_trial.py ENGINE REGISTRY MODEL HELPER OUT.json live|exact [NAME=VALUE ...]

The sandbox server is the Desktop app's own (`HELPER --agent-tools`, native tools) on a
throw-away folder: `calc.py` has a wrong `mean`, `test_calc.py` fails. Turn 1 asks to fix it and
run the test; turn 2 asks for a `median` with its test. The second turn is sent the way the
app sends it back (`turn_context` of the first one).

For every step the report keeps the prompt tokens, the tokens resumed (and how many of them
from the live state), the tokens read and the prefill seconds; the totals; the task verdict
(`python3 test_calc.py` exits 0 after each turn, `median` defined after the second). A bridge
error, a timeout or a missing `complete` is a failed run: `status` is `complete` only when both
turns finished.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bridge import Bridge  # noqa: E402

engine, registry_source, model, helper, out, mode = sys.argv[1:7]
environment = dict(pair.split("=", 1) for pair in sys.argv[7:])
assert mode in ("live", "exact"), mode
# Inkling 2 Mini generation_config: temperature 1, nothing else (the app's default path).
SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "presence_penalty": 0.0}
SYSTEM = ("You are a local coding agent working in the user's project folder. Use the sandbox tools to read "
          "files, edit them and run commands. Verify your changes by running the tests. Use relative paths. "
          "Never claim to have run something unless a tool result confirms it. When you are done, stop "
          "calling tools and give a short final answer.")
TURNS = [
    "The test in test_calc.py fails. Find the bug in calc.py, fix it, and run `python3 test_calc.py` to confirm.",
    "Now add a function median(xs) to calc.py, add a test for it in test_calc.py, and run the tests again.",
]

work = Path(tempfile.mkdtemp(prefix="mlxl3-live-agent-"))
root = work / "files"
root.mkdir()
(root / "calc.py").write_text(
    '"""Small statistics helpers."""\n\n\ndef total(xs):\n    return sum(xs)\n\n\n'
    'def mean(xs):\n    """Arithmetic mean of a non-empty list."""\n    return total(xs) / (len(xs) - 1)\n\n\n'
    'def spread(xs):\n    return max(xs) - min(xs)\n')
(root / "test_calc.py").write_text(
    'from calc import mean, spread, total\n\n\ndef main():\n    assert total([1, 2, 3]) == 6\n'
    '    assert spread([4, 9, 2]) == 7\n    assert mean([2, 4, 6]) == 4, mean([2, 4, 6])\n'
    '    assert mean([5]) == 5\n    print("all tests passed")\n\n\nif __name__ == "__main__":\n    main()\n')
(root / "README.md").write_text("# calc\n\nRun `python3 test_calc.py`.\n")
project = work / "studio" / "project.json"
project.parent.mkdir()
project.write_text(json.dumps({"permissions": {
    "read": True, "write": True, "run": True, "commandSeconds": 20, "toolHarness": "native",
    "internet": False, "internetHosts": []}}))
registry = work / "registry" / "models.json"
registry.parent.mkdir()
entry = json.loads(Path(registry_source).read_text())["models"][model]
registry.write_text(json.dumps({"version": 1, "models": {model: entry}}))
(registry.parent / "mcp.json").write_text(json.dumps({"version": 1, "mcpServers": {
    "exa": {"enabled": False, "url": "https://mcp.exa.ai/mcp"},
    "sandbox": {"enabled": True, "command": str(Path(helper).resolve()),
                "args": ["--agent-tools", str(root), str(project), "long-calls"],
                "callTimeoutSeconds": 43}}}))


def test_passes():
    try:
        run = subprocess.run(["python3", "test_calc.py"], cwd=root, capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return False
    return run.returncode == 0 and "all tests passed" in run.stdout


report = {"id": "AGENT-2026-11-26-LIVE-CACHE", "engine": engine, "helper": helper, "model": model,
          "mode": mode, "environment": environment, "sampling": SAMPLING, "status": "running", "turns": []}
bridge = Bridge(engine, registry, model, Path(out).with_suffix(".stderr.log"), environment)
try:
    ready = bridge.ready()
    report["ready"] = {key: ready.get(key) for key in ("runtime_commit", "runtime_version",
                                                        "live_prompt_cache_supported", "dspark_default")}
    bridge.send({"type": "set_mcp", "enabled": True, "request_id": "mcp"})
    while (item := bridge.event(120)).get("type") != "mcp_status":
        pass
    report["mcp"] = {"tools": item.get("mcp_tools"), "errors": item.get("mcp_errors")}
    if not item.get("mcp_tools"):
        raise RuntimeError(f"no sandbox tool: {item.get('mcp_errors')}")
    messages = [{"role": "system", "content": SYSTEM}]
    for index, text in enumerate(TURNS, 1):
        messages.append({"role": "user", "content": text})
        request = {"type": "generate", "request_id": f"turn-{index}", "conversation_id": "live-agent-trial",
                   "messages": messages, "max_tokens": -1, "tool_rounds": 256, "mcp_enabled": True, **SAMPLING}
        if mode == "live":
            request["live_prompt_cache"] = True
        done, tools, results, statuses = bridge.generate(request, seconds=1200)
        steps = [{"prompt_tokens": step.get("prompt_tokens"), "cached_prompt_tokens": step.get("cached_prompt_tokens"),
                  "live_prompt_tokens": step.get("live_prompt_tokens"),
                  "read_tokens": step.get("evaluated_prompt_tokens"), "generated_tokens": step.get("generated_tokens"),
                  "prefill_seconds": step.get("prefill_seconds"), "decode_seconds": step.get("decode_seconds")}
                 for step in done.get("round_stats") or []]
        if not steps or any(not isinstance(step["prompt_tokens"], int) for step in steps):
            raise RuntimeError(f"turn {index}: no step statistics")
        stats = done.get("stats") or {}
        report["turns"].append({
            "turn": index, "steps": steps, "tools": tools, "tool_errors": sum(1 for r in results if r["error"]),
            "results": results, "stop_reason": done.get("stop_reason"), "answer": done.get("assistant_context"),
            "prompt_tokens": stats.get("prompt_tokens"), "cached_prompt_tokens": stats.get("cached_prompt_tokens"),
            "live_prompt_tokens": stats.get("live_prompt_tokens"),
            "read_tokens": stats.get("evaluated_prompt_tokens"), "generated_tokens": stats.get("generated_tokens"),
            "prefill_seconds": sum(step["prefill_seconds"] or 0 for step in steps),
            "elapsed_seconds": stats.get("elapsed_seconds"), "tool_seconds": stats.get("tool_seconds"),
            "test_passes": test_passes(), "median_defined": "def median" in (root / "calc.py").read_text(),
            "memory": stats.get("memory")})
        # the app sends a finished turn back as its stored exchanges
        history = {"role": "assistant", "content": done.get("cache_context") or done.get("assistant_context") or ""}
        if done.get("turn_context"):
            history["turn_context"] = done["turn_context"]
        messages.append(history)
    report["status"] = "complete"
except Exception as error:  # the report keeps the failure
    report["status"] = "failed"
    report["error"] = f"{type(error).__name__}: {error}"
finally:
    report["engine_exit"] = bridge.close()
    turns = report["turns"]
    report["totals"] = {key: sum(turn[key] or 0 for turn in turns) for key in (
        "prompt_tokens", "cached_prompt_tokens", "read_tokens", "generated_tokens", "prefill_seconds", "elapsed_seconds")}
    report["totals"]["steps"] = sum(len(turn["steps"]) for turn in turns)
    report["totals"]["live_steps"] = sum(1 for turn in turns for step in turn["steps"] if step["live_prompt_tokens"])
    report["task_passed"] = bool(len(turns) == 2 and turns[0]["test_passes"] and turns[1]["test_passes"]
                                 and turns[1]["median_defined"])
    Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    shutil.rmtree(work, ignore_errors=True)
for turn in report["turns"]:
    print(f"turn {turn['turn']}: {len(turn['steps'])} steps, tools {turn['tools']}, test passes {turn['test_passes']}, "
          f"{turn['stop_reason']}, {turn['elapsed_seconds']:.1f} s")
    for number, step in enumerate(turn["steps"], 1):
        print(f"   step {number}: prompt {step['prompt_tokens']:>6} cached {step['cached_prompt_tokens']:>6} "
              f"(live {step['live_prompt_tokens']}) read {step['read_tokens']:>5} generated {step['generated_tokens']:>5} "
              f"prefill {step['prefill_seconds']:.2f} s")
print(json.dumps({"status": report["status"], "error": report.get("error"), "task_passed": report["task_passed"],
                  "totals": report["totals"]}))
sys.exit(0 if report["status"] == "complete" else 1)
