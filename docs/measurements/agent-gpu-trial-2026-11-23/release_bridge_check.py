#!/usr/bin/env python3
"""Release check of a published engine archive on the installed Inkling 2 Mini.

usage: release_bridge_check.py MLXL3 REGISTRY MODEL HELPER OUT.json

1. Agent tool call: the model is offered `sandbox.javascript` and must run it; the program's
   result (the number of files of a 3-file project) must come back from the tool.
2. DSpark on demand: greedy requests with `dspark: true`, then without the field, then
   `dspark: false`: blocks only when asked, and a complete answer each time.
Every wait is bounded; the bridge is shut down on any exit. `passed` is true only when every
check holds.
"""
import json
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

engine, registry_source, model, helper, out = sys.argv[1:6]
work = Path(tempfile.mkdtemp(prefix="mlxl3-release-bridge-"))
root, studio = work / "files", work / "studio"
for folder in (root / "src", studio):
    folder.mkdir(parents=True)
for name in ("a.txt", "b.txt", "src/c.txt"):
    (root / name).write_text(name)
(studio / "project.json").write_text(json.dumps({"permissions": {"harness": "javascript", "commandSeconds": 5}}))
registry = work / "registry" / "models.json"
registry.parent.mkdir()
entry = json.loads(Path(registry_source).read_text())["models"][model]
registry.write_text(json.dumps({"version": 1, "models": {model: entry}}))
(registry.parent / "mcp.json").write_text(json.dumps({"version": 1, "mcpServers": {
    "sandbox": {"enabled": True, "command": str(Path(helper).resolve()),
                "args": ["--agent-tools", str(root), str(studio / "project.json"), "long-calls"],
                "callTimeoutSeconds": 43}}}))
report = {"engine": engine, "model": model, "checks": {}, "passed": False}
process = subprocess.Popen([engine, "--registry", str(registry), "bridge", model, "--context-length", "16384"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                           text=True, bufsize=1)
selector = selectors.DefaultSelector()
selector.register(process.stdout, selectors.EVENT_READ)


def event(seconds=300):
    deadline = time.monotonic() + seconds
    while True:
        left = deadline - time.monotonic()
        if left <= 0 or not selector.select(timeout=left):
            raise RuntimeError(f"no event within {seconds} s")
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(f"bridge exited ({process.poll()})")
        return json.loads(line)


def send(message):
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()


def generate(request_id, extra, messages):
    send({"type": "generate", "request_id": request_id, "messages": messages, "reuse_prompt_cache": False,
          "conversation_id": request_id, "temperature": 0.0, **extra})
    answer, calls, results = "", [], []
    while True:
        item = event()
        if item.get("request_id") not in (None, request_id):
            continue
        kind = item.get("type")
        if kind == "delta" and item.get("phase") == "answer":
            answer += item["text"]
        elif kind == "tool_start":
            calls.append({"name": item.get("tool_name"), "arguments": item.get("arguments")})
        elif kind == "tool_result":
            results.append({"error": item.get("is_error"), "text": item.get("text", "")[:400]})
        elif kind == "error":
            return {"error": item.get("message")}
        elif kind == "complete":
            stats = item.get("stats", {})
            return {"answer": answer[-400:], "calls": calls, "results": results,
                    "stop_reason": item.get("stop_reason"), "dspark_blocks": stats.get("dspark_blocks"),
                    "tokens": stats.get("generated_tokens")}


try:
    while (item := event(900)).get("type") != "ready":
        if item.get("type") == "error":
            raise RuntimeError(item.get("message"))
    report["ready"] = {k: item.get(k) for k in ("runtime_commit", "runtime_version", "tool_rounds_supported",
                                                "mcp_call_timeout_supported", "dspark_supported") if k in item}
    send({"type": "set_mcp", "enabled": True, "request_id": "mcp"})
    while (item := event()).get("type") != "mcp_status":
        pass
    tool = generate("tool", {"max_tokens": 4096, "tool_rounds": 24, "mcp_enabled": True}, [
        {"role": "user", "content": "Use the javascript tool to run `return list().length` in my project, "
                                    "then tell me how many files it has."}])
    report["checks"]["tool"] = tool
    tool_ok = (not tool.get("error") and any(c["name"] == "sandbox.javascript" and c["arguments"] for c in tool.get("calls", []))
               and any("3" in r["text"] and not r["error"] for r in tool.get("results", []))
               and "<|content_invoke_tool_json|>" not in tool.get("answer", ""))
    send({"type": "set_mcp", "enabled": False, "request_id": "mcp-off"})
    while (item := event()).get("type") != "mcp_status":
        pass
    prompt = [{"role": "user", "content": "Write two sentences about the sea."}]
    on = generate("dspark-on", {"max_tokens": 96, "dspark": True}, prompt)
    plain = generate("dspark-absent", {"max_tokens": 96}, prompt)
    off = generate("dspark-off", {"max_tokens": 96, "dspark": False}, prompt)
    report["checks"].update({"dspark_on": on, "dspark_absent": plain, "dspark_off": off})
    dspark_ok = (all(not r.get("error") and r.get("tokens") for r in (on, plain, off))
                 and (on.get("dspark_blocks") or 0) > 0 and not plain.get("dspark_blocks") and not off.get("dspark_blocks"))
    report["passed"] = bool(tool_ok and dspark_ok)
    report["tool_ok"], report["dspark_ok"] = tool_ok, dspark_ok
except Exception as error:  # the report keeps the failure
    report["error"] = f"{type(error).__name__}: {error}"
finally:
    Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    if process.poll() is None:
        try:
            send({"type": "shutdown"})
            process.wait(timeout=30)
        except Exception:
            process.kill()
            process.wait()
    shutil.rmtree(work, ignore_errors=True)
print(json.dumps({k: report.get(k) for k in ("passed", "tool_ok", "dspark_ok", "ready", "error")}, indent=1))
sys.exit(0 if report["passed"] else 1)
