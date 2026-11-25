#!/usr/bin/env python3
"""Reproduction of AGENT-2026-11-25-INKLING-HEADER: Agent tool calls of Inkling 2 Mini.

usage: repro.py MLXL3 REGISTRY MODEL HELPER OUT_PREFIX [TEMPERATURE ...]

Starts one bridge (one model on the GPU), offers `sandbox.javascript` from HELPER
(`--agent-tools`), and sends the same short Agent request once per temperature. Every bridge
event goes to OUT_PREFIX.events.jsonl and the engine's stderr to OUT_PREFIX.stderr.log (a
diagnostic engine prints the raw step text there). Every wait is bounded and the bridge is
stopped on any exit. The summary counts read and unreadable calls; it claims nothing else.
"""
import json
import os
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

engine, registry_source, model, helper, prefix = sys.argv[1:6]
temperatures = [float(value) for value in sys.argv[6:]] or [0.7]
work = Path(tempfile.mkdtemp(prefix="mlxl3-inkling-header-"))
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
# MLXL3_REPRO_MESSAGES: a JSON file of messages to send instead of the short request.
MESSAGES = (json.loads(Path(os.environ["MLXL3_REPRO_MESSAGES"]).read_text()) if os.environ.get("MLXL3_REPRO_MESSAGES")
            else [{"role": "user", "content": "Regarde l'état du projet avec l'outil javascript "
                                              "(liste les fichiers), puis dis-moi combien il y en a."}])
events = open(prefix + ".events.jsonl", "w")
stderr = open(prefix + ".stderr.log", "w")
summary = {"engine": engine, "helper": helper, "model": model, "runs": [], "state": "started"}
process = subprocess.Popen([engine, "--registry", str(registry), "bridge", model, "--context-length", "16384"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1,
                           env={"PATH": "/usr/bin:/bin", "MLXL3_TMP_TRACE_STEP": "1"})
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
        events.write(line)
        return json.loads(line)


def send(message):
    process.stdin.write(json.dumps(message) + "\n")
    process.stdin.flush()


def generate(request_id, temperature):
    send({"type": "generate", "request_id": request_id, "reuse_prompt_cache": False, "conversation_id": request_id,
          "temperature": temperature, "max_tokens": 2048, "tool_rounds": 4, "mcp_enabled": True,
          "messages": MESSAGES})
    calls, results = [], []
    while True:
        item = event()
        if item.get("request_id") not in (None, request_id):
            continue
        kind = item.get("type")
        if kind == "tool_start":
            calls.append({"name": item.get("tool_name"), "arguments": item.get("arguments")})
        elif kind == "tool_result":
            results.append({"error": item.get("is_error"), "text": (item.get("text") or "")[:200]})
        elif kind == "error":
            return {"temperature": temperature, "error": item.get("message")}
        elif kind == "complete":
            return {"temperature": temperature, "calls": calls, "results": results,
                    "unreadable": sum(c["name"] == "invalid_tool_call" for c in calls),
                    "read": sum(c["name"] != "invalid_tool_call" for c in calls),
                    "stop_reason": item.get("stop_reason")}


try:
    while (item := event(900)).get("type") != "ready":
        if item.get("type") == "error":
            raise RuntimeError(item.get("message"))
    summary["ready"] = {k: item.get(k) for k in ("runtime_commit", "runtime_version") if k in item}
    send({"type": "set_mcp", "enabled": True, "request_id": "mcp"})
    while (item := event()).get("type") != "mcp_status":
        pass
    for index, temperature in enumerate(temperatures):
        summary["runs"].append(generate(f"run-{index}", temperature))
    summary["state"] = "complete"
except Exception as error:  # the failure is part of the report
    summary["state"] = "failed"
    summary["error"] = str(error)
finally:
    try:
        send({"type": "shutdown"})
        process.wait(timeout=20)
    except Exception:
        process.kill()
        process.wait(timeout=20)
    events.close()
    stderr.close()
    shutil.rmtree(work, ignore_errors=True)
    Path(prefix + ".json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"state": summary["state"], "error": summary.get("error"),
                      "runs": [{k: r.get(k) for k in ("temperature", "read", "unreadable", "error")} for r in summary["runs"]]}))
    sys.exit(0 if summary["state"] == "complete" else 1)
