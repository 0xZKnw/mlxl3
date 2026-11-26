#!/usr/bin/env python3
"""AGENT-2026-11-26-V21-GPU: the summary of earlier turns on a real model.

Drives the real bridge and the real sandbox MCP server (`--agent-tools`) with a fabricated
conversation whose earlier turns fill a 16,384-token context, then a second turn rebuilt the
way the Desktop app does after a `history_summary` event. Behaviour only: no speed is claimed.

usage: history_summary_trial.py ENGINE REGISTRY MODEL HELPER OUT.json

A timeout, a bridge error or a missing `complete` is a failed turn, never a pass; `status` is
`complete` only when both turns finished. Criteria and their verdicts are in the report.
"""
import hashlib
import json
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

READY_SECONDS = 900
EVENT_SECONDS = 600
CONTEXT = 16384
MAX_TOKENS = 32768          # StudioModel.agentTokenBudget
TOOL_ROUNDS = 24            # StudioModel.agentToolRounds
# Inkling 2 Mini generation_config: temperature 1, nothing else (the app's default path).
SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0, "presence_penalty": 0.0}
MARKER = "[Summary of the earlier conversation"            # agent_loop::HISTORY_SUMMARY_MARKER
TAIL = "[End of the summary. The conversation continues with the user's message below.]\n\n"
CODE, PORT = "PAMPLEMOUSSE-4471", "48213"
SYSTEM = "You are a coding assistant working in a sandboxed project folder. Answer briefly."
TOPICS = ["the build cache", "the login form", "the CSV importer", "the retry queue",
          "the search index", "the nightly report"]


def filler(turn, speaker, lines):
    """Distinct, plausible work-log lines (about 20 tokens each)."""
    topic = TOPICS[turn]
    out = []
    for step in range(lines):
        value = (turn * 131 + step * 17 + (7 if speaker == "user" else 3)) % 997
        if speaker == "user":
            out.append(f"- Note {step + 1} about {topic}: file part_{turn}_{step}.txt changed, "
                       f"expected value {value}, please keep it in mind.")
        else:
            out.append(f"- Item {step + 1} for {topic}: reviewed part_{turn}_{step}.txt, "
                       f"value {value} is consistent, nothing to change.")
    return "\n".join(out)


def conversation():
    messages = [{"role": "system", "content": SYSTEM}]
    for turn in range(6):
        head = f"Here are my notes about {TOPICS[turn]}."
        if turn == 0:
            head += f" Also remember this: the project code word is {CODE}."
        if turn == 1:
            head += f" Also remember this: the staging service listens on port {PORT}."
        messages.append({"role": "user", "content": head + "\n" + filler(turn, "user", 32)})
        messages.append({"role": "assistant",
                         "content": f"Understood. My review of {TOPICS[turn]}:\n" + filler(turn, "assistant", 32)})
    messages.append({"role": "user", "content": "Without calling any tool: what is the project code word I gave you at the start?"})
    return messages


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class BridgeError(RuntimeError):
    pass


def main():
    engine, registry_source, model, helper, out = sys.argv[1:6]
    work = Path(tempfile.mkdtemp(prefix="mlxl3-history-summary-"))
    root = work / "files"
    root.mkdir()
    (root / "README.md").write_text("# Empty project\n")
    project = work / "studio" / "project.json"
    project.parent.mkdir()
    project.write_text(json.dumps({"permissions": {
        "read": True, "write": True, "run": True, "commandSeconds": 5, "toolHarness": "native",
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
    report = {"id": "AGENT-2026-11-26-V21-GPU", "engine": engine, "engine_sha256": sha256(engine),
              "helper": helper, "helper_sha256": sha256(helper), "model": model,
              "model_path": entry.get("path"), "context_length": CONTEXT, "sampling": SAMPLING,
              "max_tokens": MAX_TOKENS, "tool_rounds": TOOL_ROUNDS, "status": "running", "turns": []}

    def save():
        Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))

    stderr = open(Path(out).with_suffix(".stderr.log"), "w")
    process = subprocess.Popen(
        [engine, "--registry", str(registry), "bridge", model, "--context-length", str(CONTEXT)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr, text=True, bufsize=1)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    def read_event(seconds):
        deadline = time.monotonic() + seconds
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not selector.select(timeout=left):
                raise BridgeError(f"no event within {seconds} s")
            line = process.stdout.readline()
            if not line:
                raise BridgeError(f"bridge exited (code {process.poll()})")
            try:
                return json.loads(line)
            except json.JSONDecodeError as error:
                raise BridgeError(f"invalid event line: {line[:200]!r}") from error

    def send(message):
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    def turn(name, messages):
        request_id = f"summary-{name}"
        send({"type": "generate", "request_id": request_id, "conversation_id": "history-summary-trial",
              "reuse_prompt_cache": True, "messages": messages, "max_tokens": MAX_TOKENS,
              "tool_rounds": TOOL_ROUNDS, "mcp_enabled": True, **SAMPLING})
        started = time.monotonic()
        record = {"turn": name, "request_messages": len(messages),
                  "request_roles": [message["role"] for message in messages], "answer": "",
                  "statuses": [], "summaries": [], "tool_calls": [], "error": None, "complete": None}
        while record["complete"] is None and record["error"] is None:
            event = read_event(EVENT_SECONDS)
            if event.get("request_id") not in (None, request_id):
                continue
            kind = event.get("type")
            if kind == "delta" and event.get("phase") == "answer":
                record["answer"] += event.get("text", "")
            elif kind == "generation_status":
                text = event.get("text") or ""
                if not record["statuses"] or record["statuses"][-1] != text:
                    record["statuses"].append(text)
            elif kind == "history_summary":
                record["summaries"].append({"kept_from": event.get("kept_from"), "prefix": event.get("prefix")})
            elif kind == "tool_start":
                record["tool_calls"].append(event.get("tool_name"))
            elif kind in ("error", "cancelled"):
                record["error"] = event.get("message") or kind
            elif kind == "complete":
                record["complete"] = {key: event.get(key) for key in (
                    "stop_reason", "context_full", "tool_rounds", "stats", "assistant_context",
                    "cache_context", "turn_context")}
        record["seconds"] = round(time.monotonic() - started, 1)
        report["turns"].append(record)
        save()
        return record

    try:
        while True:
            event = read_event(READY_SECONDS)
            if event.get("type") == "error":
                raise BridgeError(f"bridge error before ready: {event.get('message')}")
            if event.get("type") == "ready":
                report["ready"] = {key: event.get(key) for key in (
                    "model", "runtime_commit", "runtime_profile", "mlx_version", "context_limit",
                    "agent_loop_supported") if key in event}
                break
        send({"type": "set_mcp", "enabled": True, "request_id": "mcp"})
        while True:
            event = read_event(EVENT_SECONDS)
            if event.get("type") == "mcp_status" and event.get("request_id") == "mcp":
                break
        report["tools"] = [tool["function"]["name"] for tool in event.get("mcp_tool_catalog", [])]
        report["mcp_errors"] = event.get("mcp_errors")
        if not report["tools"]:
            raise BridgeError(f"no sandbox tool was offered: {report['mcp_errors']}")

        first_request = conversation()
        first = turn("1", first_request)
        criteria = report.setdefault("criteria", {})
        indicative = report.setdefault("indicative", {})
        summary = first["summaries"][-1] if first["summaries"] else None
        users = [index for index, message in enumerate(first_request) if message["role"] == "user"]
        prefix = (summary or {}).get("prefix") or ""
        kept = (summary or {}).get("kept_from")
        criteria["1_history_summary_names_a_later_user_message"] = bool(
            summary and isinstance(kept, int) and kept in users[1:]
            and prefix.startswith(MARKER) and prefix.endswith(TAIL))
        before = None
        for text in first["statuses"]:
            if text.startswith("Context summarized:") and " to " in text:
                before = int(text.split("(")[1].split(" to ")[0])
        report["prompt_tokens_before_summary"] = before
        indicative["summary_keeps_code_word"] = CODE in prefix
        indicative["summary_keeps_port"] = PORT in prefix
        indicative["turn1_answer_gives_code_word"] = CODE in first["answer"]

        second = None
        if criteria["1_history_summary_names_a_later_user_message"] and first["complete"]:
            # StudioModel.sendPrepared: the summary heads the first kept message, the
            # summarized ones are no longer sent; the answer goes back with its tool history.
            messages = [first_request[0]]
            for index, message in enumerate(first_request[kept:]):
                messages.append({**message, "content": (prefix if index == 0 else "") + message["content"]})
            done = first["complete"]
            answer = {"role": "assistant", "content": done.get("cache_context") or done.get("assistant_context") or first["answer"]}
            if done.get("turn_context"):
                answer["turn_context"] = done["turn_context"]
            messages.append(answer)
            messages.append({"role": "user", "content": "Without calling any tool: and which port does the staging service listen on?"})
            second = turn("2", messages)
            indicative["turn2_answer_gives_port"] = PORT in second["answer"]
        finished = [record for record in (first, second) if record and record["complete"] and not record["error"]]
        criteria["2_both_turns_complete_without_error"] = len(finished) == 2
        after = ((second or {}).get("complete") or {}).get("stats", {}) or {}
        report["turn2_prompt_tokens"] = after.get("prompt_tokens")
        criteria["3_turn2_prompt_is_smaller_and_fits"] = bool(
            second and before and isinstance(after.get("prompt_tokens"), int)
            and after["prompt_tokens"] < before and not (second["complete"] or {}).get("context_full")
            and not any("Context limit reached" in text for text in second["statuses"]))
        report["status"] = "complete" if len(finished) == 2 else "incomplete"
    except BridgeError as error:
        report["status"] = "failed"
        report["error"] = str(error)
    finally:
        save()
        if process.poll() is None:
            try:
                send({"type": "shutdown"})
                process.wait(timeout=30)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.kill()
                process.wait()
        stderr.close()
        shutil.rmtree(work, ignore_errors=True)
    verdict = report["status"] == "complete" and all(report.get("criteria", {"none": False}).values())
    report["criteria_met"] = verdict
    save()
    print(json.dumps({key: report.get(key) for key in (
        "status", "error", "criteria", "indicative", "prompt_tokens_before_summary", "turn2_prompt_tokens")}, indent=1))
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
