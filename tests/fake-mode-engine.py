#!/usr/bin/env python3
"""Disposable process-boundary fixture for Desktop's four modes; never loads MLX."""

import json
import os
import signal
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

registry = None


def watch_parent():
    parent = os.getppid()
    while os.getppid() == parent:
        time.sleep(0.1)
    os._exit(1)


threading.Thread(target=watch_parent, daemon=True).start()
if sys.argv[1:2] == ["--registry"]:
    registry = sys.argv[2]
    del sys.argv[1:3]


def emit(kind, **fields):
    print(json.dumps({"type": kind, **fields}), flush=True)


command, model = sys.argv[1:3]
if command == "serve":
    if model == "serve-fail":
        print("fixture bind failure", file=sys.stderr, flush=True)
        sys.exit(3)
    if model == "serve-silent":
        time.sleep(60)
        sys.exit(0)
    port = int(sys.argv[sys.argv.index("--port") + 1])
    assert "--api-key" not in sys.argv and "--api-key-stdin" in sys.argv
    api_key = sys.stdin.read(513)
    assert len(api_key) == 36 and api_key not in sys.argv
    host = sys.argv[sys.argv.index("--host") + 1]
    context = int(sys.argv[sys.argv.index("--context-length") + 1])

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != f"Bearer {api_key}":
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[]}')

        def log_message(self, *_):
            pass

    class FixtureServer(HTTPServer):
        def server_bind(self):
            # HTTPServer.server_bind resolves socket.getfqdn(host); for 0.0.0.0 that is a
            # reverse lookup of the machine name, which outlasts the 8 s readiness wait
            # on hosted macOS runners. The fixture never uses server_name.
            socketserver.TCPServer.server_bind(self)
            self.server_name, self.server_port = self.server_address[:2]

    server = FixtureServer((host, port), Handler)
    emit("ready", port=port, model=model, host=host, context=context)
    server.serve_forever()
elif command == "bridge":
    if model.startswith("compare-test-"):
        # A live predecessor holds the same local lock until the process exits.
        import fcntl

        lease = open(os.path.join(os.environ["MLXL3_HOME"], "compare.lock"), "a")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active = None
    ignore_stop = False
    # What an engine reports with `cancelled` / `error` once tools ran in the stopped turn.
    interrupted = None
    INTERRUPTED_HISTORY = json.dumps([
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "write", "arguments": {"file_path": "game.py"}}}]},
        {"role": "tool", "name": "write", "content": "Wrote game.py (120 lines)"},
        {"role": "assistant", "content": ""},
    ])

    def cancel(*_):
        global active
        if active and not ignore_stop:
            if interrupted:
                emit("cancelled", request_id=active, turn_context=interrupted)
            else:
                emit("cancelled", request_id=active)
            active = None

    can_yield = os.path.isfile(os.path.join(os.environ["MLXL3_HOME"], "cooperative-agent-guidance"))
    yield_requested = False

    def request_yield(*_):
        global yield_requested
        yield_requested = True

    if can_yield:
        signal.signal(signal.SIGUSR2, request_yield)
    signal.signal(signal.SIGUSR1, cancel)
    with open(os.path.join(os.environ["MLXL3_HOME"], "mode-launches.jsonl"), "a") as log:
        log.write(json.dumps({"registry": registry, "model": model, "pid": os.getpid()}) + "\n")
    emit(
        "ready",
        model=model,
        modules=1,
        resident_gb=0.01,
        context_limit=2048,
        model_context_limit=2048,
        workspace_features=True,
        agent_yield_supported=can_yield,
        bridge_protocol=1,
        runtime_commit="fixture",
        runtime_profile="release",
        mlx_version="0.32.2",
        reasoning_effort_supported=True,
        reasoning_effort_levels=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
        # AGENT-2026-11-23-BOUCLE: a newer engine when the test leaves this marker file
        agent_loop_supported=os.path.exists(os.path.join(os.environ["MLXL3_HOME"], "agent-loop-engine")),
        # AGENT-2026-11-26-LIVE-CACHE: announced with the same marker (a newer engine)
        live_prompt_cache_supported=os.path.exists(os.path.join(os.environ["MLXL3_HOME"], "agent-loop-engine")),
    )
    if model.startswith("compare-test-no-reader-"):
        time.sleep(60)
    for line in sys.stdin:
        request = json.loads(line)
        if request["type"] == "set_mcp":
            # Exercise the real asynchronous handshake gap before Agent regeneration.
            time.sleep(0.2)
            emit(
                "mcp_status",
                request_id=request.get("request_id"),
                mcp_servers=1 if request["enabled"] else 0,
                mcp_tools=1 if request["enabled"] else 0,
                mcp_errors={},
            )
        elif request["type"] == "generate":
            yield_requested = False
            active = request["request_id"]
            text = request["messages"][-1]["content"]
            if model.startswith("compare-test-"):
                with open(os.path.join(os.environ["MLXL3_HOME"], "compare-requests.jsonl"), "a") as log:
                    log.write(json.dumps({"request": request, "model": model, "registry": registry}) + "\n")
                if text == "comparison-unlimited":
                    if request["max_tokens"] != -1:
                        emit("error", request_id=active, message="Expected an uncapped Lab request")
                    else:
                        context_full = model == "compare-test-b"
                        answer = "<p>HTML fixture</p>\n" * (600 if context_full else 1) + "<footer>Done</footer>"
                        emit("delta", request_id=active, phase="answer", text=answer)
                        emit("complete", request_id=active, context_full=context_full,
                             stop_reason="context_full" if context_full else "eos",
                             stats={"ttft_seconds": 0.1, "prefill_tps": 120, "decode_tps": 100,
                                    "prompt_tokens": 12, "generated_tokens": 5000 if context_full else 690})
                    active = None
                    continue
                if text == "comparison-visible-answer":
                    controlled = (
                        request.get("reasoning_effort") == "none"
                        if model == "compare-test-a"
                        else request.get("reasoning_budget") == 256
                    )
                    if controlled:
                        emit("delta", request_id=active, phase="answer", text="Visible answer")
                        emit("complete", request_id=active, stop_reason="eos")
                    else:
                        emit("delta", request_id=active, phase="thinking", text="Still thinking")
                        emit("complete", request_id=active, stop_reason="max_tokens",
                             stats={"generated_tokens": request["max_tokens"], "decode_tps": 48.5})
                    active = None
                    continue
                if text == "comparison-errors" and model == "compare-test-a":
                    emit("delta", request_id="unrelated", phase="answer", text="wrong request")
                    emit("delta", request_id=active, phase="thinking", text="internal thought")
                    emit("delta", request_id=active, phase="answer", text="   \n")
                    emit(
                        "complete",
                        request_id=active,
                        assistant_context="",
                        cache_context="<think>private</think>",
                        stop_reason="max_tokens",
                        stats={"ttft_seconds": 0.1, "prefill_tps": 120, "decode_tps": 100,
                               "prompt_tokens": 12, "generated_tokens": 1},
                    )
                    continue
                if text == "comparison-errors" and model == "compare-test-b":
                    emit("complete", request_id=active, assistant_context="B's fallback answer", stop_reason="eos")
                    continue
                if text == "comparison-empty":
                    emit("complete", request_id=active, stop_reason="eos")
                    continue
                if text == "comparison-context":
                    answer = "Text before the context filled"
                    if model == "compare-test-a":
                        emit("delta", request_id=active, phase="answer", text=answer)
                    emit("complete", request_id=active, assistant_context=answer,
                         stop_reason="context_full", context_full=True)
                    continue
                if text == "comparison-context-empty":
                    emit("complete", request_id=active, assistant_context="",
                         stop_reason="context_full", context_full=True)
                    continue
                if text == "comparison-malformed":
                    print("invalid fixture frame", flush=True)
                    continue
            if text == "steer the second action":
                key = {"request_id": active, "tool_call_id": "first", "tool_name": "task"}
                emit("tool_start", **key)
                # The test deliberately keeps the first operation in flight after queueing.
                gate = os.path.join(os.environ["MLXL3_HOME"], "guidance-operation-finished")
                deadline = time.monotonic() + 6
                while not os.path.isfile(gate) or (can_yield and not yield_requested):
                    if time.monotonic() >= deadline:
                        emit("error", request_id=active, message="Guidance handshake timed out")
                        break
                    time.sleep(0.01)
                else:
                    emit("tool_result", **key, text="modified first file", is_error=False)
                    history = [{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "task", "arguments": {}}}]},
                               {"role": "tool", "name": "task", "content": "modified first file"}, {"role": "assistant", "content": ""}]
                    emit("complete", request_id=active, stop_reason="agent_yield" if yield_requested else "eos",
                         turn_context=json.dumps(history), assistant_context="Yielded" if yield_requested else "Completed")
                active = None
                continue
            interrupted = None
            if text in ("summarize-history", "summarize-invalid"):
                # An engine that summarized the oldest turns: the second user message of the
                # request is the first one kept. The invalid ones must all be ignored.
                users = [index for index, message in enumerate(request["messages"]) if message["role"] == "user"]
                prefix = ("[Summary of the earlier conversation, written by you to free context:]\n- fixture summary\n"
                          "[End of the summary. The conversation continues with the user's message below.]\n\n")
                if text == "summarize-invalid":
                    emit("history_summary", request_id=active, kept_from=999, prefix=prefix)
                    emit("history_summary", request_id=active, kept_from=users[1] + 1, prefix=prefix)
                    emit("history_summary", request_id=active, kept_from=users[1])
                    emit("history_summary", request_id=active, kept_from=users[1], prefix="")
                    emit("history_summary", request_id="stale", kept_from=users[1], prefix=prefix)
                    emit("history_summary", request_id=active, kept_from=0, prefix=prefix)
                else:
                    emit("history_summary", request_id=active, kept_from=users[1], prefix=prefix)
            if text in ("tools-then-hang", "tools-then-error"):
                key = {"request_id": active, "tool_call_id": "w1", "tool_name": "write"}
                emit("delta", request_id=active, phase="answer", text="Writing the game.")
                emit("tool_start", **key)
                emit("tool_result", **key, text="Wrote game.py (120 lines)", is_error=False)
                if text == "tools-then-error":
                    emit("error", request_id=active, message="Metal device lost",
                         turn_context=INTERRUPTED_HISTORY)
                    active = None
                else:
                    interrupted = INTERRUPTED_HISTORY
                    emit("delta", request_id=active, phase="thinking", text="Now the tests")
                continue
            if text.startswith("hang-"):
                ignore_stop = text == "hang-ignore"
                if ignore_stop:
                    signal.signal(signal.SIGTERM, signal.SIG_IGN)
                emit("delta", request_id=active, phase="thinking", text="Preparing a file")
                emit(
                    "generation_status",
                    request_id=active,
                    phase="tool_preparation",
                    text="Preparing tool call · 32 tokens",
                )
                if ignore_stop:
                    time.sleep(60)
                continue
            emit(
                "delta", request_id=request["request_id"], phase="answer", text=json.dumps(request)
            )
            emit(
                "complete",
                request_id=request["request_id"],
                stats={
                    "ttft_seconds": 0.125,
                    "prefill_tps": 120,
                    "decode_tps": 30,
                    "prompt_tokens": 12,
                    "generated_tokens": 6,
                    "context_used": 18,
                    "context_limit": 2048,
                },
                stop_reason="max_tokens" if text == "budget-limit" else "eos",
            )
            active = None
else:
    sys.exit(2)
