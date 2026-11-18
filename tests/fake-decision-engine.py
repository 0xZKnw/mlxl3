#!/usr/bin/env python3
"""Disposable stand-in for `mlxl3 decide-bridge` / `decide-serve`; never loads a model."""

import json
import os
import socket
import sys
import time

log = os.environ.get("MLXL3_FAKE_DECISION_LOG")
if log:
    with open(log, "a") as handle:
        handle.write(json.dumps(sys.argv[1:]) + "\n")
command, model = sys.argv[1:3]


def emit(value):
    print(json.dumps(value), flush=True)


if command == "decide-serve":
    port = int(sys.argv[sys.argv.index("--port") + 1])
    key = sys.stdin.read().strip()
    if not key:
        sys.exit(4)
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen()
    emit({"type": "ready", "model": model, "port": server.getsockname()[1], "decision": True})
    while True:
        time.sleep(0.2)
if command != "decide-bridge":
    sys.exit(2)
if model.endswith("broken"):
    print("checkpoint has no safetensors weights", file=sys.stderr, flush=True)
    sys.exit(3)
audio = "omni" in model.lower()
emit({"type": "ready", "model": model, "decision": True, "images": True, "audio": audio, "set_size": 16})
for line in sys.stdin:
    message = json.loads(line)
    request = message["request"]
    names = list(request["questions"])
    if "crash" in names:
        sys.exit(9)
    if "fail" in names:
        emit({"type": "error", "id": message["id"], "message": "question 'fail': unsupported"})
        continue
    answers = {}
    for name, q in request["questions"].items():
        if q["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.875}
        elif q["type"] == "choice" and q.get("multiple"):
            answers[name] = {"type": "choice", "choices": [list(q["criteria"])[0]],
                             "probabilities": {k: 0.9 if i == 0 else 0.1 for i, k in enumerate(q["criteria"])}}
        elif q["type"] == "choice":
            first = list(q["criteria"])
            answers[name] = {"type": "choice", "choice": first[0], "confidence": 0.75,
                             "probabilities": {k: (0.75 if i == 0 else 0.25 / (len(first) - 1)) for i, k in enumerate(first)},
                             "order_seen": first}
        elif q["type"] == "score":
            answers[name] = {"type": "score", "score": 1.0, "confidence": 0.6,
                             "probabilities": {str(i): 1 / len(q["criteria"]) for i in range(len(q["criteria"]))},
                             "legend": {str(i): t for i, t in enumerate(q["criteria"])}}
        else:
            answers[name] = {"type": q["type"], q["type"]: None, "span": None, "found": 0.1}
    tokens = 24 + sum(len(json.dumps(q)) // 8 for q in request["questions"].values())
    emit({"type": "result", "id": message["id"], "elapsed_ms": 12.5,
          "result": {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0},
                     "media_seen": request.get("media", [])}})
