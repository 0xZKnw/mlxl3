"""Bounded driver of the real bridge for the AGENT-2026-11-26-LIVE-CACHE trials."""
import json
import os
import selectors
import subprocess
import time


class BridgeError(RuntimeError):
    pass


class Bridge:
    """One engine process. Every wait is bounded; `close` always ends the process."""

    def __init__(self, engine, registry, model, stderr_path, environment=None, context="1048576"):
        self.stderr = open(stderr_path, "w")
        self.process = subprocess.Popen(
            [engine, "--registry", str(registry), "bridge", model, "--context-length", context],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            env={**os.environ, **(environment or {})})
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.pending = bytearray()

    def event(self, seconds):
        """The next event. Bytes are read from the pipe itself and split here: a line waiting
        in a reader's buffer would never wake the selector."""
        deadline = time.monotonic() + seconds
        while b"\n" not in self.pending:
            left = deadline - time.monotonic()
            if left <= 0 or not self.selector.select(timeout=left):
                raise BridgeError(f"no event within {seconds} s")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise BridgeError(f"bridge exited ({self.process.poll()})")
            self.pending.extend(data)
        line, _, rest = bytes(self.pending).partition(b"\n")
        self.pending[:] = rest
        try:
            return json.loads(line)
        except json.JSONDecodeError as error:
            raise BridgeError(f"invalid bridge line: {line[:200]!r}") from error

    def send(self, message):
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        self.process.stdin.flush()

    def ready(self, seconds=900):
        while (item := self.event(seconds)).get("type") != "ready":
            if item.get("type") == "error":
                raise BridgeError(item.get("message"))
        return item

    def generate(self, request, seconds=900):
        """Sends `request` and returns (complete event, tool names, tool results, statuses)."""
        self.send(request)
        tools, results, statuses = [], [], []
        while True:
            item = self.event(seconds)
            if item.get("request_id") not in (None, request["request_id"]):
                continue
            kind = item.get("type")
            if kind in ("error", "cancelled"):
                raise BridgeError(f"{request['request_id']}: {item.get('message', kind)}")
            if kind == "tool_start":
                tools.append(item.get("tool_name"))
            elif kind == "tool_result":
                results.append({"tool": item.get("tool_name"), "error": item.get("is_error"),
                                "text": (item.get("text") or "")[:300]})
            elif kind == "generation_status":
                text = item.get("text") or ""
                if not statuses or statuses[-1] != text:
                    statuses.append(text)
            elif kind == "complete":
                return item, tools, results, statuses

    def close(self):
        if self.process.poll() is None:
            try:
                self.send({"type": "shutdown"})
                self.process.wait(timeout=60)
            except Exception:
                self.process.kill()
                self.process.wait()
        self.stderr.close()
        return self.process.returncode
