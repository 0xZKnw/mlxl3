"""End-to-end tests of the bridge process through its real entry point (fake backend)."""

import json
import os
import queue
import subprocess
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 20


class Bridge:
    def __init__(self, test, env=None, args=()):
        environment = {**os.environ, "PYTHONPATH": str(ROOT), **(env or {})}
        self.process = subprocess.Popen(
            [sys.executable, "-m", "mlxl3_exl3", "bridge", "unused-model", "--backend", "fake", *args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
        test.addCleanup(self.close)
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for raw in self.process.stdout:
            self.lines.put(raw)
        self.lines.put(None)

    def send(self, payload):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.process.stdin.write(data + b"\n")
        self.process.stdin.flush()

    def event(self):
        raw = self.lines.get(timeout=TIMEOUT)
        if raw is None:
            raise AssertionError("bridge closed its protocol stream")
        return json.loads(raw)  # every protocol line must be JSON

    def until(self, kind):
        seen = []
        while True:
            event = self.event()
            seen.append(event)
            if event["type"] == kind:
                return seen

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=TIMEOUT)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()


def generate(request_id, text, **extra):
    return {"type": "generate", "request_id": request_id,
            "messages": [{"role": "user", "content": text}], **extra}


class BridgeTests(unittest.TestCase):
    def test_ready_generate_done(self):
        bridge = Bridge(self, args=("--context-length", "300"))
        self.assertEqual(bridge.event()["type"], "loading")
        ready = bridge.event()
        self.assertEqual((ready["type"], ready["bridge_protocol"], ready["context_limit"]), ("ready", 1, 512))
        bridge.send(generate("r1", "un deux trois"))
        events = bridge.until("done")
        self.assertEqual(events[0], {"type": "generation_status", "request_id": "r1", "phase": "prefill"})
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "delta"), "un deux trois")
        self.assertEqual(events[-1]["generated_tokens"], 3)
        self.assertTrue(all(e["request_id"] == "r1" for e in events))
        bridge.send({"type": "shutdown"})
        self.assertEqual(bridge.process.wait(timeout=TIMEOUT), 0)

    def test_errors_do_not_stop_the_bridge(self):
        bridge = Bridge(self)
        bridge.until("ready")
        bridge.send(b"{not json")
        self.assertEqual(bridge.event()["type"], "error")
        bridge.send(b"\xff\xfe")
        self.assertEqual(bridge.event()["type"], "error")
        bridge.send(generate("bad", "x", temperature=-1))
        self.assertEqual(bridge.event(), {"type": "error", "request_id": "bad",
                                          "message": "temperature must be between 0 and 5"})
        bridge.send(generate("boom", "__raise__"))
        error = bridge.until("error")[-1]
        self.assertEqual(error["request_id"], "boom")
        self.assertIn("fake backend failure", error["message"])
        bridge.send({"type": "ping", "request_id": "p"})
        self.assertEqual(bridge.event(), {"type": "pong", "request_id": "p"})

    def test_cancel_running_generation(self):
        bridge = Bridge(self, env={"MLXL3_FAKE_DELAY": "0.05"})
        bridge.until("ready")
        bridge.send(generate("long", " ".join(["mot"] * 400)))
        events = bridge.until("delta")
        bridge.send({"type": "cancel", "request_id": "long"})
        events += bridge.until("cancelled")
        self.assertLess(sum(e["type"] == "delta" for e in events), 400)
        bridge.send(generate("next", "encore"))  # the bridge stays usable
        self.assertEqual(bridge.until("done")[-1]["generated_tokens"], 1)

    def test_idle_untargeted_cancel_does_not_cancel_next(self):
        bridge = Bridge(self)
        bridge.until("ready")
        bridge.send({"type": "cancel"})
        bridge.send(generate("after", "ok"))
        self.assertEqual(bridge.until("done")[-1]["request_id"], "after")

    def test_targeted_cancel_before_start(self):
        bridge = Bridge(self, env={"MLXL3_FAKE_DELAY": "0.05"})
        bridge.until("ready")
        bridge.send(generate("first", " ".join(["a"] * 20)))
        bridge.send({"type": "cancel", "request_id": "second"})
        bridge.send(generate("second", "b c"))
        bridge.until("done")
        events = bridge.until("cancelled")
        self.assertEqual(events[-1]["request_id"], "second")
        self.assertFalse(any(e["type"] == "delta" for e in events if e["request_id"] == "second"))

    def test_stdin_eof_shuts_down_during_generation(self):
        bridge = Bridge(self, env={"MLXL3_FAKE_DELAY": "0.05"})
        bridge.until("ready")
        bridge.send(generate("r", " ".join(["x"] * 400)))
        bridge.until("delta")
        bridge.process.stdin.close()
        self.assertEqual(bridge.process.wait(timeout=TIMEOUT), 0)

    def test_library_stdout_noise_goes_to_stderr(self):
        bridge = Bridge(self, env={"MLXL3_FAKE_NOISE": "1"})
        bridge.until("ready")  # Bridge.event() fails on any non-JSON protocol line
        bridge.send({"type": "shutdown"})
        self.assertEqual(bridge.process.wait(timeout=TIMEOUT), 0)
        self.assertIn(b"native stdout noise", bridge.process.stderr.read())

    def test_load_failure_is_fatal_event(self):
        bridge = Bridge(self, env={"MLXL3_FAKE_LOAD_ERROR": "no GPU"})
        self.assertEqual(bridge.event()["type"], "loading")
        error = bridge.event()
        self.assertEqual((error["type"], error["fatal"]), ("error", True))
        self.assertIn("no GPU", error["message"])
        self.assertEqual(bridge.process.wait(timeout=TIMEOUT), 1)

    def test_context_too_large_is_fatal(self):
        bridge = Bridge(self, args=("--context-length", "999999"))
        bridge.until("loading")
        self.assertTrue(bridge.event()["fatal"])


if __name__ == "__main__":
    unittest.main()
