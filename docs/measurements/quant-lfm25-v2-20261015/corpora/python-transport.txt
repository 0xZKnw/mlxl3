"""Bounded JSON-line transport for native verification scripts on Linux/macOS."""

from __future__ import annotations

import argparse
import json
import math
import os
import selectors
import subprocess
import time

MAX_JSON_LINE = 16 * 1024 * 1024


def positive_seconds(value: str) -> float:
    """Parse a usable finite deadline duration.

    raises: argparse.ArgumentTypeError
    post: math.isfinite(__return__) and __return__ > 0
    """
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be positive and finite") from error
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive and finite")
    return seconds


def wait_for_pipe(selector, deadline, operation):
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not selector.select(remaining):
        raise TimeoutError(f"native process timed out {operation}")


class JsonProcess:
    """Own a child and its unbuffered pipes; every exchange has one deadline."""

    def __init__(self, command, *, stderr=None, env=None):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=env,
            bufsize=0,
        )
        self.buffer = bytearray()
        try:
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def send(self, value, deadline):
        payload = memoryview((json.dumps(value) + "\n").encode())
        descriptor = self.process.stdin.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_WRITE)
            while payload:
                wait_for_pipe(selector, deadline, "writing request")
                try:
                    written = os.write(descriptor, payload)
                except BlockingIOError:
                    continue
                except BrokenPipeError as error:
                    raise RuntimeError("native process closed stdin") from error
                payload = payload[written:]

    def receive(self, deadline):
        descriptor = self.process.stdout.fileno()
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_READ)
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("native process timed out waiting for JSON")
                end = self.buffer.find(b"\n")
                if end > MAX_JSON_LINE or (end < 0 and len(self.buffer) > MAX_JSON_LINE):
                    raise RuntimeError("native JSON response exceeds the line size limit")
                if end >= 0:
                    line = bytes(self.buffer[:end])
                    del self.buffer[: end + 1]
                    try:
                        value = json.loads(line)
                    except (ValueError, UnicodeDecodeError) as error:
                        raise RuntimeError("native process returned invalid JSON") from error
                    if not isinstance(value, dict):
                        raise RuntimeError("native JSON response must be an object")
                    return value
                wait_for_pipe(selector, deadline, "waiting for JSON")
                try:
                    chunk = os.read(descriptor, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise RuntimeError(
                        f"native process exited before a complete JSON response ({self.process.poll()})"
                    )
                self.buffer.extend(chunk)

    def close(self):
        # Raw unbuffered close cannot flush a blocked request. EOF lets idle
        # bridge/codec processes exit normally; stalled ones are terminated.
        try:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        finally:
            self.process.stdout.close()
