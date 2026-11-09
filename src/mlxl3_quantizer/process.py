"""Bounded, cancellable native converter process ownership."""

import os
import selectors
import subprocess
import time


class TileWorker:
    """Single-flight owned worker; pipes and response memory have fixed bounds."""

    def __init__(self, command):
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeError(f"native tile worker launch failed: {exc}") from exc
        self.selector = selectors.DefaultSelector()
        try:
            for stream in (self.process.stdout, self.process.stderr):
                os.set_blocking(stream.fileno(), False)
                self.selector.register(stream, selectors.EVENT_READ)
            os.set_blocking(self.process.stdin.fileno(), False)
        except BaseException:
            self.close()
            raise
        self.request_id = 0

    def request(self, *, timeout, cancel_check):
        deadline = time.monotonic() + timeout
        self.request_id += 1
        if self.request_id >= 1 << 64:
            raise ValueError("native worker request ID exhausted")
        pending = memoryview(f"{self.request_id}\n".encode())
        response, errors = bytearray(), bytearray()
        while True:
            if cancel_check is not None:
                cancel_check()
            if time.monotonic() >= deadline:
                raise TimeoutError("native tile worker timed out")
            if pending:
                try:
                    pending = pending[os.write(self.process.stdin.fileno(), pending) :]
                except BlockingIOError:
                    pass
            for key, _ in self.selector.select(0.01):
                data = os.read(key.fileobj.fileno(), 4096)
                if not data:
                    raise RuntimeError("native tile worker closed its response stream")
                target = response if key.fileobj is self.process.stdout else errors
                target.extend(data)
                if len(target) > 65536:
                    raise ValueError("oversized native tile worker output")
            if b"\n" in response:
                if not response.endswith(b"\n") or response.count(b"\n") != 1:
                    raise ValueError("unexpected native tile worker response framing")
                return bytes(response)
            if self.process.poll() is not None:
                raise RuntimeError("native tile worker exited: " + errors.decode(errors="replace"))

    def close(self):
        # Kill even on successful completion: no unbounded EOF/drain handshake.
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
        self.selector.close()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run_native(command, *, root, timeout, cancel_check, label):
    """Read a bounded native response and reap every child before returning."""
    with (root / "stdout").open("w+b") as stdout, (root / "stderr").open("w+b") as stderr:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        try:
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                if (
                    max(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size)
                    > 65536
                ):
                    raise ValueError("oversized native conversion output")
                if cancel_check is not None:
                    cancel_check()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"native {label} timed out")
                time.sleep(0.01)
            if max(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size) > 65536:
                raise ValueError("oversized native conversion output")
            stdout.seek(0)
            response = stdout.read(65537)
            stderr.seek(0)
            error = stderr.read(4096).decode("utf-8", errors="replace")
            if process.returncode != 0:
                raise RuntimeError(f"native {label} failed: {error}")
            if len(response) > 65536:
                raise ValueError("oversized native conversion response")
            if cancel_check is not None:
                cancel_check()
            return response
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
