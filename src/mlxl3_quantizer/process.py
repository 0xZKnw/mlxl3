"""Bounded, cancellable native converter process ownership."""

import os
import subprocess
import time


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
