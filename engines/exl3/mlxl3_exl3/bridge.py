"""Resident JSON-lines bridge (protocol v1) for the Windows app.

Requests (stdin): ping, generate, cancel, shutdown.
Events (protocol stream): loading, ready, pong, generation_status, delta,
done, cancelled, error.

Cancellation is a ``cancel`` request rather than a POSIX signal so that it
works the same on Windows and through ``wsl.exe``. A reader thread handles
``cancel`` and ``shutdown`` immediately while the main thread generates;
closing stdin is treated as ``shutdown``.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import IO, Callable

from . import BRIDGE_PROTOCOL, __version__
from .protocol import Emitter, RequestError, parse_request

_EOF = object()


class Control:
    def __init__(self):
        self.cancel = threading.Event()  # untargeted cancel: stops the running generation
        self.shutdown = threading.Event()
        self.requests: "queue.Queue[object]" = queue.Queue()
        self._lock = threading.Lock()
        self._cancelled_ids: set[str] = set()

    def cancel_request(self, request_id: str) -> None:
        if request_id:
            with self._lock:
                if len(self._cancelled_ids) >= 1024:  # stale ids of requests never sent
                    self._cancelled_ids.clear()
                self._cancelled_ids.add(request_id)
        else:
            self.cancel.set()

    def should_stop(self, request_id: str) -> bool:
        if self.cancel.is_set() or self.shutdown.is_set():
            return True
        with self._lock:
            return request_id in self._cancelled_ids

    def finish(self, request_id: str) -> None:
        with self._lock:
            self._cancelled_ids.discard(request_id)


def _reader(stream: IO[bytes], control: Control, emitter: Emitter) -> None:
    try:
        for raw in stream:
            if not raw.strip():
                continue
            try:
                request = parse_request(raw.rstrip(b"\r\n"))
            except RequestError as error:
                emitter.emit({"type": "error", "request_id": error.request_id, "message": str(error)})
                continue
            if request["type"] == "cancel":
                control.cancel_request(request["request_id"])
            elif request["type"] == "shutdown":
                control.shutdown.set()
                control.cancel.set()
                break
            else:
                control.requests.put(request)
    except (OSError, ValueError) as error:
        emitter.emit({"type": "error", "request_id": "", "message": f"stdin failed: {error}"})
    finally:
        control.shutdown.set()
        control.cancel.set()
        control.requests.put(_EOF)


def _protocol_stream() -> IO[str]:
    """Duplicate stdout for the protocol, then point fd 1 and sys.stdout at stderr.

    Library prints and native writes to fd 1 can no longer corrupt the protocol.
    """
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return os.fdopen(protocol_fd, "w", encoding="utf-8", newline="\n", buffering=1)


def run(model_dir: str, context_length: int, backend_factory: Callable,
        stdin: IO[bytes] | None = None, out: IO[str] | None = None) -> int:
    emitter = Emitter(out if out is not None else _protocol_stream())
    control = Control()
    emitter.emit({"type": "loading", "model": model_dir})
    started = time.perf_counter()
    try:
        backend = backend_factory(model_dir, context_length)
    except Exception as error:  # every load failure must reach the app as an event
        emitter.emit({"type": "error", "request_id": "", "fatal": True,
                      "message": f"{type(error).__name__}: {error}"})
        return 1
    info = backend.info()
    emitter.emit({"type": "ready", "model": model_dir, "bridge_protocol": BRIDGE_PROTOCOL,
                  "runtime_version": __version__, "load_seconds": time.perf_counter() - started,
                  **info})
    reader = threading.Thread(target=_reader, args=(stdin or sys.stdin.buffer, control, emitter),
                              name="bridge-stdin", daemon=True)
    reader.start()
    try:
        while True:
            request = control.requests.get()
            if request is _EOF or control.shutdown.is_set():
                break
            if request["type"] == "ping":
                emitter.emit({"type": "pong", "request_id": request["request_id"]})
                continue
            _generate(backend, request, control, emitter)
    finally:
        try:
            backend.close()
        except Exception as error:
            print(f"backend close failed: {error}", file=sys.stderr)
    return 0


def _generate(backend, request: dict, control: Control, emitter: Emitter) -> None:
    request_id = request["request_id"]
    # An untargeted cancel that arrived while idle targets nothing. A cancel
    # naming this request_id stops it even if it arrived before it started.
    control.cancel.clear()
    try:
        _run_generation(backend, request, control, emitter)
    finally:
        control.finish(request_id)


def _run_generation(backend, request: dict, control: Control, emitter: Emitter) -> None:
    request_id = request["request_id"]
    emitter.emit({"type": "generation_status", "request_id": request_id, "phase": "prefill"})
    stats = None
    try:
        for item in backend.generate(request, lambda: control.should_stop(request_id)):
            if "text" in item:
                emitter.emit({"type": "delta", "request_id": request_id, "text": item["text"]})
            elif item.get("cancelled"):
                emitter.emit({"type": "cancelled", "request_id": request_id})
                return
            elif "stats" in item:
                stats = item["stats"]
    except Exception as error:
        emitter.emit({"type": "error", "request_id": request_id,
                      "message": f"{type(error).__name__}: {error}"})
        return
    if stats is None:
        emitter.emit({"type": "error", "request_id": request_id,
                      "message": "the backend finished without reporting statistics"})
        return
    decode = stats.get("decode_seconds") or 0.0
    generated = stats.get("generated_tokens", 0)
    emitter.emit({"type": "done", "request_id": request_id, **stats,
                  "decode_tokens_per_second": (generated / decode) if decode > 0 else None})
