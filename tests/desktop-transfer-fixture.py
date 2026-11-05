#!/usr/bin/env python3
"""Bounded loopback transfer and denied-preview-network oracle; disposable only."""

import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

payload = bytes(range(251)) * 4096
requests = []
lock = threading.Lock()


def watch_parent():
    parent = os.getppid()
    while os.getppid() == parent:
        time.sleep(0.1)
    os._exit(1)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/requests":
            with lock:
                data = json.dumps(requests).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(data)
            return
        with lock:
            requests.append(self.path)
        if self.path == "/error":
            self.send_error(503)
            return
        body = payload * (2 if self.path == "/oversize" else 1)
        self.send_response(200)
        if self.path != "/unknown":
            self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.path == "/short":
            body = body[:1024]
        try:
            for offset in range(0, len(body), 32768):
                self.wfile.write(body[offset : offset + 32768])
                self.wfile.flush()
                time.sleep(0.04)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *_):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
metadata = {
    "port": server.server_port,
    "size": len(payload),
    "digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
}
Path(sys.argv[1]).write_text(json.dumps(metadata))
threading.Thread(target=watch_parent, daemon=True).start()
server.serve_forever(poll_interval=0.05)
