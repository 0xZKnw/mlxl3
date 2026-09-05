from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mlxl3 import mcp


@pytest.fixture(params=[False, True], ids=["json", "sse"])
def endpoint(request):
    events = []
    sse = request.param

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = message["method"]
            events.append((method, dict(self.headers), message))
            if "id" not in message:
                self.send_response(202)
                self.end_headers()
                return
            if method == "initialize":
                result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}}
            elif method == "tools/list":
                second = "cursor" in message["params"]
                result = {"tools": [{"name": "second" if second else "echo", "inputSchema": {}}]}
                if not second:
                    result["nextCursor"] = "page2"
            else:
                result = {"content": [{"type": "text", "text": "bonjour"}]}
            payload = {"jsonrpc": "2.0", "id": message["id"], "result": result}
            if method == "tools/call" and message["params"].get("name") == "error":
                payload = {"jsonrpc": "2.0", "id": message["id"],
                           "error": {"code": -1, "message": "expected failure"}}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream" if sse else "application/json")
            if method == "initialize":
                self.send_header("MCP-Session-Id", "session-test")
            self.end_headers()
            if sse:
                self.wfile.write(b': heartbeat\n\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n')
                self.wfile.write(("event: message\ndata: " + json.dumps(payload) + "\n\n").encode())
            else:
                self.wfile.write(json.dumps(payload).encode())

        def do_DELETE(self):
            events.append(("delete", dict(self.headers), {}))
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/mcp", events
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_http_handshake_pagination_tool_and_close(endpoint):
    url, events = endpoint
    client = mcp.MCPHTTPClient(mcp.MCPServerConfig(name="test", url=url))
    try:
        assert [t["name"] for t in client.connect()] == ["echo", "second"]
        result = client.call_tool("echo", {})
        assert result.text == "bonjour" and not result.is_error
        with pytest.raises(mcp.MCPError, match="expected failure"):
            client.call_tool("error", {})
    finally:
        client.close()
    assert events[-1][0] == "delete"
    for method, headers, _ in events[1:]:
        headers = {k.lower(): v for k, v in headers.items()}
        assert headers["mcp-session-id"] == "session-test"
        assert headers["mcp-protocol-version"] == "2025-03-26"


def test_master_off_does_not_read_config_or_connect(monkeypatch):
    def forbidden():
        pytest.fail("MCP disabled must not access configuration or connect")
    monkeypatch.setattr(mcp, "load_mcp_servers", forbidden)
    manager = mcp.MCPManager([])
    manager.set_enabled(False)
    assert manager.chat_tools == []
    assert manager.clients == {}
    with pytest.raises(mcp.MCPError):
        manager.set_enabled("false")


def test_master_toggle_connects_and_closes_without_duplicate_connections(endpoint, monkeypatch):
    url, events = endpoint
    monkeypatch.setattr(mcp, "load_mcp_servers", lambda: [mcp.MCPServerConfig(name="test", url=url)])
    manager = mcp.MCPManager([])
    try:
        manager.set_enabled(True)
        assert manager.connected_server_count == 1
        manager.set_enabled(True)
        assert sum(e[0] == "initialize" for e in events) == 1
        manager.set_enabled(False)
        assert not manager.enabled and manager.chat_tools == []
        assert events[-1][0] == "delete"
        manager.set_enabled(True)
        assert sum(e[0] == "initialize" for e in events) == 2
    finally:
        manager.close()


def test_preset_override_and_https_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("MLXL3_HOME", str(tmp_path))
    config = mcp.ensure_mcp_config()
    config.write_text(json.dumps({"mcpServers": {"exa": {"url": "https://example.org/mcp", "enabled": False}}}))
    servers = mcp.load_mcp_servers()
    assert len(servers) == 1 and not servers[0].enabled
    assert servers[0].url == "https://example.org/mcp"
    config.write_text(json.dumps({"mcpServers": {"exa": {"url": "http://example.org/mcp"}}}))
    with pytest.raises(mcp.MCPError, match="HTTPS"):
        mcp.load_mcp_servers()


def test_configuration_failure_is_optional(monkeypatch):
    def broken():
        raise ValueError("bad config")
    monkeypatch.setattr(mcp, "load_mcp_servers", broken)
    manager = mcp.MCPManager([])
    manager.set_enabled(True)
    assert not manager.chat_tools and manager.errors == {"configuration": "bad config"}
    manager.set_enabled(False)
    assert manager.errors == {}
