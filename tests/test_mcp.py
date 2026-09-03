from __future__ import annotations

import json
import sys

from mlxl3.mcp import (
    MCPManager,
    MCPServerConfig,
    add_mcp_server,
    load_mcp_servers,
    mcp_config_path,
    remove_mcp_server,
)


def test_mcp_configuration_round_trip(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MLXL3_HOME", str(tmp_path))

    assert load_mcp_servers() == []
    added = add_mcp_server("local-tools", "python3", ["server.py"])
    assert added.name == "local-tools"
    assert load_mcp_servers() == [added]
    assert json.loads(mcp_config_path().read_text())["mcpServers"]["local-tools"] == {
        "args": ["server.py"],
        "command": "python3",
        "enabled": True,
    }

    remove_mcp_server("local-tools")
    assert load_mcp_servers() == []


def test_stdio_manager_lists_and_calls_tools(tmp_path) -> None:
    server = tmp_path / "fake_mcp.py"
    server.write_text(
        """
import json
import sys

for line in sys.stdin:
    message = json.loads(line)
    if "id" not in message:
        continue
    method = message["method"]
    if method == "initialize":
        result = {
            "protocolVersion": "2025-11-25",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1"},
        }
    elif method == "tools/list":
        result = {"tools": [{
            "name": "echo",
            "description": "Echo a value",
            "inputSchema": {"type": "object", "properties": {"value": {"type": "string"}}},
        }]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": message["params"]["arguments"]["value"]}]}
    else:
        result = {}
    print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}), flush=True)
""".lstrip(),
        encoding="utf-8",
    )
    manager = MCPManager(
        [MCPServerConfig(name="fake", command=sys.executable, args=(str(server),))]
    )
    try:
        manager.connect()
        assert list(manager.tools) == ["fake.echo"]
        assert manager.chat_tools[0]["function"]["name"] == "fake.echo"
        result = manager.call("fake.echo", {"value": "bonjour"})
        assert result.text == "bonjour"
        assert result.is_error is False
    finally:
        manager.close()
