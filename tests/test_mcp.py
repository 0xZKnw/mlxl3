from __future__ import annotations

import json
import sys
import pytest
from unittest.mock import Mock

from mlxl3.mcp import (
    MCPManager,
    MCPServerConfig,
    add_mcp_server,
    load_mcp_servers,
    mcp_config_path,
    remove_mcp_server,
    MCPStdioClient, MCPError, MCPTool,
)


def test_tool_schema_and_protocol_limits():
    manager = MCPManager([])
    client = Mock()
    manager.clients['local'] = client
    manager.tools['search'] = MCPTool('local', 'search', 'search', '',
                                     {'type': 'object', 'required': ['query'], 'properties': {'query': {'type': 'string'}}})
    assert manager.call('search', {}).is_error
    client.call_tool.assert_not_called()
    manager.call('search', {'query': 'test'})
    client.call_tool.assert_called_once()

    stdio = MCPStdioClient(MCPServerConfig('fake'), timeout=0.001)
    stdio._send = Mock()
    stdio._messages.get = Mock(return_value={'method': 'notification'})
    with pytest.raises(MCPError, match='timed out'):
        stdio._request('test', {})
    stdio._request = Mock(side_effect=[{}, {'tools': [], 'nextCursor': 'a'}, {'tools': [], 'nextCursor': 'a'}])
    stdio._notify = Mock()
    with pytest.raises(MCPError, match='pagination'):
        stdio._initialize_and_list_tools()


def test_mcp_configuration_round_trip(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MLXL3_HOME", str(tmp_path))

    builtin = load_mcp_servers()
    assert len(builtin) == 1 and builtin[0].url == "https://mcp.exa.ai/mcp"
    added = add_mcp_server("local-tools", "python3", ["server.py"])
    assert added.name == "local-tools"
    assert load_mcp_servers() == [*builtin, added]
    assert json.loads(mcp_config_path().read_text())["mcpServers"]["local-tools"] == {
        "args": ["server.py"],
        "command": "python3",
        "enabled": True,
    }

    remove_mcp_server("local-tools")
    assert load_mcp_servers() == builtin
    remove_mcp_server("exa")
    assert load_mcp_servers()[0].enabled is False


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
