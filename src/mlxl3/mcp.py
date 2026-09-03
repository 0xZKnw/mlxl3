"""Small, dependency-free MCP stdio client used by MLXL3 Desktop."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from mlxl3.registry import registry_path

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_PROTOCOL_VERSION = "2025-11-25"


class MCPError(RuntimeError):
    """An MCP server or configuration failed."""


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class MCPTool:
    server: str
    name: str
    public_name: str
    description: str
    input_schema: dict[str, Any]

    def chat_template_payload(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.public_name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(frozen=True)
class MCPToolResult:
    text: str
    is_error: bool


def mcp_config_path() -> Path:
    return registry_path().with_name("mcp.json")


def ensure_mcp_config() -> Path:
    path = mcp_config_path()
    if not path.exists():
        _write_config({"version": 1, "mcpServers": {}})
    return path


def _read_config() -> dict[str, Any]:
    path = ensure_mcp_config()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MCPError(f"invalid MCP configuration {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("mcpServers"), dict):
        raise MCPError(f"MCP configuration {path} must contain an mcpServers object")
    return payload


def _write_config(payload: dict[str, Any]) -> None:
    path = mcp_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_mcp_servers() -> list[MCPServerConfig]:
    payload = _read_config()
    servers: list[MCPServerConfig] = []
    for name, raw in payload["mcpServers"].items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise MCPError("MCP server entries require a non-empty name and an object value")
        command = raw.get("command")
        args = raw.get("args", [])
        env = raw.get("env")
        cwd = raw.get("cwd")
        enabled = raw.get("enabled", True)
        if not isinstance(command, str) or not command:
            raise MCPError(f"MCP server {name!r} requires a command")
        if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
            raise MCPError(f"MCP server {name!r} args must be strings")
        if env is not None and (
            not isinstance(env, dict)
            or not all(isinstance(key, str) and isinstance(value, str) for key, value in env.items())
        ):
            raise MCPError(f"MCP server {name!r} env must contain string values")
        if cwd is not None and not isinstance(cwd, str):
            raise MCPError(f"MCP server {name!r} cwd must be a string")
        if not isinstance(enabled, bool):
            raise MCPError(f"MCP server {name!r} enabled must be a boolean")
        servers.append(
            MCPServerConfig(
                name=name,
                command=command,
                args=tuple(args),
                env=env,
                cwd=cwd,
                enabled=enabled,
            )
        )
    return servers


def add_mcp_server(name: str, command: str, args: list[str]) -> MCPServerConfig:
    if not name or _SAFE_NAME.search(name):
        raise MCPError("MCP server names may contain only letters, digits, '.', '_', and '-'")
    payload = _read_config()
    payload["mcpServers"][name] = {"command": command, "args": args, "enabled": True}
    _write_config(payload)
    return MCPServerConfig(name=name, command=command, args=tuple(args))


def remove_mcp_server(name: str) -> None:
    payload = _read_config()
    if name not in payload["mcpServers"]:
        raise MCPError(f"unknown MCP server {name!r}")
    del payload["mcpServers"][name]
    _write_config(payload)


class MCPStdioClient:
    def __init__(self, config: MCPServerConfig, *, timeout: float = 12.0):
        self.config = config
        self.timeout = timeout
        self.process: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._stderr: deque[str] = deque(maxlen=40)
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()

    def connect(self) -> list[dict[str, Any]]:
        home = str(Path.home())
        search_path = os.pathsep.join(
            dict.fromkeys(
                [
                    str(Path(sys.executable).parent),
                    "/opt/homebrew/bin",
                    "/usr/local/bin",
                    f"{home}/.local/bin",
                    f"{home}/.npm-global/bin",
                    *os.environ.get("PATH", "").split(os.pathsep),
                ]
            )
        )
        executable = shutil.which(self.config.command, path=search_path)
        if executable is None:
            raise MCPError(f"command not found: {self.config.command}")
        environment = os.environ.copy()
        environment["PATH"] = search_path
        environment.update(self.config.env or {})
        self.process = subprocess.Popen(
            [executable, *self.config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=Path(self.config.cwd).expanduser() if self.config.cwd else None,
            env=environment,
        )
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "MLXL3 Desktop", "version": "0.1.0"},
            },
        )
        self._notify("notifications/initialized", {})

        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = {"cursor": cursor} if cursor else {}
            result = self._request("tools/list", params)
            page = result.get("tools", [])
            if not isinstance(page, list):
                raise MCPError(f"server {self.config.name!r} returned an invalid tools list")
            tools.extend(tool for tool in page if isinstance(tool, dict))
            cursor = result.get("nextCursor")
            if not isinstance(cursor, str) or not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPToolResult:
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        text_parts: list[str] = []
        structured = result.get("structuredContent")
        if structured is not None:
            text_parts.append(json.dumps(structured, ensure_ascii=False, default=str))
        for block in result.get("content", []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                text_parts.append(block["text"])
            elif block.get("type") == "resource_link" and isinstance(block.get("uri"), str):
                text_parts.append(f"Resource: {block['uri']}")
            elif block.get("type") in {"image", "audio"}:
                text_parts.append(f"[{block['type']} content returned by MCP server]")
        return MCPToolResult(
            text="\n".join(dict.fromkeys(text_parts)) or "Tool completed without text output.",
            is_error=bool(result.get("isError", False)),
        )

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1)

    def _drain_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            self._stderr.append(line.rstrip())

    def _drain_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict):
                self._messages.put(message)

    def _send(self, payload: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None or process.poll() is not None:
            details = self._stderr[-1] if self._stderr else "server exited"
            raise MCPError(f"MCP server {self.config.name!r}: {details}")
        process.stdin.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
        process.stdin.flush()

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        self._send(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )
        while True:
            try:
                message = self._messages.get(timeout=self.timeout)
            except queue.Empty:
                raise MCPError(f"MCP server {self.config.name!r} timed out during {method}")
            except Exception as error:
                raise MCPError(f"MCP server {self.config.name!r} failed during {method}") from error
            if "method" in message and "id" in message:
                self._send(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {"code": -32601, "message": "Client method not supported"},
                    }
                )
                continue
            if message.get("id") != request_id:
                continue
            if isinstance(message.get("error"), dict):
                error = message["error"]
                raise MCPError(
                    f"MCP server {self.config.name!r}: {error.get('message', 'request failed')}"
                )
            result = message.get("result")
            if not isinstance(result, dict):
                raise MCPError(f"MCP server {self.config.name!r} returned an invalid result")
            return result


class MCPManager:
    def __init__(self, configs: list[MCPServerConfig] | None = None):
        self.configs = load_mcp_servers() if configs is None else configs
        self.clients: dict[str, MCPStdioClient] = {}
        self.tools: dict[str, MCPTool] = {}
        self.errors: dict[str, str] = {}

    def connect(self) -> None:
        self.close()
        for config in self.configs:
            if not config.enabled:
                continue
            client = MCPStdioClient(config)
            try:
                raw_tools = client.connect()
                self.clients[config.name] = client
                for raw in raw_tools:
                    name = raw.get("name")
                    if not isinstance(name, str) or not name:
                        continue
                    public_name = self._unique_public_name(config.name, name)
                    description = raw.get("description")
                    schema = raw.get("inputSchema", {"type": "object", "properties": {}})
                    self.tools[public_name] = MCPTool(
                        server=config.name,
                        name=name,
                        public_name=public_name,
                        description=(
                            f"[{config.name}] {description}"
                            if isinstance(description, str) and description
                            else f"Tool {name} from MCP server {config.name}."
                        ),
                        input_schema=(
                            schema
                            if isinstance(schema, dict)
                            else {"type": "object", "properties": {}}
                        ),
                    )
            except Exception as error:  # noqa: BLE001 - isolate optional servers
                client.close()
                self.errors[config.name] = str(error)

    @property
    def chat_tools(self) -> list[dict[str, Any]]:
        return [tool.chat_template_payload() for tool in self.tools.values()]

    @property
    def connected_server_count(self) -> int:
        return len(self.clients)

    def call(self, public_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        tool = self.tools.get(public_name)
        if tool is None:
            return MCPToolResult(f"Unknown MCP tool: {public_name}", True)
        client = self.clients.get(tool.server)
        if client is None:
            return MCPToolResult(f"MCP server unavailable: {tool.server}", True)
        try:
            return client.call_tool(tool.name, arguments)
        except Exception as error:  # noqa: BLE001 - tool errors are model-visible results
            return MCPToolResult(str(error), True)

    def close(self) -> None:
        for client in self.clients.values():
            client.close()
        self.clients.clear()
        self.tools.clear()
        self.errors.clear()

    def _unique_public_name(self, server: str, tool: str) -> str:
        base = _SAFE_NAME.sub("_", f"{server}.{tool}")[:128]
        candidate = base
        suffix = 2
        while candidate in self.tools:
            ending = f"_{suffix}"
            candidate = base[: 128 - len(ending)] + ending
            suffix += 1
        return candidate

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
