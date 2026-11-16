"""Bridge protocol v1: request validation and event emission.

Every line on stdin is one JSON object with a ``type``. Every line written on
the protocol stream is one JSON object. Nothing else may be written there, so
the bridge moves library output (progress bars, warnings) to stderr.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import threading
from typing import IO, Any

MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_MESSAGES = 4096
MAX_IMAGES = 16
MAX_TOOLS = 256
MAX_TOOL_CALLS = 64
IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff")
ROLES = frozenset({"system", "user", "assistant", "tool"})
REQUEST_TYPES = frozenset({"ping", "generate", "cancel", "shutdown"})


class RequestError(ValueError):
    """A request that must be answered with an ``error`` event."""

    def __init__(self, message: str, request_id: str = ""):
        super().__init__(message)
        self.request_id = request_id


def _finite(value: Any, name: str, request_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{name} must be a number", request_id)
    value = float(value)
    if not math.isfinite(value):
        raise RequestError(f"{name} must be finite", request_id)
    return value


def _integer(value: Any, name: str, request_id: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RequestError(f"{name} must be an integer", request_id)
    return value


def parse_request(line: str | bytes) -> dict:
    """Parse and validate one request line. Raises RequestError."""
    if isinstance(line, bytes):
        if len(line) > MAX_REQUEST_BYTES:
            raise RequestError("request line exceeds 16 MiB")
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as error:
            raise RequestError(f"request is not UTF-8: {error}") from None
    elif len(line.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise RequestError("request line exceeds 16 MiB")
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as error:
        raise RequestError(f"invalid bridge request: {error}") from None
    if not isinstance(raw, dict):
        raise RequestError("bridge request must be a JSON object")
    request_id = raw.get("request_id", "")
    if not isinstance(request_id, str):
        raise RequestError("request_id must be a string")
    kind = raw.get("type")
    if kind not in REQUEST_TYPES:
        raise RequestError(f"unknown request type: {kind!r}", request_id)
    if kind != "generate":
        return {"type": kind, "request_id": request_id}
    if not request_id:
        raise RequestError("generate requires a request_id")
    return {"type": "generate", "request_id": request_id, **_generate_fields(raw, request_id)}


def _image(item: Any, index: int, request_id: str) -> bytes:
    """One base64 PNG, JPEG or WebP image, decoded strictly."""
    if not isinstance(item, str):
        raise RequestError(f"message {index}: images must be base64 strings", request_id)
    try:
        data = base64.b64decode(item, validate=True)
    except (binascii.Error, ValueError):
        raise RequestError(f"message {index}: an image is not valid base64", request_id) from None
    webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if not (webp or data.startswith(IMAGE_MAGIC)):
        raise RequestError(f"message {index}: images must be PNG, JPEG or WebP", request_id)
    return data


def _generate_fields(raw: dict, request_id: str) -> dict:
    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list", request_id)
    if len(messages) > MAX_MESSAGES:
        raise RequestError(f"at most {MAX_MESSAGES} messages are accepted", request_id)
    clean = []
    image_count = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RequestError(f"message {index} must be an object", request_id)
        role, content = message.get("role"), message.get("content")
        if role not in ROLES:
            raise RequestError(f"message {index} has an unsupported role", request_id)
        if not isinstance(content, str):
            raise RequestError(f"message {index} content must be text", request_id)
        entry = {"role": role, "content": content}
        images = message.get("images")
        if images is not None:
            if role != "user" or not isinstance(images, list):
                raise RequestError(f"message {index}: images are accepted on user messages only", request_id)
            if images:
                entry["images"] = [_image(item, index, request_id) for item in images]
                image_count += len(images)
                if image_count > MAX_IMAGES:
                    raise RequestError(f"at most {MAX_IMAGES} images are accepted", request_id)
        tool_calls = message.get("tool_calls")
        if tool_calls is not None:
            if role != "assistant" or not isinstance(tool_calls, list) or len(tool_calls) > MAX_TOOL_CALLS:
                raise RequestError(f"message {index}: tool_calls must be a list on an assistant message", request_id)
            if not all(isinstance(call, dict) and isinstance(call.get("function"), dict)
                       and isinstance(call["function"].get("name"), str) for call in tool_calls):
                raise RequestError(f"message {index}: each tool call needs a function name", request_id)
            if tool_calls:
                entry["tool_calls"] = tool_calls
        name = message.get("name")
        if name is not None:
            if role != "tool" or not isinstance(name, str) or not 0 < len(name) <= 256:
                raise RequestError(f"message {index}: name is accepted on tool messages only", request_id)
            entry["name"] = name
        clean.append(entry)
    if clean[-1]["role"] == "assistant":
        raise RequestError("the last message must not be an assistant message", request_id)

    max_tokens = _integer(raw.get("max_tokens", -1), "max_tokens", request_id)
    if max_tokens == 0 or max_tokens < -1:
        raise RequestError("max_tokens must be -1 (unlimited) or positive", request_id)
    temperature = _finite(raw.get("temperature", 0.0), "temperature", request_id)
    if not 0.0 <= temperature <= 5.0:
        raise RequestError("temperature must be between 0 and 5", request_id)
    top_k = _integer(raw.get("top_k", 0), "top_k", request_id)
    if top_k < 0:
        raise RequestError("top_k must be 0 (disabled) or positive", request_id)
    top_p = _finite(raw.get("top_p", 1.0), "top_p", request_id)
    if not 0.0 < top_p <= 1.0:
        raise RequestError("top_p must be in (0, 1]", request_id)
    min_p = _finite(raw.get("min_p", 0.0), "min_p", request_id)
    if not 0.0 <= min_p < 1.0:
        raise RequestError("min_p must be in [0, 1)", request_id)
    repetition_penalty = _finite(raw.get("repetition_penalty", 1.0), "repetition_penalty", request_id)
    if not 0.5 <= repetition_penalty <= 3.0:
        raise RequestError("repetition_penalty must be between 0.5 and 3", request_id)
    seed = raw.get("seed")
    if seed is not None:
        seed = _integer(seed, "seed", request_id)
        if not 0 <= seed < 2**63:
            raise RequestError("seed must be a non-negative 63-bit integer", request_id)
    enable_thinking = raw.get("enable_thinking")
    if enable_thinking is not None and not isinstance(enable_thinking, bool):
        raise RequestError("enable_thinking must be a boolean", request_id)
    tools = raw.get("tools")
    if tools is not None:
        if not isinstance(tools, list) or len(tools) > MAX_TOOLS:
            raise RequestError(f"tools must be a list of at most {MAX_TOOLS} functions", request_id)
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
                raise RequestError("each tool needs a function name", request_id)
    return {
        "messages": clean,
        "tools": tools or [],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "seed": seed,
        "enable_thinking": enable_thinking,
    }


def is_greedy(request: dict) -> bool:
    return request["temperature"] == 0.0 or request["top_k"] == 1


class Emitter:
    """Thread-safe writer of protocol events, one JSON object per line."""

    def __init__(self, stream: IO[str]):
        self._stream = stream
        self._lock = threading.Lock()
        self.closed = False

    def emit(self, event: dict) -> None:
        line = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._lock:
            if self.closed:
                return
            try:
                self._stream.write(line + "\n")
                self._stream.flush()
            except (BrokenPipeError, ValueError, OSError):
                # The app went away; the reader sees EOF and shuts the worker down.
                self.closed = True
