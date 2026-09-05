"""Ollama-style local model registry and streaming terminal chat."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mlxl3.mcp import (
    MCPError,
    MCPManager,
    add_mcp_server,
    ensure_mcp_config,
    load_mcp_servers,
    remove_mcp_server,
)
from mlxl3.registry import (
    ModelEntry,
    RegistryError,
    human_size,
    load_registry,
    managed_models_path,
    register_model,
    remove_model,
    resolve_model,
    validate_model_name,
)


@dataclass(frozen=True)
class GenerationStats:
    ttft_seconds: float
    prefill_tps: float
    decode_tps: float
    prompt_tokens: int
    generated_tokens: int
    peak_memory_gb: float
    cached_prompt_tokens: int = 0
    evaluated_prompt_tokens: int = 0
    template_seconds: float = 0.0
    stable_prefill_seconds: float = 0.0
    stable_prefill_tokens: int = 0
    cache_fork_seconds: float = 0.0
    first_visible_text_seconds: float | None = None
    cache_lookup_seconds: float = 0.0
    tokenizer_encode_seconds: float = 0.0
    stream_setup_seconds: float = 0.0
    suffix_prefill_chunk_seconds: float = 0.0


class MemorySafetyError(RuntimeError):
    """Generation stopped before Metal pressure can destabilize the desktop."""


class GenerationCancelled(RuntimeError):
    """A bridge client cooperatively cancelled the active generation."""


# Zero selects a memory-aware serialized-QMM step. An explicit environment
# override remains useful for profiling and unusually constrained hosts.
_PREFILL_STEP_SIZE = int(os.environ.get("MLXL3_PREFILL_STEP_SIZE", "0"))
_PREFIX_CACHE_BLOCK_SIZE = int(os.environ.get("MLXL3_PREFIX_CACHE_BLOCK_SIZE", "2048"))
_PREFIX_CACHE_BUDGET_BYTES = int(
    float(os.environ.get("MLXL3_PREFIX_CACHE_GB", "0.5")) * 1_000_000_000
)
_SESSION_CACHE_BUDGET_BYTES = int(
    float(os.environ.get("MLXL3_SESSION_CACHE_GB", "1.0")) * 1_000_000_000
)
_WARM_MODEL_ON_LOAD = os.environ.get("MLXL3_WARM_MODEL_ON_LOAD", "1") != "0"
_USE_COW_KV = os.environ.get('MLXL3_COW_KV', '0') == '1'
_COMPACT_CONV_STATES = os.environ.get('MLXL3_COMPACT_CONV_STATES', '0') == '1'
_RETAIN_PREFILL_ALLOCATOR = os.environ.get('MLXL3_RETAIN_PREFILL_ALLOCATOR', '0') == '1'


def _model_num_experts(model: Any) -> int:
    """Read common MLX-LM config locations without tying the scheduler to Qwen."""

    language_model = getattr(model, "language_model", None)
    for owner in (model, language_model):
        args = getattr(owner, "args", None)
        for name in ("num_experts", "num_local_experts"):
            value = getattr(args, name, 0)
            if isinstance(value, int) and value > 0:
                return value
    return 0


def _cache_nbytes(cache: list[Any] | None, seen: set[int] | None = None) -> int:
    """Conservative cache accounting; not a physical allocation census.

    For known KV caches count the owned reserve, not the used-prefix view.
    Unknown view aliases cannot be deduplicated reliably at the Python level.
    """

    if cache is None:
        return 0
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    from mlxl3.cache import SharedKVCache

    seen = set() if seen is None else seen

    def visit(value: Any) -> int:
        if isinstance(value, mx.array):
            identity = id(value)
            if identity in seen:
                return 0
            seen.add(identity)
            return int(value.nbytes)
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        return 0

    total = 0
    for item in cache:
        if type(item) is SharedKVCache and item.storage_owner is not None:
            owner = item.storage_owner
            if id(owner) not in seen:
                total += owner.nbytes
                seen.add(id(owner))
        elif type(item) in (KVCache, SharedKVCache):
            total += visit((item.keys, item.values))
        else:
            total += visit(item.state)
    return total


class GenerationSession:
    """Reuse the stable chat prefix and its model cache across turns."""

    def __init__(self) -> None:
        self.prompt_cache: list[Any] | None = None
        self.tokens: list[int] = []
        self.exact_cache: list[Any] | None = None
        self.exact_tokens: list[int] = []
        self.block_caches: OrderedDict[int, list[Any]] = OrderedDict()
        self.stable_prefill_seconds = 0.0
        self.stable_prefill_tokens = 0
        self.cache_fork_seconds = 0.0
        self.cache_lookup_seconds = 0.0

    def reset(self) -> None:
        self.prompt_cache = None
        self.tokens = []
        self.exact_cache = None
        self.exact_tokens = []
        self.block_caches.clear()

    @staticmethod
    def _common_prefix(left: list[int], right: list[int]) -> int:
        common = 0
        for old, new in zip(left, right, strict=False):
            if old != new:
                break
            common += 1
        return common

    @staticmethod
    def _clone_cache(
        cache: list[Any], *, synchronize: bool = True, reserve_tokens: int = 1,
    ) -> list[Any]:
        import mlx.core as mx
        from mlx.utils import tree_map
        from mlx_lm.models.cache import ArraysCache, KVCache

        from mlxl3.cache import SharedKVCache

        def copy_array(value: Any) -> Any:
            return mx.array(value) if isinstance(value, mx.array) else value

        cloned = []
        for item in cache:
            if type(item) is SharedKVCache or (_USE_COW_KV and type(item) is KVCache):
                cloned.append(SharedKVCache.adopt(item).fork())
                continue
            if type(item) is KVCache:
                fork = KVCache()
                if item.keys is not None:
                    length = item.offset
                    capacity = length
                    if reserve_tokens > 0:
                        capacity = ((length + reserve_tokens + item.step - 1) // item.step) * item.step
                    def copy_prefix(value, capacity=capacity, length=length):
                        shape = (*value.shape[:2], capacity, value.shape[3])
                        storage = mx.zeros(shape, dtype=value.dtype)
                        if length:
                            storage[..., :length, :] = value[..., :length, :]
                        return storage
                    fork.keys = copy_prefix(item.keys)
                    fork.values = copy_prefix(item.values)
                    fork.offset = length
                cloned.append(fork)
                continue
            # Qwen Gated DeltaNet replaces its ArraysCache states on every
            # update instead of writing into their buffers. Rebuild the Python
            # containers but share those immutable arrays across forks. KV
            # caches retain real copies because their reserved buffers are
            # updated in place during decode.
            mapper = (lambda value: value) if isinstance(item, ArraysCache) else copy_array
            cloned.append(
                type(item).from_state(
                    tree_map(mapper, item.state),
                    tree_map(mapper, item.meta_state),
                )
            )
        states = [(item.keys, item.values) if type(item) in (KVCache, SharedKVCache) else item.state for item in cloned]
        if synchronize:
            mx.eval(states)
        else:
            mx.async_eval(states)
        return cloned

    def nbytes(self) -> int:
        """Return conservative retained bytes, including owned KV reserves."""

        seen: set[int] = set()
        total = _cache_nbytes(self.prompt_cache, seen)
        total += _cache_nbytes(self.exact_cache, seen)
        for cache in self.block_caches.values():
            total += _cache_nbytes(cache, seen)
        return total

    def _prune_block_caches(self) -> None:
        while self.block_caches and self.nbytes() > _PREFIX_CACHE_BUDGET_BYTES:
            self.block_caches.popitem(last=False)

    def _store_block_cache(self, length: int) -> None:
        if (
            self.prompt_cache is None
            or _PREFIX_CACHE_BLOCK_SIZE <= 0
            or _PREFIX_CACHE_BUDGET_BYTES <= 0
        ):
            return
        # Decide admission before allocating a copy. ArraysCache states retain
        # the same objects in the existing immutable-state fork implementation.
        from mlx_lm.models.cache import ArraysCache, KVCache

        from mlxl3.cache import SharedKVCache

        incremental = 0
        for item in self.prompt_cache:
            if isinstance(item, ArraysCache):
                continue
            if type(item) is SharedKVCache or (_USE_COW_KV and type(item) is KVCache):
                continue
            if type(item) is KVCache and item.keys is not None:
                incremental += sum(
                    value.shape[0] * value.shape[1] * item.offset * value.shape[3] * value.itemsize
                    for value in (item.keys, item.values)
                )
            else:
                incremental += _cache_nbytes([item])
        seen: set[int] = set()
        active = _cache_nbytes(self.prompt_cache, seen) + _cache_nbytes(self.exact_cache, seen)
        if active + incremental > _PREFIX_CACHE_BUDGET_BYTES:
            return
        self.block_caches.pop(length, None)
        while self.block_caches and self.nbytes() + incremental > _PREFIX_CACHE_BUDGET_BYTES:
            self.block_caches.popitem(last=False)
        self.block_caches[length] = self._clone_cache(
            self.prompt_cache,
            synchronize=False,
            reserve_tokens=0,
        )
        self.block_caches.move_to_end(length)
        self._prune_block_caches()

    def _restore_block_cache(self, common: int) -> int:
        candidates = [length for length in self.block_caches if length <= common]
        if not candidates:
            return 0
        length = max(candidates)
        self.prompt_cache = self._clone_cache(self.block_caches[length])
        self.tokens = self.tokens[:length]
        for stale in [position for position in self.block_caches if position > length]:
            del self.block_caches[stale]
        self.block_caches.move_to_end(length)
        return length

    def prepare(
        self,
        model: Any,
        full_tokens: list[int],
        stable_tokens: list[int],
        *,
        should_cancel: Callable[[], bool] | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> tuple[list[int], list[Any], int, int]:
        import mlx.core as mx
        from mlx_lm.models.cache import (
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )

        def check_cancel():
            if should_cancel is not None and should_cancel():
                # A partially updated recurrent state is not a valid snapshot
                # of self.tokens. Retire it rather than reuse a wrong offset.
                self.reset()
                raise GenerationCancelled

        check_cancel()
        lookup_started = time.monotonic()
        self.stable_prefill_seconds = 0.0
        self.stable_prefill_tokens = 0
        self.cache_fork_seconds = 0.0
        self.cache_lookup_seconds = 0.0

        # A template should append its assistant-generation marker to the
        # stable history. If it does not, fall back to a clean full prefill.
        if full_tokens[: len(stable_tokens)] != stable_tokens:
            self.reset()
            stable_tokens = []
        if not stable_tokens:
            return full_tokens, make_prompt_cache(model), 0, len(full_tokens)

        # A normally completed assistant turn is already in the exact byte
        # form emitted by the template. Promote its post-generation cache and
        # avoid reevaluating that answer. Truncated reasoning falls back to the
        # stable pre-assistant snapshot below.
        if self.exact_cache is not None and self.exact_tokens:
            exact_common = self._common_prefix(self.exact_tokens, stable_tokens)
            if exact_common == len(self.exact_tokens):
                self.prompt_cache = self.exact_cache
                self.tokens = self.exact_tokens
            self.exact_cache = None
            self.exact_tokens = []

        common = self._common_prefix(self.tokens, stable_tokens)
        if self.prompt_cache is None:
            self.prompt_cache = make_prompt_cache(model)
            common = 0
        elif common < len(self.tokens):
            if can_trim_prompt_cache(self.prompt_cache):
                requested = len(self.tokens) - common
                if trim_prompt_cache(self.prompt_cache, requested) != requested:
                    self.prompt_cache = make_prompt_cache(model)
                    common = 0
            else:
                # Recurrent-state models such as Qwen3.6 cannot trim their
                # state cache. Restore the nearest immutable block snapshot;
                # only the suffix after the fork point must be evaluated.
                common = self._restore_block_cache(common)
                if common == 0:
                    self.prompt_cache = make_prompt_cache(model)
                    self.block_caches.clear()

        extension = stable_tokens[common:]
        if on_status is not None:
            on_status(f"Lecture du contexte · {len(extension):,} nouveaux tokens · {common:,} en cache")
        self.cache_lookup_seconds = time.monotonic() - lookup_started
        prefill_started = time.monotonic()
        processed = common
        step_size = _select_prefill_step_size(
            mx.get_active_memory(),
            _mlx_memory_guard_bytes(),
            token_count=len(extension),
            num_experts=_model_num_experts(model),
        )
        while processed < len(stable_tokens):
            check_cancel()
            active = mx.get_active_memory()
            guard = _mlx_memory_guard_bytes()
            if active >= guard:
                self.reset()
                raise MemorySafetyError("prefill arrêté avant saturation de la mémoire unifiée")
            step_size = min(step_size, _select_prefill_step_size(
                active, guard, token_count=len(stable_tokens) - processed,
                num_experts=_model_num_experts(model),
            ))
            chunk_end = min(processed + step_size, len(stable_tokens))
            if _PREFIX_CACHE_BLOCK_SIZE > 0:
                next_boundary = (
                    (processed // _PREFIX_CACHE_BLOCK_SIZE) + 1
                ) * _PREFIX_CACHE_BLOCK_SIZE
                chunk_end = min(chunk_end, next_boundary)
            chunk = stable_tokens[processed:chunk_end]
            model(mx.array(chunk)[None], cache=self.prompt_cache)
            if _COMPACT_CONV_STATES:
                from mlxl3.cache import compact_conv_states
                compact_conv_states(self.prompt_cache, len(chunk))
            # Submit successive cache states without stalling the CPU between
            # chunks. MLX preserves their data dependencies on the generation
            # stream, so one terminal synchronization is sufficient.
            mx.async_eval([item.state for item in self.prompt_cache])
            processed = chunk_end
            if (
                _PREFIX_CACHE_BLOCK_SIZE > 0
                and processed % _PREFIX_CACHE_BLOCK_SIZE == 0
            ):
                self._store_block_cache(processed)
        if extension:
            if on_status is not None:
                on_status(f"Traitement du contexte · {len(extension):,} nouveaux tokens · {common:,} en cache")
            mx.eval([item.state for item in self.prompt_cache])
            if not _RETAIN_PREFILL_ALLOCATOR:
                mx.clear_cache()
        check_cancel()
        self.stable_prefill_seconds = time.monotonic() - prefill_started if extension else 0.0
        self.stable_prefill_tokens = len(extension)
        self.tokens = list(stable_tokens)

        # Generation mutates its cache. Preserve the stable-history cache so
        # recurrent layers can reuse it on the following turn without rewind.
        fork_started = time.monotonic()
        generation_cache = self._clone_cache(self.prompt_cache)
        self.cache_fork_seconds = time.monotonic() - fork_started
        generation_tokens = full_tokens[len(stable_tokens) :]
        return (
            generation_tokens,
            generation_cache,
            common,
            len(full_tokens) - common,
        )

    def finish(
        self,
        prompt_tokens: list[int],
        generated_tokens: list[int],
        generation_cache: list[Any],
    ) -> None:
        self.exact_tokens = [*prompt_tokens, *generated_tokens]
        self.exact_cache = generation_cache


class GenerationSessionPool:
    """RAM-bounded LRU of warm prompt caches keyed by conversation."""

    def __init__(self, budget_bytes: int = _SESSION_CACHE_BUDGET_BYTES) -> None:
        self.budget_bytes = max(0, budget_bytes)
        self.sessions: OrderedDict[str, GenerationSession] = OrderedDict()

    def acquire(self, conversation_id: str) -> GenerationSession:
        session = self.sessions.get(conversation_id)
        if session is None:
            session = GenerationSession()
            self.sessions[conversation_id] = session
        self.sessions.move_to_end(conversation_id)
        return session

    def prune(self, active_id: str) -> None:
        while len(self.sessions) > 1:
            retained = sum(session.nbytes() for session in self.sessions.values())
            if retained <= self.budget_bytes:
                break
            oldest = next(iter(self.sessions))
            if oldest == active_id:
                self.sessions.move_to_end(oldest)
                continue
            self.sessions.pop(oldest).reset()


def _mlx_memory_guard_bytes() -> int:
    import mlx.core as mx

    info = mx.device_info()
    physical = int(info.get("memory_size") or 0)
    recommended = int(info.get("max_recommended_working_set_size") or 0)
    candidates = [value for value in (int(physical * 0.82), int(recommended * 0.96)) if value]
    return min(candidates) if candidates else 16_000_000_000


def _select_prefill_step_size(
    active_bytes: int,
    limit_bytes: int,
    *,
    token_count: int = 0,
    num_experts: int = 0,
) -> int:
    """Choose the widest measured-fast QMM block that fits with headroom."""

    if _PREFILL_STEP_SIZE > 0:
        return _PREFILL_STEP_SIZE
    # Very long prefills on 128+ expert MoEs can exceed Metal's resource-count
    # ceiling even when byte memory remains available. A 512-token command is
    # the proven-safe fallback; short and cached extensions retain wider QMMs.
    if num_experts >= 128 and token_count >= 32_768:
        return 512
    headroom = max(0, limit_bytes - active_bytes)
    if headroom >= 2_000_000_000:
        return 2048
    if headroom >= 750_000_000:
        return 1024
    return 512


class ThinkingSplitter:
    """Split streamed model text into reasoning and final-answer fragments."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, *, initial_mode: str = "answer"):
        if initial_mode not in {"answer", "thinking"}:
            raise ValueError(f"unsupported thinking mode: {initial_mode!r}")
        self.mode = initial_mode
        self.buffer = ""
        self.drop_leading_newline = False

    def configure_for_prompt(self, prompt: str) -> None:
        """Account for chat templates that prefill the opening think tag."""

        self.mode = "thinking" if _prompt_prefills_thinking(prompt) else "answer"

    @staticmethod
    def _partial_marker_length(text: str, marker: str) -> int:
        limit = min(len(text), len(marker) - 1)
        for length in range(limit, 0, -1):
            if text.endswith(marker[:length]):
                return length
        return 0

    def _fragment(self, text: str) -> tuple[str, str] | None:
        if self.drop_leading_newline and text:
            if text.startswith("\r\n"):
                text = text[2:]
            elif text.startswith("\n"):
                text = text[1:]
            self.drop_leading_newline = False
        return (self.mode, text) if text else None

    def feed(self, text: str) -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        self.buffer += text
        while self.buffer:
            markers = (self._CLOSE, '<channel|>') if self.mode == 'thinking' else (self._OPEN, '<|channel>thought')
            matches = [(self.buffer.find(m), m) for m in markers if m in self.buffer]
            position, marker = min(matches) if matches else (-1, markers[0])
            if position >= 0:
                fragment = self._fragment(self.buffer[:position])
                if fragment is not None:
                    fragments.append(fragment)
                self.buffer = self.buffer[position + len(marker) :]
                self.mode = "answer" if self.mode == "thinking" else "thinking"
                self.drop_leading_newline = True
                continue
            keep = max(self._partial_marker_length(self.buffer, m) for m in markers)
            emit_until = len(self.buffer) - keep
            fragment = self._fragment(self.buffer[:emit_until])
            if fragment is not None:
                fragments.append(fragment)
            self.buffer = self.buffer[emit_until:]
            break
        return fragments

    def finish(self) -> list[tuple[str, str]]:
        fragment = self._fragment(self.buffer)
        self.buffer = ""
        return [] if fragment is None else [fragment]


def _prompt_prefills_thinking(prompt: str) -> bool:
    """Return whether the generation prompt already opened a reasoning block."""

    return prompt.rstrip().endswith((ThinkingSplitter._OPEN, '<|channel>thought'))


class ThinkingRenderer:
    """Incrementally render ``<think>`` blocks separately from final text."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(
        self,
        *,
        stream=None,
        color: bool | None = None,
        initial_mode: str = "answer",
    ):
        self.stream = sys.stdout if stream is None else stream
        self.color = (
            bool(getattr(self.stream, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ
            if color is None
            else color
        )
        self.splitter = ThinkingSplitter(initial_mode=initial_mode)
        self.section: str | None = None
        self.last_was_newline = True

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()
        if text:
            self.last_was_newline = text.endswith("\n")

    def _start_section(self, mode: str) -> None:
        if self.section == mode:
            return
        if self.section is not None and not self.last_was_newline:
            self._write("\n")
        label = "Réflexion" if mode == "thinking" else "Réponse"
        if self.color:
            style = "\033[1;35m" if mode == "thinking" else "\033[1;36m"
            self._write(f"{style}── {label} ──\033[0m\n")
        else:
            self._write(f"── {label} ──\n")
        self.section = mode

    def _emit(self, mode: str, text: str) -> None:
        if not text:
            return
        self._start_section(mode)
        if self.color and mode == "thinking":
            self._write(f"\033[35m{text}\033[0m")
        else:
            self._write(text)

    def feed(self, text: str) -> None:
        for mode, fragment in self.splitter.feed(text):
            self._emit(mode, fragment)

    def finish(self) -> None:
        for mode, fragment in self.splitter.finish():
            self._emit(mode, fragment)
        if self.section is not None and not self.last_was_newline:
            self._write("\n")


@dataclass(frozen=True)
class ToolCallRequest:
    name: str
    arguments: dict[str, Any]


class ToolCallStreamFilter:
    """Hide streamed tool payloads while preserving ordinary answer text."""

    _OPEN = "<tool_call>"
    _CLOSE = "</tool_call>"

    def __init__(self):
        self.buffer = ""
        self.inside_call = False

    def feed(self, text: str) -> list[str]:
        visible: list[str] = []
        self.buffer += text
        while self.buffer:
            marker = self._CLOSE if self.inside_call else self._OPEN
            position = self.buffer.find(marker)
            if position >= 0:
                if not self.inside_call and position:
                    visible.append(self.buffer[:position])
                self.buffer = self.buffer[position + len(marker) :]
                self.inside_call = not self.inside_call
                continue
            keep = ThinkingSplitter._partial_marker_length(self.buffer, marker)
            emit_until = len(self.buffer) - keep
            if not self.inside_call and emit_until:
                visible.append(self.buffer[:emit_until])
            self.buffer = self.buffer[emit_until:]
            break
        return visible

    def finish(self) -> list[str]:
        visible = [] if self.inside_call or not self.buffer else [self.buffer]
        self.buffer = ""
        return visible


_TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_FUNCTION = re.compile(r"<function=([^>\n]+)>\s*(.*?)\s*</function>", re.DOTALL)
_TOOL_PARAMETER = re.compile(r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def _parse_tool_calls(response: str) -> list[ToolCallRequest]:
    calls: list[ToolCallRequest] = []
    for match in _TOOL_BLOCK.finditer(response):
        body = match.group(1).strip()
        function = _TOOL_FUNCTION.fullmatch(body)
        if function is not None:
            arguments = {
                parameter.group(1).strip(): _parse_tool_argument(parameter.group(2))
                for parameter in _TOOL_PARAMETER.finditer(function.group(2))
            }
            calls.append(ToolCallRequest(function.group(1).strip(), arguments))
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        payloads = payload if isinstance(payload, list) else [payload]
        for item in payloads:
            if not isinstance(item, dict):
                continue
            function_payload = item.get("function", item)
            if not isinstance(function_payload, dict):
                continue
            name = function_payload.get("name")
            arguments = function_payload.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if isinstance(name, str) and isinstance(arguments, dict):
                calls.append(ToolCallRequest(name, arguments))
    return calls


def _parse_tool_argument(raw: str) -> Any:
    value = raw.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _without_tool_calls(response: str) -> str:
    return _TOOL_BLOCK.sub("", _assistant_context(response)).strip()


def _cache_context(response: str) -> str:
    """Keep exact generated reasoning/final bytes for lossless prefix reuse."""

    return _TOOL_BLOCK.sub("", response)


def _reasoning_context(response: str) -> str:
    if '<|channel>thought' in response:
        return response.split('<|channel>thought', 1)[1].split('<channel|>', 1)[0].strip()
    close = response.rfind("</think>")
    if close < 0:
        return ""
    return response[:close].removeprefix("<think>").strip()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxl3",
        description="Run registered EXL3 models locally on Apple Metal.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    list_models = commands.add_parser(
        "list", aliases=["ls"], help="list registered local models"
    )
    list_models.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    register = commands.add_parser("register", aliases=["add"], help="register a model")
    register.add_argument("name")
    register.add_argument("path", type=Path)
    register.add_argument("--force", action="store_true")

    download = commands.add_parser(
        "download", help="download and register an EXL3 model from Hugging Face"
    )
    download.add_argument("repo", help="Hugging Face repository, for example owner/model")
    download.add_argument("--revision", help="branch, tag, or EXL3 BPW revision")
    download.add_argument("--name", help="local registry name; defaults to the repository name")
    download.add_argument("--directory", type=Path, help="custom destination directory")
    download.add_argument("--force", action="store_true", help="redownload existing files")
    download.add_argument("--json", action="store_true", help=argparse.SUPPRESS)

    remove = commands.add_parser("remove", aliases=["rm"], help="remove a registry entry")
    remove.add_argument("name")

    mcp = commands.add_parser("mcp", help="manage local MCP stdio servers")
    mcp_commands = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_list = mcp_commands.add_parser("list", aliases=["ls"], help="list MCP servers")
    mcp_list.add_argument("--json", action="store_true")
    mcp_check = mcp_commands.add_parser("check", help="connect and list MCP tools without loading a model")
    mcp_check.add_argument("--json", action="store_true")
    mcp_commands.add_parser("config", help="print and create the MCP configuration path")
    mcp_add = mcp_commands.add_parser("add", help="add or replace an MCP stdio server")
    mcp_add.add_argument("name")
    mcp_add.add_argument("server_command")
    mcp_add.add_argument("server_args", nargs=argparse.REMAINDER)
    mcp_remove = mcp_commands.add_parser("remove", aliases=["rm"], help="remove an MCP server")
    mcp_remove.add_argument("name")

    run = commands.add_parser("run", help="stream a response or start an interactive chat")
    run.add_argument("model", help="registered model name or local model directory")
    run.add_argument("prompt", nargs="?", help="one-shot prompt; omit for interactive chat")
    run.add_argument("--max-tokens", type=int, default=512)
    run.add_argument("--system", help="optional system message")
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--top-k", type=int, default=80)
    run.add_argument("--repetition-penalty", type=float, default=1.05)
    bridge = commands.add_parser("bridge", help=argparse.SUPPRESS)
    bridge.add_argument("model", help="registered model name or local model directory")
    return parser


def _format_models(entries: dict[str, ModelEntry]) -> str:
    headers = ("NAME", "FORMAT", "BPW", "SIZE", "MODEL TYPE", "PATH")
    rows = [
        (
            entry.name,
            entry.format,
            "-" if entry.bits is None else f"{entry.bits:.2f}",
            human_size(entry.size_bytes),
            entry.model_type,
            entry.path,
        )
        for entry in sorted(entries.values(), key=lambda item: item.name.lower())
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)).rstrip()

    return "\n".join([render(headers), *(render(row) for row in rows)])


def _format_mcp_servers() -> str:
    servers = load_mcp_servers()
    if not servers:
        return "Aucun serveur MCP configuré."
    return "\n".join(
        f"{server.name}\t{'ACTIF' if server.enabled else 'INACTIF'}\t"
        f"{server.url or server.command} {' '.join(server.args)}".rstrip()
        for server in servers
    )


def _legacy_tool_messages(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, str]]:
    specification = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    instructions = (
        "You can use local tools. Available tools as JSON: "
        f"{specification}\n"
        "When a tool is necessary, output exactly one or more calls as "
        '<tool_call>{"name":"tool.name","arguments":{}}</tool_call>. '
        "Do not invent tool results. After a tool response, answer the user normally."
    )
    rendered: list[dict[str, str]] = []
    inserted_instructions = False
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "system" and not inserted_instructions:
            rendered.append({"role": "system", "content": instructions + "\n\n" + content})
            inserted_instructions = True
        elif role == "assistant":
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list):
                blocks = []
                for call in tool_calls:
                    function = call.get("function", {}) if isinstance(call, dict) else {}
                    if isinstance(function, dict):
                        blocks.append(
                            "<tool_call>"
                            + json.dumps(function, ensure_ascii=False, separators=(",", ":"))
                            + "</tool_call>"
                        )
                content = "\n".join(part for part in (content, *blocks) if part)
            rendered.append({"role": "assistant", "content": content})
        elif role == "tool":
            name = str(message.get("name", "tool"))
            rendered.append(
                {
                    "role": "user",
                    "content": f"<tool_response name={name}>\n{content}\n</tool_response>",
                }
            )
        else:
            rendered.append({"role": role, "content": content})
    if not inserted_instructions:
        rendered.insert(0, {"role": "system", "content": instructions})
    return rendered


def _render_generation_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    add_generation_prompt: bool = True,
) -> str:
    template_options: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        # Qwen templates otherwise delete earlier reasoning. Keeping the
        # generated bytes makes the next prompt an exact cache extension.
        "preserve_thinking": True,
    }
    if tools:
        template_options["tools"] = tools
    prompt = tokenizer.apply_chat_template(messages, **template_options)
    if not tools:
        return prompt
    tool_names = [tool.get("function", {}).get("name") for tool in tools]
    if all(isinstance(name, str) and name in prompt for name in tool_names):
        return prompt
    return tokenizer.apply_chat_template(
        _legacy_tool_messages(messages, tools),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        preserve_thinking=True,
    )


def _load_model(model_path: Path):
    import mlx.core as mx
    from mlx_lm.utils import load_tokenizer

    from mlxl3.checkpoint import load_exl3_model

    mx.set_memory_limit(_mlx_memory_guard_bytes())
    started = time.perf_counter()
    model, _, report = load_exl3_model(model_path, lazy=False)
    generation_config_path = model_path / 'generation_config.json'
    generation_config = json.loads(generation_config_path.read_text()) if generation_config_path.is_file() else {}
    tokenizer = load_tokenizer(str(model_path), eos_token_ids=generation_config.get('eos_token_id'))
    if os.environ.get('MLXL3_CACHED_DETOKENIZER', '1') != '0':
        from mlxl3.tokenizer import cache_detokenizer_metadata
        cache_detokenizer_metadata(tokenizer)
    if _WARM_MODEL_ON_LOAD:
        _warm_model(model, tokenizer)
    return (
        model,
        tokenizer,
        len(report.replaced),
        time.perf_counter() - started,
        mx.get_active_memory() / 1e9,
    )


def _warm_model(model: Any, tokenizer: Any, *, prompt_tokens: int = 32) -> None:
    """Compile representative prefill/decode graphs before reporting ready."""

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    encoded = tokenizer.encode("MLXL3 warmup", add_special_tokens=False)
    token = int(encoded[0]) if encoded else int(tokenizer.eos_token_id or 0)
    prefix = mx.array([[token] * prompt_tokens], dtype=mx.int32)
    prompt_cache = make_prompt_cache(model)
    logits = model(prefix, cache=prompt_cache)
    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(next_token)
    decode_logits = model(next_token[:, None], cache=prompt_cache)
    mx.eval(decode_logits)
    mx.clear_cache()


def _prefix_processors(processors, prefix, windows):
    """Wrap explicit history contracts; unknown/unbounded processors keep all.

    This bounds the stable-prefix work, not MLX-LM's own internal history.
    Multiplicities and ordering are retained for frequency/presence penalties.
    """
    import mlx.core as mx

    wrapped = []
    for processor, window in zip(processors, windows, strict=True):
        if window is not None and window <= 0:
            raise ValueError("processor window must be positive or None (unbounded)")
        retained = mx.array(prefix if window is None else prefix[-window:], dtype=mx.int32)

        def apply(tokens, logits, processor=processor, retained=retained, window=window):
            if window is None:
                history = mx.concatenate([retained, tokens])
            elif tokens.size >= window:
                history = tokens[-window:]
            else:
                missing = window - tokens.size
                history = mx.concatenate([retained[-missing:], tokens])
            return processor(history, logits)

        wrapped.append(apply)
    return wrapped


def _stream_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    on_text: Callable[[str], None] | None = None,
    on_prompt: Callable[[str], None] | None = None,
    tools: list[dict[str, Any]] | None = None,
    session: GenerationSession | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, GenerationStats]:
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    started = time.perf_counter()
    template_started = time.monotonic()
    prompt = _render_generation_prompt(tokenizer, messages, tools)
    template_seconds = time.monotonic() - template_started
    if on_prompt is not None:
        on_prompt(prompt)
    sampler = make_sampler(temp=temperature, top_k=top_k)
    processors = make_logits_processors(repetition_penalty=repetition_penalty, repetition_context_size=20)
    first_token_at: float | None = None
    first_visible_at: float | None = None
    final = None
    pieces: list[str] = []
    generated_token_ids: list[int] = []
    full_prompt_tokens: list[int] | None = None
    cached_prompt_tokens = 0
    evaluated_prompt_tokens = 0
    generation_prompt: str | list[int] = prompt
    prompt_cache = None
    tokenizer_encode_seconds = 0.0
    if session is not None:
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        encode_started = time.monotonic()
        full_prompt_tokens = list(
            tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        )
        tokenizer_encode_seconds += time.monotonic() - encode_started
        stable_template_started = time.monotonic()
        stable_prompt = _render_generation_prompt(
            tokenizer,
            messages,
            tools,
            add_generation_prompt=False,
        )
        template_seconds += time.monotonic() - stable_template_started
        encode_started = time.monotonic()
        stable_tokens = list(
            tokenizer.encode(stable_prompt, add_special_tokens=add_special_tokens)
        )
        tokenizer_encode_seconds += time.monotonic() - encode_started
        (
            generation_prompt,
            prompt_cache,
            cached_prompt_tokens,
            evaluated_prompt_tokens,
        ) = session.prepare(
            model,
            full_prompt_tokens,
            stable_tokens,
            should_cancel=should_cancel,
            on_status=on_status,
        )
        if stable_tokens and processors:
            # The only enabled factory option above is repetition_penalty.
            # Pin its history contract instead of introspecting opaque closures.
            processors = _prefix_processors(processors, stable_tokens, [20] * len(processors))
    memory_guard = _mlx_memory_guard_bytes()
    initial_mode = "thinking" if _prompt_prefills_thinking(prompt) else "answer"
    renderer = ThinkingRenderer(initial_mode=initial_mode) if on_text is None else None
    stream_started = time.monotonic()
    progress_started = None
    stream_setup_seconds = 0.0
    suffix_prefill_chunk_seconds = 0.0

    def prefill_progress(processed, total):
        nonlocal progress_started, stream_setup_seconds, suffix_prefill_chunk_seconds
        now = time.monotonic()
        if progress_started is None:
            progress_started = now
            stream_setup_seconds = now - stream_started
        if 0 < processed < total:
            # The final callback includes first decode, so keep it separate.
            suffix_prefill_chunk_seconds = now - progress_started
        if on_status is not None and processed < total:
            if total <= 16:
                on_status("Finalisation du contexte · lancement de la réponse")
            else:
                on_status(f"Lecture du contexte · {processed:,} / {total:,} tokens traités")
        # MLX-LM invokes this between its synchronized prefill chunks as well
        # as before prefill and after the first token. No extra GPU barrier.
        if should_cancel is not None and should_cancel():
            if session is not None:
                session.reset()
            raise GenerationCancelled
        if mx.get_active_memory() >= memory_guard:
            if session is not None:
                session.reset()
            raise MemorySafetyError("prefill arrêté avant saturation de la mémoire unifiée")

    try:
        for response in stream_generate(
            model,
            tokenizer,
            generation_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors,
            prompt_cache=prompt_cache,
            prompt_progress_callback=prefill_progress,
        ):
            if should_cancel is not None and should_cancel():
                raise GenerationCancelled
            if first_token_at is None:
                first_token_at = time.perf_counter()
            if response.text and first_visible_at is None:
                first_visible_at = time.monotonic()
            final = response
            if session is not None:
                generated_token_ids.append(int(response.token))
            pieces.append(response.text)
            if renderer is not None:
                renderer.feed(response.text)
            else:
                on_text(response.text)
            if response.generation_tokens % 32 == 0:
                active_memory = mx.get_active_memory()
                if active_memory >= memory_guard:
                    raise MemorySafetyError(
                        "génération arrêtée avant saturation de la mémoire unifiée "
                        f"({active_memory / 1e9:.1f} / {memory_guard / 1e9:.1f} GB)"
                    )
    finally:
        if renderer is not None:
            renderer.finish()
    finished = time.perf_counter()
    if final is None or first_token_at is None:
        raise RuntimeError("generation returned no response")
    text = "".join(pieces)
    if (
        session is not None
        and full_prompt_tokens is not None
        and prompt_cache is not None
    ):
        session.finish(full_prompt_tokens, generated_token_ids, prompt_cache)
    decode_tokens = max(0, final.generation_tokens - 1)
    decode_seconds = max(0.0, finished - first_token_at)
    decode_tps = decode_tokens / decode_seconds if decode_tokens and decode_seconds else 0.0
    ttft_seconds = first_token_at - started
    prefill_tps = final.prompt_tps
    if session is not None and evaluated_prompt_tokens:
        # Upstream prompt_tps includes processing the first decode token.
        # Combine its suffix interval with our measured stable prefill; never
        # label total request TTFT (template/cache fork included) as prefill.
        suffix_seconds = final.prompt_tokens / final.prompt_tps if final.prompt_tps > 0 else 0.0
        prefill_seconds = session.stable_prefill_seconds + suffix_seconds
        prefill_tps = evaluated_prompt_tokens / prefill_seconds if prefill_seconds else 0.0
    return text, GenerationStats(
        ttft_seconds=ttft_seconds,
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        prompt_tokens=len(full_prompt_tokens) if full_prompt_tokens is not None else final.prompt_tokens,
        generated_tokens=final.generation_tokens,
        peak_memory_gb=final.peak_memory,
        cached_prompt_tokens=cached_prompt_tokens,
        evaluated_prompt_tokens=evaluated_prompt_tokens or final.prompt_tokens,
        template_seconds=template_seconds,
        stable_prefill_seconds=session.stable_prefill_seconds if session else 0.0,
        stable_prefill_tokens=session.stable_prefill_tokens if session else 0,
        cache_fork_seconds=session.cache_fork_seconds if session else 0.0,
        first_visible_text_seconds=(first_visible_at - template_started) if first_visible_at is not None else None,
        cache_lookup_seconds=session.cache_lookup_seconds if session else 0.0,
        tokenizer_encode_seconds=tokenizer_encode_seconds,
        stream_setup_seconds=stream_setup_seconds,
        suffix_prefill_chunk_seconds=suffix_prefill_chunk_seconds,
    )


def _print_stats(stats: GenerationStats) -> None:
    cache = f" · cache {stats.cached_prompt_tokens} tok" if stats.cached_prompt_tokens else ""
    print(
        f"[TTFT {stats.ttft_seconds * 1000:.0f} ms · "
        f"prefill {stats.prefill_tps:.1f} tok/s ({stats.prompt_tokens} tok) · "
        f"decode {stats.decode_tps:.1f} tok/s ({stats.generated_tokens} tok) · "
        f"peak {stats.peak_memory_gb:.2f} GB{cache}]"
    )


def _messages(system: str | None) -> list[dict[str, str]]:
    return [] if system is None else [{"role": "system", "content": system}]


def _assistant_context(response: str) -> str:
    """Keep final answers, not hidden or truncated reasoning, in chat history."""

    if '<channel|>' in response:
        return response.rsplit('<channel|>', 1)[1].strip() or '(La réponse précédente ne contenait pas de conclusion.)'
    if '<|channel>thought' in response:
        return '(La réponse précédente a été interrompue avant sa conclusion.)'
    close = response.rfind("</think>")
    if close >= 0:
        final = response[close + len("</think>") :].strip()
        return final or "(La réponse précédente ne contenait pas de conclusion.)"
    if "<think>" in response:
        return "(La réponse précédente a été interrompue avant sa conclusion.)"
    return response.strip()


def _generate_turn(model, tokenizer, messages, user_text, args, session=None) -> None:
    messages.append({"role": "user", "content": user_text})
    try:
        response, stats = _stream_response(
            model,
            tokenizer,
            messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            session=session,
        )
    except KeyboardInterrupt:
        messages.pop()
        print("\n[génération interrompue]")
        return
    messages.append({"role": "assistant", "content": _cache_context(response)})
    _print_stats(stats)


def _interactive_chat(model, tokenizer, args) -> None:
    messages = _messages(args.system)
    session = GenerationSession()
    print("Chat interactif. Commandes: /clear, /help, /exit")
    while True:
        try:
            user_text = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return
        if user_text == "/clear":
            messages = _messages(args.system)
            session.reset()
            print("Historique effacé.")
            continue
        if user_text == "/help":
            print("/clear efface l'historique · /exit quitte le chat")
            continue
        _generate_turn(model, tokenizer, messages, user_text, args, session)


def _run(args) -> int:
    if args.max_tokens == 0 or args.max_tokens < -1:
        raise RegistryError("--max-tokens must be positive, or -1 for no output limit")
    if args.temperature < 0:
        raise RegistryError("--temperature must be non-negative")
    name, path = resolve_model(args.model)
    print(f"Chargement de {name} sur Metal…", flush=True)
    model, tokenizer, modules, load_seconds, resident_gb = _load_model(path)
    print(f"Prêt · {modules} modules EXL3 · {load_seconds:.2f} s · {resident_gb:.2f} GB résidents")
    if args.prompt is not None:
        messages = _messages(args.system)
        _generate_turn(model, tokenizer, messages, args.prompt, args)
    else:
        _interactive_chat(model, tokenizer, args)
    return 0


def _json_event(event_type: str, **payload: Any) -> None:
    print(
        json.dumps({"type": event_type, **payload}, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


def _model_payload(entry: ModelEntry) -> dict[str, Any]:
    return {**asdict(entry), "size": human_size(entry.size_bytes)}


def _bridge_messages(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        raise TypeError("messages must be an array")
    messages: list[dict[str, str]] = []
    for message in payload:
        if not isinstance(message, dict):
            raise TypeError("each message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("messages require a valid role and string content")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("messages cannot be empty")
    return messages


def _emit_split_fragment(
    phase: str,
    fragment: str,
    *,
    request_id: str,
    tool_filter: ToolCallStreamFilter | None,
) -> None:
    if phase == "thinking" or tool_filter is None:
        _json_event("delta", request_id=request_id, phase=phase, text=fragment)
        return
    for visible in tool_filter.feed(fragment):
        _json_event("delta", request_id=request_id, phase="answer", text=visible)


def _bridge_generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    request_id: str,
    max_tokens: int,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    mcp: MCPManager,
    session: GenerationSession | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    dialogue = list(messages)
    session = session or GenerationSession()
    chat_tools = mcp.chat_tools
    for tool_round in range(5):
        _json_event(
            "generation_status", request_id=request_id, phase="prefill",
            text="Lecture des résultats MCP" if tool_round else "Préparation du contexte",
        )
        splitter = ThinkingSplitter()
        tool_filter = ToolCallStreamFilter() if chat_tools else None

        def emit_text(
            text: str,
            event_splitter: ThinkingSplitter = splitter,
            event_id: str = request_id,
            event_filter: ToolCallStreamFilter | None = tool_filter,
        ) -> None:
            for phase, fragment in event_splitter.feed(text):
                _emit_split_fragment(
                    phase,
                    fragment,
                    request_id=event_id,
                    tool_filter=event_filter,
                )

        response, stats = _stream_response(
            model,
            tokenizer,
            dialogue,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            on_text=emit_text,
            on_prompt=splitter.configure_for_prompt,
            tools=chat_tools,
            session=session,
            should_cancel=should_cancel,
            on_status=lambda label: _json_event(
                "generation_status", request_id=request_id, phase="prefill", text=label,
            ),
        )
        for phase, fragment in splitter.finish():
            _emit_split_fragment(
                phase,
                fragment,
                request_id=request_id,
                tool_filter=tool_filter,
            )
        if tool_filter is not None:
            for visible in tool_filter.finish():
                _json_event("delta", request_id=request_id, phase="answer", text=visible)

        calls = _parse_tool_calls(response) if chat_tools else []
        if not calls:
            _json_event(
                "complete",
                request_id=request_id,
                assistant_context=_without_tool_calls(response),
                cache_context=_cache_context(response),
                stats=asdict(stats),
            )
            return
        if tool_round == 4:
            raise MCPError("the model exceeded the limit of 5 consecutive MCP tool rounds")

        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": _without_tool_calls(response),
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in calls
            ],
        }
        reasoning = _reasoning_context(response)
        if reasoning:
            assistant_message["reasoning_content"] = reasoning
        dialogue.append(assistant_message)

        for call in calls:
            tool = mcp.tools.get(call.name)
            call_id = f"mcp-{tool_round}-{len(dialogue)}-{call.name}"
            _json_event(
                "tool_start",
                request_id=request_id,
                tool_call_id=call_id,
                tool_name=call.name,
                server_name=tool.server if tool else None,
            )
            result = mcp.call(call.name, call.arguments)
            _json_event(
                "tool_result",
                request_id=request_id,
                tool_call_id=call_id,
                tool_name=call.name,
                server_name=tool.server if tool else None,
                text=result.text[:4_000],
                is_error=result.is_error,
            )
            dialogue.append(
                {
                    "role": "tool",
                    "name": call.name,
                    "content": result.text,
                }
            )


def _bridge(args) -> int:
    name, path = resolve_model(args.model)
    _json_event("loading", model=name)
    model, tokenizer, modules, load_seconds, resident_gb = _load_model(path)
    sessions = GenerationSessionPool()
    cancel_requested = threading.Event()
    previous_cancel_handler = signal.getsignal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, lambda _signum, _frame: cancel_requested.set())
    # Off until the desktop explicitly restores/enables the user's preference.
    mcp = MCPManager([])
    try:
        _json_event(
            "ready",
            model=name,
            modules=modules,
            load_seconds=load_seconds,
            resident_gb=resident_gb,
            mcp_servers=mcp.connected_server_count,
            mcp_tools=len(mcp.tools),
            mcp_errors=mcp.errors,
        )

        for line in sys.stdin:
            if not line.strip():
                continue
            request_id: str | None = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise TypeError("request must be an object")
                request_type = request.get("type")
                request_id = str(request.get("request_id", ""))
                if request_type == "shutdown":
                    return 0
                if request_type == "ping":
                    _json_event("pong", request_id=request_id)
                    continue
                if request_type == "set_mcp":
                    mcp.set_enabled(request.get("enabled", False), refresh=True)
                    _json_event("mcp_status", mcp_servers=mcp.connected_server_count,
                                mcp_tools=len(mcp.tools), mcp_errors=mcp.errors)
                    continue
                if request_type != "generate":
                    raise ValueError(f"unsupported request type: {request_type!r}")

                messages = _bridge_messages(request.get("messages"))
                mcp.set_enabled(request.get("mcp_enabled", False))
                conversation_id = str(request.get("conversation_id") or "default")
                max_tokens = int(request.get("max_tokens", -1))
                temperature = float(request.get("temperature", 0.2))
                top_k = int(request.get("top_k", 80))
                repetition_penalty = float(request.get("repetition_penalty", 1.05))
                if max_tokens == 0 or max_tokens < -1:
                    raise ValueError("max_tokens must be positive, or -1 for no output limit")
                if temperature < 0:
                    raise ValueError("temperature must be non-negative")

                _bridge_generate(
                    model,
                    tokenizer,
                    messages,
                    request_id=request_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    mcp=mcp,
                    session=sessions.acquire(conversation_id),
                    should_cancel=cancel_requested.is_set,
                )
                sessions.prune(conversation_id)
            except GenerationCancelled:
                _json_event("cancelled", request_id=request_id)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                _json_event("error", request_id=request_id, message=str(error))
            except Exception as error:  # noqa: BLE001 - keep the resident model available
                _json_event(
                    "error",
                    request_id=request_id,
                    message=f"{type(error).__name__}: {error}",
                )
            finally:
                cancel_requested.clear()
    finally:
        signal.signal(signal.SIGUSR1, previous_cancel_handler)
        mcp.close()
    return 0


def _download_hugging_face_model(
    repo: str,
    *,
    revision: str | None,
    name: str | None,
    directory: Path | None,
    force: bool,
) -> ModelEntry:
    """Download into app-owned storage and register only a complete model."""

    from huggingface_hub import snapshot_download

    repo = repo.strip()
    if not repo or repo.startswith("/") or repo.endswith("/"):
        raise RegistryError("invalid Hugging Face repository name")
    model_name = (name or repo.rsplit("/", 1)[-1]).strip()
    validate_model_name(model_name)
    # The registry accepts slashes for names supplied by advanced CLI users,
    # but a managed download must never turn them into directory traversal.
    directory_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip(".-")
    if not directory_name:
        raise RegistryError("invalid local model name")
    destination = (
        directory.expanduser()
        if directory is not None
        else managed_models_path() / directory_name
    ).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = snapshot_download(
            repo_id=repo,
            revision=revision or None,
            local_dir=destination,
            force_download=force,
        )
    except Exception as error:
        raise RegistryError(f"Hugging Face download failed: {error}") from error
    return register_model(model_name, downloaded, force=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"list", "ls"}:
            entries = load_registry()
            if args.json:
                print(
                    json.dumps(
                        [_model_payload(entry) for entry in sorted(
                            entries.values(), key=lambda item: item.name.lower()
                        )],
                        ensure_ascii=False,
                    )
                )
            elif not entries:
                print("Aucun modèle enregistré. Utilise: mlxl3 register NOM CHEMIN")
            else:
                print(_format_models(entries))
            return 0
        if args.command in {"register", "add"}:
            entry = register_model(args.name, args.path, force=args.force)
            print(f"Modèle {entry.name!r} enregistré ({human_size(entry.size_bytes)}).")
            return 0
        if args.command == "download":
            entry = _download_hugging_face_model(
                args.repo,
                revision=args.revision,
                name=args.name,
                directory=args.directory,
                force=args.force,
            )
            if args.json:
                print(json.dumps(_model_payload(entry), ensure_ascii=False))
            else:
                print(
                    f"Modèle {entry.name!r} téléchargé et enregistré "
                    f"({human_size(entry.size_bytes)})."
                )
            return 0
        if args.command in {"remove", "rm"}:
            entry = remove_model(args.name)
            print(f"Modèle {entry.name!r} retiré du registre; ses fichiers sont conservés.")
            return 0
        if args.command == "mcp":
            if args.mcp_command == "check":
                manager = MCPManager([])
                try:
                    manager.set_enabled(True)
                    result = {"connected": manager.connected_server_count,
                              "tools": list(manager.tools), "errors": manager.errors}
                    print(json.dumps(result, ensure_ascii=False, indent=None if args.json else 2))
                    return 1 if manager.errors else 0
                finally:
                    manager.close()
            if args.mcp_command in {"list", "ls"}:
                servers = load_mcp_servers()
                if args.json:
                    print(
                        json.dumps(
                            [asdict(server) for server in servers],
                            ensure_ascii=False,
                        )
                    )
                else:
                    print(_format_mcp_servers())
                return 0
            if args.mcp_command == "config":
                print(ensure_mcp_config())
                return 0
            if args.mcp_command == "add":
                server = add_mcp_server(
                    args.name,
                    args.server_command,
                    args.server_args,
                )
                print(f"Serveur MCP {server.name!r} enregistré.")
                return 0
            if args.mcp_command in {"remove", "rm"}:
                remove_mcp_server(args.name)
                print(f"Serveur MCP {args.name!r} retiré.")
                return 0
            raise AssertionError(f"unknown MCP command {args.mcp_command}")
        if args.command == "run":
            return _run(args)
        if args.command == "bridge":
            return _bridge(args)
        raise AssertionError(f"unknown command {args.command}")
    except (MCPError, RegistryError, FileNotFoundError, KeyError) as error:
        print(f"Erreur: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
