"""Ollama-style local model registry and streaming terminal chat."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mlxl3.registry import (
    ModelEntry,
    RegistryError,
    human_size,
    load_registry,
    register_model,
    remove_model,
    resolve_model,
)


@dataclass(frozen=True)
class GenerationStats:
    ttft_seconds: float
    prefill_tps: float
    decode_tps: float
    prompt_tokens: int
    generated_tokens: int
    peak_memory_gb: float


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
            marker = self._CLOSE if self.mode == "thinking" else self._OPEN
            position = self.buffer.find(marker)
            if position >= 0:
                fragment = self._fragment(self.buffer[:position])
                if fragment is not None:
                    fragments.append(fragment)
                self.buffer = self.buffer[position + len(marker) :]
                self.mode = "answer" if self.mode == "thinking" else "thinking"
                self.drop_leading_newline = True
                continue
            keep = self._partial_marker_length(self.buffer, marker)
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

    return prompt.rstrip().endswith(ThinkingSplitter._OPEN)


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

    remove = commands.add_parser("remove", aliases=["rm"], help="remove a registry entry")
    remove.add_argument("name")

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


def _load_model(model_path: Path):
    import mlx.core as mx
    from mlx_lm.utils import load_tokenizer

    from mlxl3.checkpoint import load_exl3_model

    started = time.perf_counter()
    model, _, report = load_exl3_model(model_path, lazy=False)
    tokenizer = load_tokenizer(str(model_path))
    return (
        model,
        tokenizer,
        len(report.replaced),
        time.perf_counter() - started,
        mx.get_active_memory() / 1e9,
    )


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
) -> tuple[str, GenerationStats]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if on_prompt is not None:
        on_prompt(prompt)
    sampler = make_sampler(temp=temperature, top_k=top_k)
    processors = make_logits_processors(repetition_penalty=repetition_penalty)
    started = time.perf_counter()
    first_token_at: float | None = None
    final = None
    pieces: list[str] = []
    initial_mode = "thinking" if _prompt_prefills_thinking(prompt) else "answer"
    renderer = ThinkingRenderer(initial_mode=initial_mode) if on_text is None else None
    try:
        for response in stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=processors,
        ):
            if first_token_at is None:
                first_token_at = time.perf_counter()
            final = response
            pieces.append(response.text)
            if renderer is not None:
                renderer.feed(response.text)
            else:
                on_text(response.text)
    finally:
        if renderer is not None:
            renderer.finish()
    finished = time.perf_counter()
    if final is None or first_token_at is None:
        raise RuntimeError("generation returned no response")
    text = "".join(pieces)

    decode_tokens = max(0, final.generation_tokens - 1)
    decode_seconds = max(0.0, finished - first_token_at)
    decode_tps = decode_tokens / decode_seconds if decode_tokens and decode_seconds else 0.0
    return text, GenerationStats(
        ttft_seconds=first_token_at - started,
        prefill_tps=final.prompt_tps,
        decode_tps=decode_tps,
        prompt_tokens=final.prompt_tokens,
        generated_tokens=final.generation_tokens,
        peak_memory_gb=final.peak_memory,
    )


def _print_stats(stats: GenerationStats) -> None:
    print(
        f"[TTFT {stats.ttft_seconds * 1000:.0f} ms · "
        f"prefill {stats.prefill_tps:.1f} tok/s ({stats.prompt_tokens} tok) · "
        f"decode {stats.decode_tps:.1f} tok/s ({stats.generated_tokens} tok) · "
        f"peak {stats.peak_memory_gb:.2f} GB]"
    )


def _messages(system: str | None) -> list[dict[str, str]]:
    return [] if system is None else [{"role": "system", "content": system}]


def _assistant_context(response: str) -> str:
    """Keep final answers, not hidden or truncated reasoning, in chat history."""

    close = response.rfind("</think>")
    if close >= 0:
        final = response[close + len("</think>") :].strip()
        return final or "(La réponse précédente ne contenait pas de conclusion.)"
    if "<think>" in response:
        return "(La réponse précédente a été interrompue avant sa conclusion.)"
    return response.strip()


def _generate_turn(model, tokenizer, messages, user_text, args) -> None:
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
        )
    except KeyboardInterrupt:
        messages.pop()
        print("\n[génération interrompue]")
        return
    messages.append({"role": "assistant", "content": _assistant_context(response)})
    _print_stats(stats)


def _interactive_chat(model, tokenizer, args) -> None:
    messages = _messages(args.system)
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
            print("Historique effacé.")
            continue
        if user_text == "/help":
            print("/clear efface l'historique · /exit quitte le chat")
            continue
        _generate_turn(model, tokenizer, messages, user_text, args)


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


def _bridge(args) -> int:
    name, path = resolve_model(args.model)
    _json_event("loading", model=name)
    model, tokenizer, modules, load_seconds, resident_gb = _load_model(path)
    _json_event(
        "ready",
        model=name,
        modules=modules,
        load_seconds=load_seconds,
        resident_gb=resident_gb,
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
            if request_type != "generate":
                raise ValueError(f"unsupported request type: {request_type!r}")

            messages = _bridge_messages(request.get("messages"))
            max_tokens = int(request.get("max_tokens", -1))
            temperature = float(request.get("temperature", 0.2))
            top_k = int(request.get("top_k", 80))
            repetition_penalty = float(request.get("repetition_penalty", 1.05))
            if max_tokens == 0 or max_tokens < -1:
                raise ValueError("max_tokens must be positive, or -1 for no output limit")
            if temperature < 0:
                raise ValueError("temperature must be non-negative")

            splitter = ThinkingSplitter()

            def emit_text(
                text: str,
                event_splitter: ThinkingSplitter = splitter,
                event_id: str | None = request_id,
            ) -> None:
                for phase, fragment in event_splitter.feed(text):
                    _json_event(
                        "delta",
                        request_id=event_id,
                        phase=phase,
                        text=fragment,
                    )

            response, stats = _stream_response(
                model,
                tokenizer,
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                on_text=emit_text,
                on_prompt=splitter.configure_for_prompt,
            )
            for phase, fragment in splitter.finish():
                _json_event(
                    "delta",
                    request_id=request_id,
                    phase=phase,
                    text=fragment,
                )
            _json_event(
                "complete",
                request_id=request_id,
                assistant_context=_assistant_context(response),
                stats=asdict(stats),
            )
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            _json_event("error", request_id=request_id, message=str(error))
        except Exception as error:  # noqa: BLE001 - report failed turns without killing the bridge
            _json_event("error", request_id=request_id, message=f"{type(error).__name__}: {error}")
    return 0


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
        if args.command in {"remove", "rm"}:
            entry = remove_model(args.name)
            print(f"Modèle {entry.name!r} retiré du registre; ses fichiers sont conservés.")
            return 0
        if args.command == "run":
            return _run(args)
        if args.command == "bridge":
            return _bridge(args)
        raise AssertionError(f"unknown command {args.command}")
    except (RegistryError, FileNotFoundError, KeyError) as error:
        print(f"Erreur: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
