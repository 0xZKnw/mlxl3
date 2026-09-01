"""Ollama-style local model registry and streaming terminal chat."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
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


class ThinkingRenderer:
    """Incrementally render ``<think>`` blocks separately from final text."""

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self, *, stream=None, color: bool | None = None):
        self.stream = sys.stdout if stream is None else stream
        self.color = (
            bool(getattr(self.stream, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ
            if color is None
            else color
        )
        self.mode = "answer"
        self.section: str | None = None
        self.buffer = ""
        self.last_was_newline = True
        self.drop_leading_newline = False

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()
        if text:
            self.last_was_newline = text.endswith("\n")

    def _start_section(self) -> None:
        if self.section == self.mode:
            return
        if self.section is not None and not self.last_was_newline:
            self._write("\n")
        label = "Réflexion" if self.mode == "thinking" else "Réponse"
        if self.color:
            style = "\033[1;35m" if self.mode == "thinking" else "\033[1;36m"
            self._write(f"{style}── {label} ──\033[0m\n")
        else:
            self._write(f"── {label} ──\n")
        self.section = self.mode

    def _emit(self, text: str) -> None:
        if self.drop_leading_newline and text:
            if text.startswith("\r\n"):
                text = text[2:]
            elif text.startswith("\n"):
                text = text[1:]
            self.drop_leading_newline = False
        if not text:
            return
        self._start_section()
        if self.color and self.mode == "thinking":
            self._write(f"\033[35m{text}\033[0m")
        else:
            self._write(text)

    @staticmethod
    def _partial_marker_length(text: str, marker: str) -> int:
        limit = min(len(text), len(marker) - 1)
        for length in range(limit, 0, -1):
            if text.endswith(marker[:length]):
                return length
        return 0

    def feed(self, text: str) -> None:
        self.buffer += text
        while self.buffer:
            marker = self._CLOSE if self.mode == "thinking" else self._OPEN
            position = self.buffer.find(marker)
            if position >= 0:
                self._emit(self.buffer[:position])
                self.buffer = self.buffer[position + len(marker) :]
                self.mode = "answer" if self.mode == "thinking" else "thinking"
                self.drop_leading_newline = True
                continue
            keep = self._partial_marker_length(self.buffer, marker)
            emit_until = len(self.buffer) - keep
            self._emit(self.buffer[:emit_until])
            self.buffer = self.buffer[emit_until:]
            break

    def finish(self) -> None:
        self._emit(self.buffer)
        self.buffer = ""
        if self.section is not None and not self.last_was_newline:
            self._write("\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlxl3",
        description="Run registered EXL3 models locally on Apple Metal.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", aliases=["ls"], help="list registered local models")

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
) -> tuple[str, GenerationStats]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    sampler = make_sampler(temp=temperature, top_k=top_k)
    processors = make_logits_processors(repetition_penalty=repetition_penalty)
    started = time.perf_counter()
    first_token_at: float | None = None
    final = None
    pieces: list[str] = []
    renderer = ThinkingRenderer()
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
            renderer.feed(response.text)
    finally:
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
    if args.max_tokens < 1:
        raise RegistryError("--max-tokens must be positive")
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"list", "ls"}:
            entries = load_registry()
            if not entries:
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
        raise AssertionError(f"unknown command {args.command}")
    except (RegistryError, FileNotFoundError, KeyError) as error:
        print(f"Erreur: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
