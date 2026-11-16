"""Hugging Face chat templates rendered without ``transformers``.

ExLlamaV3's ``Tokenizer.hf_chat_template`` needs ``transformers``. The worker
renders the checkpoint's own Jinja template with the same sandboxed Jinja
settings ``transformers`` uses, then tokenizes the text with special tokens
enabled and no implicit BOS (templates insert their own BOS when they need it).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class TemplateError(RuntimeError):
    pass


def _token_text(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    return None


def _select(template) -> str | None:
    if isinstance(template, str):
        return template
    if isinstance(template, list):
        named = {t.get("name"): t.get("template") for t in template if isinstance(t, dict)}
        chosen = named.get("default") or next(iter(named.values()), None)
        return chosen if isinstance(chosen, str) else None
    return None


def load(model_dir: str | Path) -> tuple[str, dict]:
    """Return (template source, special tokens) from a checkpoint folder."""
    root = Path(model_dir)
    tokenizer_config = {}
    path = root / "tokenizer_config.json"
    if path.is_file():
        try:
            tokenizer_config = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise TemplateError(f"unreadable tokenizer_config.json: {error}") from None
    special = {
        "bos_token": _token_text(tokenizer_config.get("bos_token")) or "",
        "eos_token": _token_text(tokenizer_config.get("eos_token")) or "",
    }
    jinja_file = root / "chat_template.jinja"
    if jinja_file.is_file():
        return jinja_file.read_text(encoding="utf-8"), special
    json_file = root / "chat_template.json"
    if json_file.is_file():
        try:
            source = _select(json.loads(json_file.read_text(encoding="utf-8")).get("chat_template"))
        except (OSError, json.JSONDecodeError, AttributeError) as error:
            raise TemplateError(f"unreadable chat_template.json: {error}") from None
        if source:
            return source, special
    source = _select(tokenizer_config.get("chat_template"))
    if source:
        return source, special
    raise TemplateError("the checkpoint has no chat template (chat_template.jinja, "
                        "chat_template.json or tokenizer_config.json)")


def _raise(message):
    raise TemplateError(str(message))


def render(source: str, special: dict, messages: list, enable_thinking: bool | None = None) -> str:
    try:
        from jinja2.ext import loopcontrols
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        raise TemplateError("jinja2 is required to render chat templates") from None

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = _raise
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    try:
        template = env.from_string(source)
        context = {"messages": messages, "add_generation_prompt": True, **special}
        if enable_thinking is not None:
            context["enable_thinking"] = enable_thinking
        return template.render(**context)
    except TemplateError:
        raise
    except Exception as error:  # Jinja syntax/runtime errors come in many classes
        raise TemplateError(f"chat template failed: {error}") from None
