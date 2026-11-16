"""Inference backends behind the bridge.

``Exl3Backend`` drives upstream ExLlamaV3. The same Python code serves the
CUDA wheel on Windows and the ROCm wheel inside WSL2: ExLlamaV3 selects its
CUDA or HIP kernels from the installed torch build. ``FakeBackend`` is a
deterministic stand-in used by tests of the bridge and of the Windows app's
process supervision; it never touches a GPU.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable, Iterator

from . import chat_template
from .protocol import is_greedy

PAGE_SIZE = 256
DEFAULT_CONTEXT = 16384
MAX_CONTEXT = 1 << 20


class BackendError(RuntimeError):
    pass


def resolve_context(requested: int, model_limit: int) -> int:
    """Cache size in tokens: 0 picks min(model limit, 16384); always a whole number of pages."""
    if model_limit < PAGE_SIZE:
        model_limit = PAGE_SIZE
    limit = min(model_limit, MAX_CONTEXT)
    if requested < 0:
        raise BackendError("context length must be 0 (automatic) or positive")
    if requested == 0:
        requested = min(limit, DEFAULT_CONTEXT)
    if requested > limit:
        raise BackendError(f"context length must be at most {limit} tokens for this model")
    pages = -(-requested // PAGE_SIZE)
    return pages * PAGE_SIZE


def new_token_budget(max_tokens: int, prompt_tokens: int, context: int) -> int:
    room = context - prompt_tokens
    if room <= 0:
        raise BackendError(
            f"the conversation uses {prompt_tokens} tokens and the context holds {context}; "
            "start a new conversation or raise the context length")
    return room if max_tokens == -1 else min(max_tokens, room)


def is_exl3_checkpoint(model_dir: str | Path) -> bool:
    import json
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    quant = config.get("quantization_config") or {}
    return isinstance(quant, dict) and str(quant.get("quant_method", "")).lower() == "exl3"


class Exl3Backend:
    def __init__(self, model_dir: str, context_length: int):
        if not is_exl3_checkpoint(model_dir):
            raise BackendError("config.json does not declare quant_method \"exl3\"")
        import torch
        import exllamav3
        from exllamav3 import Cache, Config, Generator, Model, Tokenizer
        from exllamav3.version import __version__ as exl3_version

        self._exl3_version = exl3_version
        if not torch.cuda.is_available():
            raise BackendError("torch sees no GPU (CUDA/HIP device count is 0)")
        self._torch = torch
        self._exl3 = exllamav3
        self.template, self.special = chat_template.load(model_dir)
        self.config = Config.from_directory(model_dir)
        self.model_limit = int(getattr(self.config, "max_position_embeddings", 0) or DEFAULT_CONTEXT)
        self.context = resolve_context(context_length, self.model_limit)
        self.model = Model.from_config(self.config)
        self.cache = Cache(self.model, max_num_tokens=self.context)
        self.model.load()
        self.tokenizer = Tokenizer.from_config(self.config)
        self.generator = Generator(model=self.model, cache=self.cache, tokenizer=self.tokenizer)
        # Vision tower, for VL checkpoints: loaded on the first image so text-only use keeps
        # its VRAM.
        self.vision_capable = "vision" in getattr(self.config, "model_classes", {})
        self.vision = None
        stops = list(getattr(self.config, "eos_token_id_list", None) or [])
        if not stops and self.tokenizer.eos_token_id is not None:
            stops = [self.tokenizer.eos_token_id]
        self.stop_conditions = [s for s in stops if s is not None]

    def info(self) -> dict:
        torch = self._torch
        device = torch.cuda.get_device_properties(0)
        return {
            "backend": "rocm" if torch.version.hip else "cuda",
            "device": device.name,
            "device_memory_gb": round(device.total_memory / 1e9, 2),
            "torch_version": torch.__version__,
            "vision": self.vision_capable,
            "gpu_runtime": torch.version.hip or torch.version.cuda,
            "exllamav3_version": self._exl3_version,
            "context_limit": self.context,
            "model_context_limit": self.model_limit,
            "resident_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        }

    def _sampler(self, request: dict):
        from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
        rep = request["repetition_penalty"]
        if is_greedy(request) and rep == 1.0:
            return ArgmaxSampler()
        if is_greedy(request):
            return ComboSampler(rep_p=rep, temperature=1.0, top_k=1)
        return ComboSampler(rep_p=rep, temperature=request["temperature"], top_k=request["top_k"],
                            top_p=request["top_p"], min_p=request["min_p"])

    def _embed_images(self, messages: list[dict]) -> tuple[list[dict], list]:
        """Replace each message's images by the placeholders ExLlamaV3 tokenizes as embeddings."""
        if not any(m.get("images") for m in messages):
            return messages, []
        if not self.vision_capable:
            raise BackendError("this model has no vision component; images need a vision-language model")
        import io
        from exllamav3 import Model
        from PIL import Image
        if self.vision is None:
            vision = Model.from_config(self.config, component="vision")
            vision.load()
            self.vision = vision
        embeddings, out = [], []
        for message in messages:
            images = message.get("images") or []
            aliases = []
            for data in images:
                image = Image.open(io.BytesIO(data))
                image.load()
                embedding = self.vision.get_image_embeddings(tokenizer=self.tokenizer, image=image.convert("RGB"))
                embeddings.append(embedding)
                aliases.append(embedding.text_alias)
            content = message["content"]
            if aliases:
                content = "\n".join(aliases) + "\n" + content
            entry = {key: value for key, value in message.items() if key != "images"}
            entry["content"] = content
            out.append(entry)
        return out, embeddings

    def generate(self, request: dict, should_stop: Callable[[], bool]) -> Iterator[dict]:
        from exllamav3 import Job
        messages, embeddings = self._embed_images(request["messages"])
        text = chat_template.render(self.template, self.special, messages,
                                    request.get("enable_thinking"), request.get("tools"))
        extra = {"embeddings": embeddings} if embeddings else {}
        input_ids = self.tokenizer.encode(text, add_bos=False, encode_special_tokens=True, **extra)
        prompt_tokens = int(input_ids.shape[-1])
        budget = new_token_budget(request["max_tokens"], prompt_tokens, self.context)
        job = Job(input_ids=input_ids, max_new_tokens=budget, sampler=self._sampler(request),
                  seed=request.get("seed"), stop_conditions=self.stop_conditions,
                  identifier=request["request_id"], **extra)
        self.generator.enqueue(job)
        started = time.perf_counter()
        try:
            while self.generator.num_remaining_jobs():
                if should_stop():
                    self.generator.cancel(job)
                    yield {"cancelled": True}
                    return
                for result in self.generator.iterate():
                    if result.get("stage") != "streaming":
                        continue
                    if result.get("text"):
                        yield {"text": result["text"]}
                    if result.get("eos"):
                        yield {"stats": {
                            "prompt_tokens": int(result.get("prompt_tokens", prompt_tokens)),
                            "cached_tokens": int(result.get("cached_tokens", 0) or 0),
                            "generated_tokens": int(result.get("new_tokens", 0)),
                            "prefill_seconds": float(result.get("time_prefill", 0.0)),
                            "decode_seconds": float(result.get("time_generate", 0.0)),
                            "wall_seconds": time.perf_counter() - started,
                            "stop_reason": str(result.get("eos_reason", "")),
                        }}
                        return
        except BaseException:
            if self.generator.num_remaining_jobs():
                self.generator.cancel(job)
            raise

    def close(self) -> None:
        close = getattr(self.generator, "close", None)
        if close:
            close()
        self.model.unload()


class FakeBackend:
    """Deterministic backend for tests: echoes the last user message word by word."""

    def __init__(self, model_dir: str, context_length: int):
        self.context = resolve_context(context_length, 4096)
        self.delay = float(os.environ.get("MLXL3_FAKE_DELAY", "0"))
        if os.environ.get("MLXL3_FAKE_LOAD_ERROR"):
            raise BackendError(os.environ["MLXL3_FAKE_LOAD_ERROR"])
        if os.environ.get("MLXL3_FAKE_NOISE"):
            # Simulates library progress output on both Python and native stdout.
            print("loading layers 1/1")
            os.write(1, b"native stdout noise\n")

    def info(self) -> dict:
        return {"backend": "fake", "device": "none", "device_memory_gb": 0.0,
                "torch_version": "none", "gpu_runtime": "none", "exllamav3_version": "none", "vision": True,
                "context_limit": self.context, "model_context_limit": 4096, "resident_gb": 0.0}

    def generate(self, request: dict, should_stop: Callable[[], bool]) -> Iterator[dict]:
        final = next(m for m in reversed(request["messages"]) if m["role"] != "assistant")
        last = final["content"]
        if final.get("images"):
            last = f"[{len(final['images'])} image(s)] {last}"
        if last == "__raise__":
            raise BackendError("fake backend failure")
        words = last.split() or ["(empty)"]
        prompt_tokens = sum(len(m["content"].split()) + 4 for m in request["messages"])
        budget = new_token_budget(request["max_tokens"], prompt_tokens, self.context)
        produced = 0
        for index, word in enumerate(words[:budget]):
            if should_stop():
                yield {"cancelled": True}
                return
            if self.delay:
                time.sleep(self.delay)
            produced += 1
            yield {"text": word if index == 0 else " " + word}
        yield {"stats": {"prompt_tokens": prompt_tokens, "cached_tokens": 0,
                         "generated_tokens": produced, "prefill_seconds": 0.0,
                         "decode_seconds": 0.0, "wall_seconds": 0.0,
                         "stop_reason": "eos" if produced == len(words) else "max_new_tokens"}}

    def close(self) -> None:
        pass
