"""Reproducible greedy LFM2.5 generation benchmark for MLXL3 and MLX 8-bit."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.utils import load_tokenizer

from mlxl3.checkpoint import load_exl3_model


def chat_prompt(tokenizer, text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def run_generation(model, tokenizer, prompt: str, max_tokens: int):
    final = None
    pieces = []
    for response in stream_generate(
            model,
            tokenizer,
            chat_prompt(tokenizer, prompt),
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
        ):
        final = response
        pieces.append(response.text)
    if final is None:
        raise RuntimeError('generation returned no responses')
    return final, "".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=("mlxl3", "mlx8"))
    parser.add_argument("model", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--prompt",
        default="Explique en français, en trois points concis, pourquoi le ciel est bleu.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    if args.engine == "mlxl3":
        model, _, report = load_exl3_model(args.model, lazy=False)
        tokenizer = load_tokenizer(str(args.model))
        module_count = len(report.replaced)
    else:
        model, tokenizer = load(str(args.model), lazy=False)
        module_count = None
    load_seconds = time.perf_counter() - started

    # Compile both the prefill and decode graphs before collecting warm stats.
    run_generation(model, tokenizer, "Dis simplement bonjour.", 8)
    runs = []
    texts = []
    for _ in range(args.repeats):
        mx.clear_cache()
        mx.reset_peak_memory()
        final, text = run_generation(model, tokenizer, args.prompt, args.max_tokens)
        texts.append(text)
        runs.append(
            {
                "prompt_tokens": final.prompt_tokens,
                "prompt_tps": final.prompt_tps,
                "generation_tokens": final.generation_tokens,
                "generation_tps": final.generation_tps,
                "peak_memory_gb": final.peak_memory,
                "finish_reason": final.finish_reason,
            }
        )
    print(
        json.dumps(
            {
                "engine": args.engine,
                "model": str(args.model),
                "load_seconds": load_seconds,
                "exl3_modules": module_count,
                "repeats": args.repeats,
                "median_prompt_tps": statistics.median(run["prompt_tps"] for run in runs),
                "median_generation_tps": statistics.median(run["generation_tps"] for run in runs),
                "median_peak_memory_gb": statistics.median(run["peak_memory_gb"] for run in runs),
                "active_memory_gb": mx.get_active_memory() / 1e9,
                "deterministic_text": len(set(texts)) == 1,
                "runs": runs,
                "text": texts[0],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
