"""Paired CLI streaming on a shared, reference-prefilled Gemma prompt cache."""

import argparse
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from mlxl3 import attention
from mlxl3.cli import GenerationSession, _load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()
    attention._MODE = "grouped"
    model, tokenizer, *_ = _load_model(args.model)
    attention._MODE = "off"
    attention._MIN_LENGTH = 0  # also exercise short-context numeric validation
    text = "La lumière du Soleil est diffusée par les molécules de notre atmosphère. "
    messages = [
        {
            "role": "user",
            "content": text * args.context
            + " Explique pourquoi le ciel est bleu, puis pourquoi un coucher de soleil est rouge.",
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # Preserve the end question and assistant header, bound the measured prefix.
    prompt = prompt[:1] + prompt[-(args.context - 1) :]
    saved = make_prompt_cache(model)
    start = time.perf_counter()
    for position in range(0, len(prompt) - 1, 128):
        chunk = prompt[position : min(position + 128, len(prompt) - 1)]
        model(mx.array([chunk]), cache=saved)
        mx.eval([c.state for c in saved])
        mx.clear_cache()
        if position % 512 == 0:
            print(
                json.dumps(
                    {
                        "prefill_tokens": position + len(chunk),
                        "seconds": time.perf_counter() - start,
                    }
                ),
                flush=True,
            )
    print(json.dumps({"prefill_complete": len(prompt) - 1}), flush=True)

    def run(mode, count):
        attention._MODE = mode
        cache = GenerationSession._clone_cache(saved)
        tokens = []
        pieces = []
        last = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt[-1:],
            prompt_cache=cache,
            max_tokens=count,
            sampler=make_sampler(temp=0),
        ):
            tokens.append(response.token)
            pieces.append(response.text)
            last = response
        return last.generation_tps, tokens, "".join(pieces)

    modes = ("off", "grouped", "matrix")
    for mode in modes:
        run(mode, 8)
    speeds = {mode: [] for mode in modes}
    all_equal = True
    for pair in range(args.pairs):
        rows = {}
        for mode in modes if pair % 2 == 0 else modes[::-1]:
            speed, tokens, text = run(mode, args.tokens)
            speeds[mode].append(speed)
            rows[mode] = {"tok_s": speed, "tokens": tokens, "text": text}
        equality = {mode: rows[mode]["tokens"] == rows["off"]["tokens"] for mode in modes}
        all_equal &= all(equality.values())
        print(
            json.dumps(
                {
                    "pair": pair,
                    "speeds": {m: rows[m]["tok_s"] for m in modes},
                    "tokens_equal": equality,
                    "token_counts": {m: len(rows[m]["tokens"]) for m in modes},
                }
            ),
            flush=True,
        )
        if not all(equality.values()):
            print(json.dumps({"mismatch": rows}, ensure_ascii=False), flush=True)
            break
    print(
        json.dumps(
            {
                "all_tokens_equal": all_equal,
                "median_tok_s": {m: statistics.median(v) for m, v in speeds.items()},
                "paired_gain_percent": {
                    m: statistics.median(
                        [(b / a - 1) * 100 for a, b in zip(speeds["off"], speeds[m])]
                    )
                    for m in modes
                },
                "peak_gb": mx.get_peak_memory() / 1e9,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
