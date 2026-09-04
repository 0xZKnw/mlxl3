"""Paired greedy streaming benchmark with explicit compiled-graph refresh."""

import argparse
import json
import statistics
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from mlxl3.cli import _load_model
from mlxl3.kernels import qmv


def refresh(model):
    for _, module in model.named_modules():
        if hasattr(module, "_mlxl3_compiled_decode"):
            from mlx_lm.models.cache import ArraysCache

            original = module._mlxl3_original_call

            def run(x, state, call=original):
                cache = ArraysCache(len(state[0]))
                cache.state = state
                output = call(x, mask=None, cache=cache)
                return output, cache.state

            module._mlxl3_compiled_decode = mx.compile(run)
        if hasattr(module, "_mlxl3_compiled_stateless"):
            module._mlxl3_compiled_stateless = mx.compile(module._mlxl3_stateless_original_call)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument(
        "--feature", default="window", choices=["hadamard", "window", "gather", "lut"]
    )
    args = parser.parse_args()
    model, tokenizer, *_ = _load_model(args.model)
    prompts = [
        "Explique en détail pourquoi le ciel est bleu et pourquoi le Soleil devient rouge au coucher.",
        "Écris une fonction Python de tri fusion et explique sa complexité et les cas limites.",
    ]
    speeds = {False: [], True: []}
    for pair in range(args.pairs):
        tokens = {}
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[pair % len(prompts)]}],
            tokenize=True,
            add_generation_prompt=True,
        )
        for enabled in [False, True] if pair % 2 == 0 else [True, False]:
            if args.feature == "hadamard":
                qmv._USE_SCALED_HADAMARD = enabled
            elif args.feature == "gather":
                qmv._USE_GATHER_HADAMARD = enabled
            elif args.feature == "lut":
                qmv._USE_CODEBOOK_LUT = enabled
                if enabled:
                    for mode in range(3):
                        qmv._codebook_table(mode)
            else:
                qmv._USE_K24_WINDOW_DECODE = enabled
            refresh(model)
            list(
                stream_generate(
                    model, tokenizer, prompt, max_tokens=8, sampler=make_sampler(temp=0)
                )
            )
            responses = list(
                stream_generate(
                    model, tokenizer, prompt, max_tokens=args.tokens, sampler=make_sampler(temp=0)
                )
            )
            tokens[enabled] = [r.token for r in responses]
            speeds[enabled].append(responses[-1].generation_tps)
        print(
            json.dumps(
                {
                    "pair": pair,
                    "baseline": speeds[False][-1],
                    "candidate": speeds[True][-1],
                    "tokens_equal": tokens[False] == tokens[True],
                    "tokens": len(tokens[False]),
                }
            ),
            flush=True,
        )
        assert tokens[False] == tokens[True], "token divergence"
    print(
        json.dumps(
            {
                "median": {str(k): statistics.median(v) for k, v in speeds.items()},
                "paired_gain_percent": statistics.median(
                    [(b / a - 1) * 100 for a, b in zip(speeds[False], speeds[True])]
                ),
                "peak_gb": mx.get_peak_memory() / 1e9,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
