"""Capture one warmed decode step for Xcode Metal analysis, not TPS timing.

Run with MTL_CAPTURE_ENABLED=1. Trace may contain model weights and prompt
buffers: keep it local. Viewing GPU counters requires full Xcode.
"""
import argparse
import cProfile
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from mlxl3.cli import _load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--trace', type=Path)
    parser.add_argument('--cpu-profile', type=Path)
    args = parser.parse_args()
    if args.trace is None and args.cpu_profile is None:
        parser.error('provide --trace and/or --cpu-profile')
    for path in (args.trace, args.cpu_profile):
        if path is not None and path.exists():
            raise FileExistsError(path)
    model, tokenizer, *_ = _load_model(args.model)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Explique la gravité.'}],
                                           tokenize=True, add_generation_prompt=True)
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)
    token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
    for _ in range(8):
        logits = model(token, cache=cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
        mx.eval(token)
    profiler = cProfile.Profile()
    if args.trace:
        mx.metal.start_capture(str(args.trace))
    try:
        profiler.enable()
        logits = model(token, cache=cache)
        mx.eval(logits, [c.state for c in cache])
        profiler.disable()
    finally:
        if args.trace:
            mx.metal.stop_capture()
    if args.cpu_profile:
        profiler.dump_stats(str(args.cpu_profile))


if __name__ == '__main__':
    main()
