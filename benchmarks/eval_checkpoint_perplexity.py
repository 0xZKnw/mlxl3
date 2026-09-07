"""Small, fixed-window held-out perplexity check for native MLX and EXL3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import mlx.core as mx


def mean_nll(logits, targets):
    logits = logits.astype(mx.float32)
    chosen = mx.take_along_axis(logits, targets[..., None], axis=-1)[..., 0]
    return mx.mean(mx.logsumexp(logits, axis=-1) - chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--data", type=Path, default=Path("benchmarks/data/wikitext-2-raw/wiki.test.raw")
    )
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens < 1 or args.window < 1:
        parser.error("tokens and window must be positive")
    if (args.model / "quantization_config.json").exists():
        from mlx_lm.utils import load_tokenizer

        from mlxl3.checkpoint import load_exl3_model

        model, _, report = load_exl3_model(args.model, lazy=False)
        if report.skipped:
            raise ValueError(f"Unloaded modules: {report.skipped}")
        tokenizer = load_tokenizer(args.model)
    else:
        from mlx_lm import load

        model, tokenizer = load(str(args.model), lazy=False)
    corpus = args.data.read_bytes()
    ids = tokenizer.encode(corpus.decode("utf-8"))[: args.tokens + 1]
    if len(ids) != args.tokens + 1:
        raise ValueError("Evaluation corpus is too short")
    total_nll = 0.0
    for start in range(0, args.tokens, args.window):
        tokens = mx.array(ids[start : min(start + args.window, args.tokens) + 1])[None]
        logits = model(tokens[:, :-1])
        loss = float(mean_nll(logits, tokens[:, 1:]))
        if not math.isfinite(loss):
            raise ValueError("Non-finite evaluation loss")
        total_nll += loss * (tokens.size - 1)
        del logits, tokens
        mx.clear_cache()
    result = {
        "model": args.model.name,
        "dataset": "WikiText-2 test",
        "dataset_sha256": hashlib.sha256(corpus).hexdigest(),
        "tokens": args.tokens,
        "window": args.window,
        "mean_nll": total_nll / args.tokens,
        "perplexity": math.exp(total_nll / args.tokens),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
