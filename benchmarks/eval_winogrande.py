"""Small deterministic WinoGrande quality check for two local MLX engines."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.utils import load_tokenizer

from mlxl3.checkpoint import load_exl3_model


def sentence_score(model, tokenizer, sentence: str) -> float:
    tokens = mx.array(tokenizer.encode(sentence), dtype=mx.int32)
    logits = model(tokens[None])[:, :-1].astype(mx.float32)
    targets = tokens[1:]
    target_logits = mx.take_along_axis(logits[0], targets[:, None], axis=-1)[:, 0]
    log_probs = target_logits - mx.logsumexp(logits[0], axis=-1)
    score = mx.mean(log_probs)
    mx.eval(score)
    return float(score.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=("mlxl3", "mlx8"))
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--data", type=Path, default=Path("benchmarks/data/winogrande-debiased-eval.csv")
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.data.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    indices = sorted(random.Random(args.seed).sample(range(len(rows)), args.limit))

    if args.engine == "mlxl3":
        model, _, _ = load_exl3_model(args.model, lazy=False)
        tokenizer = load_tokenizer(str(args.model))
    else:
        model, tokenizer = load(str(args.model), lazy=False)

    started = time.perf_counter()
    predictions = []
    scores = []
    correct = 0
    for position, index in enumerate(indices, start=1):
        _, sentence, option1, option2, answer = rows[index]
        pair = [
            sentence_score(model, tokenizer, sentence.replace("_", option1)),
            sentence_score(model, tokenizer, sentence.replace("_", option2)),
        ]
        prediction = 1 if pair[0] >= pair[1] else 2
        predictions.append(prediction)
        scores.append(pair)
        correct += prediction == int(answer)
        if position % 16 == 0:
            print(f"{args.engine}: {position}/{args.limit}", file=sys.stderr)

    result = {
        "engine": args.engine,
        "model": str(args.model),
        "limit": args.limit,
        "seed": args.seed,
        "indices": indices,
        "accuracy": correct / args.limit,
        "seconds": time.perf_counter() - started,
        "predictions": predictions,
        "scores": scores,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
